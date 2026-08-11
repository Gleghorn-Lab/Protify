from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

try:
    from src.protify.base_models import carbon
    from src.protify.base_models import get_base_models as base_model_dispatch
    from src.protify.base_models.get_base_models import get_raw_sequence_length_limit, resolve_data_max_length
    from src.protify.data.data_mixin import DataArguments, DataMixin
    from src.protify.embedder import get_embedding_filename
    from src.protify.base_models.supported_models import gui_models
except ImportError:
    try:
        from protify.base_models import carbon
        from protify.base_models import get_base_models as base_model_dispatch
        from protify.base_models.get_base_models import get_raw_sequence_length_limit, resolve_data_max_length
        from protify.data.data_mixin import DataArguments, DataMixin
        from protify.embedder import get_embedding_filename
        from protify.base_models.supported_models import gui_models
    except ImportError:
        from ..base_models import carbon
        from ..base_models import get_base_models as base_model_dispatch
        from ..base_models.get_base_models import get_raw_sequence_length_limit, resolve_data_max_length
        from ..data.data_mixin import DataArguments, DataMixin
        from ..embedder import get_embedding_filename
        from ..base_models.supported_models import gui_models


class FakeCarbonTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    cls_token_id = None
    mask_token_id = None
    vocab_size = 128

    def __init__(self):
        self.padding_side = "left"
        self.last_texts = None
        self.last_kwargs = None
        self._vocab = {
            carbon.DNA_OPEN_TOKEN: 10,
            carbon.DNA_CLOSE_TOKEN: 11,
        }

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]

    def __call__(self, texts, **kwargs):
        self.last_texts = list(texts)
        self.last_kwargs = dict(kwargs)
        rows = []
        for text in texts:
            assert text.startswith(carbon.DNA_OPEN_TOKEN)
            assert text.endswith(carbon.DNA_CLOSE_TOKEN)
            dna = text[len(carbon.DNA_OPEN_TOKEN):-len(carbon.DNA_CLOSE_TOKEN)]
            n_kmers = (len(dna) + carbon.DNA_KMER_SIZE - 1) // carbon.DNA_KMER_SIZE
            ids = [10] + [20] * n_kmers + [11]
            max_length = kwargs.get("max_length")
            if kwargs.get("truncation") and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)

        if kwargs.get("padding") == "max_length":
            target_length = kwargs["max_length"]
        elif kwargs.get("padding"):
            target_length = max(len(row) for row in rows)
        else:
            target_length = None

        masks = []
        if target_length is not None:
            padded_rows = []
            for row in rows:
                padding = [self.pad_token_id] * (target_length - len(row))
                if self.padding_side == "left":
                    padded_rows.append(padding + row)
                    masks.append([0] * len(padding) + [1] * len(row))
                else:
                    padded_rows.append(row + padding)
                    masks.append([1] * len(row) + [0] * len(padding))
            rows = padded_rows
        else:
            masks = [[1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class ShortMaxPaddingTokenizer(FakeCarbonTokenizer):
    """Match the upstream bug: max_length padding only pads to longest input."""

    def __call__(self, texts, **kwargs):
        short_kwargs = dict(kwargs)
        if short_kwargs.get("padding") == "max_length":
            short_kwargs["padding"] = True
        return super().__call__(texts, **short_kwargs)


class FakeQwenTokenizer:
    pad_token_id = 1
    eos_token_id = 1
    cls_token_id = None
    mask_token_id = None
    sep_token_id = None

    def __init__(self):
        self._vocab = {"A": 0, "<|endoftext|>": 1}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, 0)

    def save_pretrained(self, save_directory):
        return (str(Path(save_directory) / "tokenizer.json"),)


class BadBoundaryTokenizer(FakeCarbonTokenizer):
    def __call__(self, texts, **kwargs):
        encoded = super().__call__(texts, **kwargs)
        encoded["input_ids"][encoded["input_ids"] == self._vocab[carbon.DNA_CLOSE_TOKEN]] = 20
        return encoded


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=None)
        self.output_hidden_states_calls = []

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        self.output_hidden_states_calls.append(output_hidden_states)
        first = input_ids.unsqueeze(-1).float()
        last = first + 100
        hidden_states = (first, last) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=last, hidden_states=hidden_states)


def test_carbon_token_budget_maps_to_six_mer_dna_budget():
    assert carbon.carbon_dna_length_for_tokens(2048) == 12_276
    assert get_raw_sequence_length_limit("CARBON-500M", 2048) == 12_276
    assert get_raw_sequence_length_limit("ESM2-8", 2048) == 2048
    with pytest.raises(ValueError, match="at least 3"):
        carbon.carbon_dna_length_for_tokens(2)


def test_data_mixin_uses_carbon_raw_sequence_budget():
    args = DataArguments(data_names=[], max_length=8)
    mixin = DataMixin(args)
    mixin._data_max_length = get_raw_sequence_length_limit("CARBON-500M", 8)
    split = Dataset.from_dict({"seqs": ["A" * 60, "C" * 12], "labels": [0, 1]})

    datasets, all_seqs = mixin.process_datasets(
        [(split, split, split, False)],
        ["dna"],
    )

    train_set = datasets["dna"][0]
    assert len(train_set["seqs"][0]) == 36
    assert max(map(len, all_seqs)) == 36


def test_data_mixin_preserves_and_uppercases_lowercase_dna():
    args = DataArguments(data_names=[], max_length=12)
    mixin = DataMixin(args)
    split = Dataset.from_dict({"seqs": ["acgtacg"], "labels": [0]})

    datasets, all_seqs = mixin.process_datasets(
        [(split, split, split, False)],
        ["lowercase_dna"],
    )

    assert datasets["lowercase_dna"][0]["seqs"] == ["ACGTACG"]
    assert all_seqs == ["ACGTACG"]


@pytest.mark.parametrize(
    "translation_flag,source,max_length,expected",
    [
        ("aa_to_dna", "MKT", 6, "ATGAAG"),
        ("dna_to_aa", "ATGAAGACT", 2, "MK"),
    ],
)
def test_length_enforcement_happens_after_translation(
    translation_flag,
    source,
    max_length,
    expected,
):
    kwargs = {translation_flag: True}
    args = DataArguments(data_names=[], max_length=max_length, **kwargs)
    mixin = DataMixin(args)
    split = Dataset.from_dict({"seqs": [source], "labels": [0]})

    datasets, _ = mixin.process_datasets(
        [(split, split, split, False)],
        [translation_flag],
    )

    assert datasets[translation_flag][0]["seqs"] == [expected]


def test_tokenizer_preserves_full_token_budget_and_boundaries():
    backend = FakeCarbonTokenizer()
    tokenizer = carbon.CarbonTokenizerWrapper(backend)
    sequence = "AAAAAA" * 6 + "CCCCCC" * 6

    encoded = tokenizer(sequence, max_length=8)

    assert tokenizer.pooling_token_id == 11
    assert backend.padding_side == "right"
    assert backend.last_texts == [f"<dna>{'A' * 36}</dna>"]
    assert encoded["attention_mask"].sum().item() == 8
    assert encoded["input_ids"][0, -1].item() == 11
    assert backend.last_kwargs["add_special_tokens"] is False


def test_tokenizer_uppercases_and_preserves_partial_six_mer():
    backend = FakeCarbonTokenizer()
    tokenizer = carbon.CarbonTokenizerWrapper(backend)

    tokenizer("acgtacg", max_length=8)

    assert backend.last_texts == ["<dna>ACGTACG</dna>"]


def test_tokenizer_post_pads_upstream_short_max_length_batches():
    tokenizer = carbon.CarbonTokenizerWrapper(ShortMaxPaddingTokenizer())

    encoded = tokenizer(["ACGTAC", "ACGTAC" * 3], max_length=8)

    assert encoded["input_ids"].shape == (2, 8)
    assert encoded["attention_mask"].sum(dim=1).tolist() == [3, 5]
    assert encoded["input_ids"][0, 3:].tolist() == [tokenizer.pad_token_id] * 5


def test_tokenizer_rejects_user_boundaries_and_missing_close_token():
    tokenizer = carbon.CarbonTokenizerWrapper(FakeCarbonTokenizer())
    with pytest.raises(ValueError, match="raw DNA"):
        tokenizer("<dna>ACGTAC</dna>", max_length=8)

    tokenizer = carbon.CarbonTokenizerWrapper(BadBoundaryTokenizer())
    with pytest.raises(ValueError, match="exactly one active </dna>"):
        tokenizer("ACGTAC", max_length=8)


def test_tokenizer_requires_required_vocab_tokens():
    backend = FakeCarbonTokenizer()
    del backend._vocab[carbon.DNA_CLOSE_TOKEN]
    with pytest.raises(ValueError, match="missing required tokens"):
        carbon.CarbonTokenizerWrapper(backend)


def test_default_hub_sources_are_commit_pinned():
    source, revision = carbon.resolve_carbon_source("CARBON-500M")
    assert source == carbon.presets["CARBON-500M"]
    assert revision == carbon.DEFAULT_REVISIONS["CARBON-500M"]


def test_carbon_cache_identity_includes_length_revision_and_preprocessing():
    filename = get_embedding_filename(
        "CARBON-500M",
        False,
        ["mean"],
        max_length=2048,
    )
    shorter = get_embedding_filename(
        "CARBON-500M",
        False,
        ["mean"],
        max_length=1024,
    )
    custom = get_embedding_filename(
        "custom-carbon",
        False,
        ["mean"],
        max_length=2048,
        model_type="carbon",
        model_path=f"example/carbon@{'b' * 40}",
    )

    assert "_len2048_" in filename
    assert f"_rev{carbon.DEFAULT_REVISIONS['CARBON-500M']}_" in filename
    assert f"_prep{carbon.CARBON_PREPROCESSING_SCHEMA}_" in filename
    assert filename != shorter
    assert f"_rev{'b' * 40}_" in custom


def test_gui_model_picker_exposes_all_carbon_presets():
    assert set(carbon.presets).issubset(gui_models)


def test_custom_remote_sources_must_be_commit_pinned():
    with pytest.raises(ValueError, match="must be pinned"):
        carbon.resolve_carbon_source("carbon", model_path="example/carbon")

    revision = "a" * 40
    assert carbon.resolve_carbon_source(
        "carbon",
        model_path=f"example/carbon@{revision}",
    ) == ("example/carbon", revision)
    with pytest.raises(ValueError, match="40-character"):
        carbon.resolve_carbon_source("carbon", model_path="example/carbon@main")


def test_local_model_source_does_not_require_revision(tmp_path: Path):
    assert carbon.resolve_carbon_source(
        "carbon",
        model_path=str(tmp_path),
    ) == (str(tmp_path), None)
    with pytest.raises(ValueError, match="local"):
        carbon.resolve_carbon_source(
            "carbon",
            model_path=str(tmp_path),
            revision="a" * 40,
        )


def test_tokenizer_remote_code_is_pinned_but_model_remote_code_is_disabled():
    backend = FakeQwenTokenizer()
    revision = carbon.DEFAULT_REVISIONS["CARBON-500M"]
    with patch.object(carbon.AutoTokenizer, "from_pretrained", return_value=backend) as load_tokenizer:
        tokenizer = carbon.get_carbon_tokenizer("CARBON-500M")
    assert tokenizer.model_revision == revision
    assert tokenizer.preprocessing_schema == carbon.CARBON_PREPROCESSING_SCHEMA
    load_tokenizer.assert_called_once_with(
        carbon.QWEN_TOKENIZER_MODEL,
        revision=carbon.QWEN_TOKENIZER_REVISION,
    )

    fake_model = FakeModel()
    with patch.object(
        carbon.AutoModel,
        "from_pretrained",
        return_value=fake_model,
    ) as load_model:
        model = carbon.CarbonForEmbedding("example/carbon", revision=revision)
    _, kwargs = load_model.call_args
    assert kwargs["revision"] == revision
    assert "trust_remote_code" not in kwargs
    assert model.carbon is fake_model


def test_embedding_model_hidden_state_selection_and_attentions_error():
    with patch.object(
        carbon.AutoModel,
        "from_pretrained",
        return_value=FakeModel(),
    ) as load_model:
        model = carbon.CarbonForEmbedding("local")
    input_ids = torch.tensor([[1, 2, 3]])

    assert torch.equal(model(input_ids), input_ids.unsqueeze(-1).float() + 100)
    assert model.carbon.output_hidden_states_calls == [False]
    assert torch.equal(
        model(input_ids, hidden_state_index=0),
        input_ids.unsqueeze(-1).float(),
    )
    assert model.carbon.output_hidden_states_calls == [False, True]
    _, load_kwargs = load_model.call_args
    assert "trust_remote_code" not in load_kwargs
    with pytest.raises(ValueError, match="parti"):
        model(input_ids, output_attentions=True)


@pytest.mark.slow
def test_real_pinned_tokenizer_preserves_partial_kmer_and_exact_max_padding():
    tokenizer = carbon.get_carbon_tokenizer("CARBON-500M")

    encoded = tokenizer(["acgtacg", "ACGTAC" * 4], max_length=16)

    assert encoded["input_ids"].shape == (2, 16)
    assert encoded["attention_mask"].sum(dim=1).tolist() == [4, 6]
    assert tokenizer.tokenizer.base_revision == carbon.QWEN_TOKENIZER_REVISION
    assert tokenizer.dna_open_token_id == 151_669
    partial_id = tokenizer.tokenizer.dna_token_to_id["GAAAAA"]
    assert encoded["input_ids"][0, 2].item() == partial_id
    assert torch.equal(
        encoded["input_ids"][0],
        tokenizer(["ACGTACG"], max_length=16)["input_ids"][0],
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_training_models_do_not_enable_remote_code(hybrid):
    backend = FakeQwenTokenizer()
    fake_model = FakeModel()
    model_loader = carbon.AutoModel if hybrid else carbon.AutoModelForSequenceClassification
    with patch.object(carbon.AutoTokenizer, "from_pretrained", return_value=backend), patch.object(
        model_loader,
        "from_pretrained",
        return_value=fake_model,
    ) as load_model:
        model, tokenizer = carbon.get_carbon_for_training(
            "CARBON-500M",
            hybrid=hybrid,
            num_labels=3,
        )

    _, kwargs = load_model.call_args
    assert kwargs["revision"] == carbon.DEFAULT_REVISIONS["CARBON-500M"]
    assert "trust_remote_code" not in kwargs
    if hybrid:
        assert "num_labels" not in kwargs
    else:
        assert kwargs["num_labels"] == 3
        assert model.config.pad_token_id == tokenizer.pad_token_id


def test_tokenwise_training_is_rejected_before_loading():
    with pytest.raises(NotImplementedError, match="sequence-level"):
        carbon.get_carbon_for_training("CARBON-500M", tokenwise=True)


def test_masked_lm_is_rejected_before_loading():
    with pytest.raises(ValueError, match="masked language"):
        carbon.build_carbon_model("CARBON-500M", masked_lm=True)


def test_carbon_dispatch_covers_embedding_training_and_tokenizer_paths():
    sentinel_model = object()
    sentinel_tokenizer = object()
    with patch.object(
        carbon,
        "build_carbon_model",
        return_value=(sentinel_model, sentinel_tokenizer),
    ) as build_model:
        assert base_model_dispatch.get_base_model("CARBON-500M") == (
            sentinel_model,
            sentinel_tokenizer,
        )
    build_model.assert_called_once_with(
        "CARBON-500M",
        masked_lm=False,
        dtype=None,
        model_path=None,
    )

    with patch.object(
        carbon,
        "get_carbon_for_training",
        return_value=(sentinel_model, sentinel_tokenizer),
    ) as build_training_model:
        assert base_model_dispatch.get_base_model_for_training(
            "CARBON-3B",
            num_labels=2,
        ) == (sentinel_model, sentinel_tokenizer)
    build_training_model.assert_called_once_with(
        "CARBON-3B",
        False,
        2,
        False,
        dtype=None,
        model_path=None,
    )

    with patch.object(
        carbon,
        "get_carbon_tokenizer",
        return_value=sentinel_tokenizer,
    ) as build_tokenizer:
        assert base_model_dispatch.get_tokenizer("CARBON-8B") is sentinel_tokenizer
    build_tokenizer.assert_called_once_with("CARBON-8B", model_path=None)


def test_raw_sequence_budget_combines_models_and_survives_an_empty_selection():
    carbon_budget = get_raw_sequence_length_limit("CARBON-500M", 2048)  # 12276

    # Truncating: keep the longest raw input any tokenizer can use, and let each tokenizer
    # cut its own batch down. Trimming: rows are dropped, so only what fits everything survives.
    assert resolve_data_max_length(["CARBON-500M", "ESM2-8"], 2048, trim=False) == carbon_budget
    assert resolve_data_max_length(["CARBON-500M", "ESM2-8"], 2048, trim=True) == 2048
    assert resolve_data_max_length(["ESM2-8"], 2048, trim=False) == 2048
    # 'exp' resolves to the empty experimental_models list, and min()/max() of nothing raises.
    assert resolve_data_max_length([], 2048, trim=False) == 2048
    assert resolve_data_max_length([], 2048, trim=True) == 2048


def test_carbon_lora_targets_do_not_leak_into_other_models():
    try:
        from src.protify.base_models.utils import LLAMA_LORA_TARGET_MODULES, LORA_TARGET_MODULES
    except ImportError:
        try:
            from protify.base_models.utils import LLAMA_LORA_TARGET_MODULES, LORA_TARGET_MODULES
        except ImportError:
            from ..base_models.utils import LLAMA_LORA_TARGET_MODULES, LORA_TARGET_MODULES

    # PEFT matches by name suffix, so a global union would attach adapters inside any other
    # Llama-derived checkpoint loaded through --model_types custom.
    assert not set(LORA_TARGET_MODULES) & set(LLAMA_LORA_TARGET_MODULES)
    assert "q_proj" in LLAMA_LORA_TARGET_MODULES
    assert "q_proj" not in LORA_TARGET_MODULES


def test_tokenizer_honors_pad_to_multiple_of_under_longest_padding():
    # build_collator adds pad_to_multiple_of=8 whenever padding is not 'max_length', so
    # without this CARBON batches would be the only ones that are not 8-aligned.
    backend = carbon.AuditedCarbonTokenizer(FakeQwenTokenizer(), "q", "r", "schema")
    encoded = backend(
        ["<dna>" + "ACGTAC" * 2 + "</dna>"],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    # 1 open + 2 six-mers + 1 close = 4 tokens, rounded up to the next multiple of 8.
    assert encoded["input_ids"].shape == (1, 8)
    assert encoded["attention_mask"].sum().item() == 4


def test_vocabulary_disagreement_between_checkpoint_and_tokenizer_is_fatal():
    tokenizer = carbon.CarbonTokenizerWrapper(FakeCarbonTokenizer())
    matching = SimpleNamespace(config=SimpleNamespace(vocab_size=tokenizer.vocab_size))
    carbon._check_vocabulary_agreement(matching, tokenizer, "example/carbon")

    mismatched = SimpleNamespace(config=SimpleNamespace(vocab_size=tokenizer.vocab_size + 128))
    with pytest.raises(ValueError, match="audited tokenizer"):
        carbon._check_vocabulary_agreement(mismatched, tokenizer, "example/carbon")


def test_audited_vocabulary_matches_the_published_carbon_layout():
    # Values read from HuggingFaceBio/Carbon-500M at the pinned revision: dna_config.json
    # reports dna_start_id 151669 and dna_vocab_size 4107, and config.json vocab_size 155776.
    # Qwen3-4B-Base contributes 151669 base tokens.
    base_vocab_size = 151_669
    dna_special_tokens = 3
    kmers = 4 ** carbon.DNA_KMER_SIZE
    unpadded = base_vocab_size + dna_special_tokens + kmers

    assert ((unpadded + 127) // 128) * 128 == 155_776
    assert dna_special_tokens + kmers + (155_776 - unpadded) == 4_107
