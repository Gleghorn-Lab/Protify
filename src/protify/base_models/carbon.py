import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .base_tokenizer import BaseSequenceTokenizer
from .carbon_tokenizer import AuditedCarbonTokenizer
from .utils import select_hidden_state


presets = {
    "CARBON-500M": "HuggingFaceBio/Carbon-500M",
    "CARBON-3B": "HuggingFaceBio/Carbon-3B",
    "CARBON-8B": "HuggingFaceBio/Carbon-8B",
}

# CARBON's tokenizer is custom Python code. Pin the trusted preset revisions so
# routine model loading cannot execute newly published code without a Protify
# update and review. These are the public main revisions verified 2026-07-14.
DEFAULT_REVISIONS = {
    "CARBON-500M": "106e36ff51b5dfbfe0b078ad18ad37a6956c5714",
    "CARBON-3B": "96ff92c0cfa5ae1d72ba64abce07804fd1203ecc",
    "CARBON-8B": "f77b7d60eaf767778dbd53482f56e70ad2558ecd",
}

DNA_OPEN_TOKEN = "<dna>"
DNA_CLOSE_TOKEN = "</dna>"
DNA_KMER_SIZE = 6
DNA_BOUNDARY_TOKEN_COUNT = 2
QWEN_TOKENIZER_MODEL = "Qwen/Qwen3-4B-Base"
QWEN_TOKENIZER_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
CARBON_PREPROCESSING_SCHEMA = (
    "carbon_dna6_v3_upper_prefix_partialA_"
    f"qwen_{QWEN_TOKENIZER_REVISION[:12]}"
)
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def carbon_dna_length_for_tokens(max_length: int) -> int:
    """Return the usable DNA base-pair budget for a CARBON token budget."""
    if not isinstance(max_length, int) or isinstance(max_length, bool):
        raise TypeError("max_length must be an integer token count")
    if max_length <= DNA_BOUNDARY_TOKEN_COUNT:
        raise ValueError(
            f"CARBON max_length must be at least {DNA_BOUNDARY_TOKEN_COUNT + 1} "
            "tokens (opening tag, at least one 6-mer, and closing tag)"
        )
    return (max_length - DNA_BOUNDARY_TOKEN_COUNT) * DNA_KMER_SIZE


def _validate_pinned_revision(revision: str) -> str:
    if not _COMMIT_RE.fullmatch(revision):
        raise ValueError(
            "Remote CARBON revisions must be immutable 40-character commit hashes; "
            f"got {revision!r}"
        )
    return revision.lower()


def resolve_carbon_source(
    preset: str,
    model_path: Optional[str] = None,
    revision: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve a local model path or a revision-pinned Hub model reference.

    A custom remote model can be supplied as ``org/repo@<40-char commit>`` or by
    passing ``revision`` explicitly. Local paths are trusted as user-controlled
    input and do not need a Hub revision.
    """
    if model_path is None:
        if preset not in presets:
            raise ValueError(f"Unknown CARBON preset: {preset}")
        resolved_revision = revision or DEFAULT_REVISIONS[preset]
        return presets[preset], _validate_pinned_revision(resolved_revision)

    expanded_path = Path(os.path.expanduser(model_path))
    if expanded_path.exists():
        if revision is not None:
            raise ValueError("revision cannot be used with a local CARBON model path")
        return str(expanded_path), None

    inline_revision = None
    remote_path = model_path
    if "@" in model_path:
        remote_path, inline_revision = model_path.rsplit("@", 1)
        if not remote_path:
            raise ValueError("CARBON model path before '@' cannot be empty")

    if revision is not None and inline_revision is not None and revision != inline_revision:
        raise ValueError("Conflicting CARBON revisions were provided")
    resolved_revision = revision or inline_revision
    if resolved_revision is None:
        raise ValueError(
            "Remote CARBON model_path values must be pinned as "
            "'org/repo@<40-character commit>'"
        )
    return remote_path, _validate_pinned_revision(resolved_revision)


def _revision_kwargs(revision: Optional[str]) -> Dict[str, str]:
    return {} if revision is None else {"revision": revision}


class CarbonTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        # Right padding is the safest default for CARBON's closing DNA boundary
        # token and its Llama sequence-classification head.
        self.tokenizer.padding_side = "right"
        vocab = self.tokenizer.get_vocab()
        missing = [token for token in (DNA_OPEN_TOKEN, DNA_CLOSE_TOKEN) if token not in vocab]
        if missing:
            raise ValueError(f"CARBON tokenizer is missing required tokens: {missing}")
        self.dna_open_token_id = int(self.tokenizer.convert_tokens_to_ids(DNA_OPEN_TOKEN))
        self.dna_close_token_id = int(self.tokenizer.convert_tokens_to_ids(DNA_CLOSE_TOKEN))
        # Pooler uses this explicit boundary token when 'eos' pooling is requested.
        self.pooling_token_id = self.dna_close_token_id

    @staticmethod
    def _prepare_sequence(sequence: str, char_budget: Optional[int]) -> str:
        if not isinstance(sequence, str):
            raise TypeError(f"CARBON sequences must be strings, got {type(sequence).__name__}")
        lowered = sequence.lower()
        if DNA_OPEN_TOKEN in lowered or DNA_CLOSE_TOKEN in lowered:
            raise ValueError("Pass raw DNA to CARBON; Protify adds the <dna> boundaries")
        sequence = sequence.upper()
        if char_budget is not None:
            sequence = sequence[:char_budget]
        # The published CARBON tokenizer preserves a final 1-5 bp remainder and
        # right-pads its token with A while tracking the real-base count.
        return sequence

    @staticmethod
    def _as_batched_tensor(value: object, field_name: str) -> torch.Tensor:
        try:
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"CARBON tokenizer returned invalid {field_name}") from exc
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"CARBON tokenizer {field_name} must be rank 2, got shape {tuple(tensor.shape)}"
            )
        return tensor

    def _validate_boundaries(self, encoded: Dict[str, object]) -> None:
        if "input_ids" not in encoded:
            raise ValueError("CARBON tokenizer did not return input_ids")
        input_ids = self._as_batched_tensor(encoded["input_ids"], "input_ids")
        if "attention_mask" in encoded:
            attention_mask = self._as_batched_tensor(encoded["attention_mask"], "attention_mask")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("CARBON input_ids and attention_mask shapes do not match")
            active = attention_mask.to(dtype=torch.bool)
        else:
            active = torch.ones_like(input_ids, dtype=torch.bool)

        for token_name, token_id in (
            (DNA_OPEN_TOKEN, self.dna_open_token_id),
            (DNA_CLOSE_TOKEN, self.dna_close_token_id),
        ):
            counts = ((input_ids == token_id) & active).sum(dim=1)
            invalid = torch.nonzero(counts != 1, as_tuple=False).flatten().tolist()
            if invalid:
                observed = [int(counts[index].item()) for index in invalid]
                raise ValueError(
                    f"CARBON tokenization requires exactly one active {token_name} token per "
                    f"sample; samples {invalid} had counts {observed}"
                )

    def _post_pad_max_length(self, encoded: Dict[str, object], max_length: int) -> None:
        """Force exact right padding despite the upstream tokenizer's short padding."""
        input_ids = self._as_batched_tensor(encoded["input_ids"], "input_ids")
        current_length = input_ids.shape[1]
        if current_length > max_length:
            raise ValueError(
                f"CARBON tokenizer returned {current_length} tokens for max_length={max_length}"
            )
        if current_length == max_length:
            return

        pad_length = max_length - current_length
        input_padding = torch.full(
            (input_ids.shape[0], pad_length),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        encoded["input_ids"] = torch.cat((input_ids, input_padding), dim=1)

        if "attention_mask" in encoded:
            attention_mask = self._as_batched_tensor(
                encoded["attention_mask"], "attention_mask"
            )
        else:
            attention_mask = torch.ones_like(input_ids)
        mask_padding = torch.zeros(
            (attention_mask.shape[0], pad_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        encoded["attention_mask"] = torch.cat((attention_mask, mask_padding), dim=1)

    def __call__(
        self,
        sequences: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        else:
            sequences = list(sequences)
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "max_length")
        kwargs.setdefault("truncation", True)
        kwargs["add_special_tokens"] = False

        max_length = kwargs.get("max_length")
        char_budget = None
        if kwargs["truncation"] and max_length is not None:
            char_budget = carbon_dna_length_for_tokens(max_length)

        prepared = [self._prepare_sequence(sequence, char_budget) for sequence in sequences]
        wrapped = [f"{DNA_OPEN_TOKEN}{sequence}{DNA_CLOSE_TOKEN}" for sequence in prepared]
        encoded = self.tokenizer(wrapped, **kwargs)
        if kwargs["padding"] == "max_length" and max_length is not None:
            self._post_pad_max_length(encoded, max_length)
        self._validate_boundaries(encoded)
        return encoded


class CarbonForEmbedding(nn.Module):
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = None,
        revision: Optional[str] = None,
    ):
        super().__init__()
        # CARBON is a stock Llama backbone. AutoModel avoids allocating the
        # unused causal-language-model head for embedding extraction.
        self.carbon = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            **_revision_kwargs(revision),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        hidden_state_index: int = -1,
        **kwargs,
    ) -> torch.Tensor:
        # input_ids: (b, l); attention_mask: (b, l); l counts 6-mer tokens plus two boundaries
        if output_attentions:
            raise ValueError("CARBON does not return attentions; 'parti' pooling is unsupported")
        needs_hidden_states = hidden_state_index != -1
        out = self.carbon(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=needs_hidden_states,
        )
        return select_hidden_state(  # (b, l, d)
            out.last_hidden_state,  # (b, l, d)
            out.hidden_states if needs_hidden_states else None,
            hidden_state_index,
        )


def _check_vocabulary_agreement(model: nn.Module, tokenizer: BaseSequenceTokenizer, source: str) -> None:
    """Fail loudly when the checkpoint was not built against the vocabulary we reconstruct.

    Protify rebuilds CARBON's DNA vocabulary locally instead of running the model
    repository's tokenizer code, so a checkpoint with a different k, base vocabulary, or
    128-alignment would index the wrong embedding rows and return confident nonsense.
    """
    model_vocab_size = getattr(model.config, "vocab_size", None)
    if model_vocab_size is not None and model_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"CARBON checkpoint {source} has vocab_size {model_vocab_size}, but Protify's "
            f"audited tokenizer builds {tokenizer.vocab_size} tokens. The checkpoint was not "
            "produced with the pinned Qwen vocabulary and 6-mer layout this module reproduces."
        )


def get_carbon_tokenizer(
    preset: str,
    model_path: str = None,
    revision: Optional[str] = None,
) -> BaseSequenceTokenizer:
    source, resolved_revision = resolve_carbon_source(preset, model_path, revision)
    base_tokenizer = AutoTokenizer.from_pretrained(
        QWEN_TOKENIZER_MODEL,
        revision=QWEN_TOKENIZER_REVISION,
    )
    tokenizer = AuditedCarbonTokenizer(
        base_tokenizer,
        base_model_id=QWEN_TOKENIZER_MODEL,
        base_revision=QWEN_TOKENIZER_REVISION,
        preprocessing_schema=CARBON_PREPROCESSING_SCHEMA,
    )
    wrapped = CarbonTokenizerWrapper(tokenizer)
    wrapped.model_source = source
    wrapped.model_revision = resolved_revision
    wrapped.preprocessing_schema = CARBON_PREPROCESSING_SCHEMA
    return wrapped


def build_carbon_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
    revision: Optional[str] = None,
    **kwargs,
) -> Tuple[nn.Module, BaseSequenceTokenizer]:
    if masked_lm:
        raise ValueError(f"Model {preset} does not support masked language modeling")
    source, resolved_revision = resolve_carbon_source(preset, model_path, revision)
    model = CarbonForEmbedding(source, dtype=dtype, revision=resolved_revision).eval()
    tokenizer = get_carbon_tokenizer(preset, model_path=source, revision=resolved_revision)
    _check_vocabulary_agreement(model.carbon, tokenizer, source)
    return model, tokenizer


def get_carbon_for_training(
    preset: str,
    tokenwise: bool = False,
    num_labels: int = None,
    hybrid: bool = False,
    dtype: torch.dtype = None,
    model_path: str = None,
    revision: Optional[str] = None,
):
    if tokenwise:
        raise NotImplementedError(
            "CARBON uses 6-mer DNA tokens (one token spans 6 nt), so per-token labels do "
            "not align; only sequence-level fine-tuning is supported."
        )
    source, resolved_revision = resolve_carbon_source(preset, model_path, revision)
    tokenizer = get_carbon_tokenizer(preset, model_path=source, revision=resolved_revision)
    load_kwargs = {"dtype": dtype, **_revision_kwargs(resolved_revision)}
    if hybrid:
        model = AutoModel.from_pretrained(source, **load_kwargs).eval()
    else:
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(source, **load_kwargs).eval()
        # Llama sequence classification needs the pad id to identify the final
        # active token in batches. CARBON's published config may leave it unset.
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    _check_vocabulary_agreement(model, tokenizer, source)
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = build_carbon_model("CARBON-500M")
    print(model)
    print(tokenizer)
    print(tokenizer("ATCG" * 30))
