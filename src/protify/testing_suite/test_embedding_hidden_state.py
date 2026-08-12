import os
import sys

import pytest
import torch
from types import SimpleNamespace


PROTIFY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(PROTIFY_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)
for path in (PROTIFY_ROOT, SRC_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


from src.protify.base_models.utils import select_hidden_state
from src.protify.embedder import (
    PUBLISHED_EMBEDDING_MAX_LENGTH,
    Embedder,
    EmbeddingArguments,
    get_embedding_filename,
    legacy_embedding_filename,
)
from src.protify.hyperopt_utils import HyperoptModule


def test_embedding_filename_includes_default_length_cache_identity():
    filename = get_embedding_filename("ESM2-8", False, ["var", "mean"])

    assert filename == "ESM2-8_False_len2048_mean_var.pth"


def test_embedding_filename_adds_hidden_state_suffix_for_non_default():
    filename = get_embedding_filename(
        "ESM2-8",
        False,
        ["var", "mean"],
        hidden_state_index=6,
    )
    db_filename = get_embedding_filename(
        "ESM2-8",
        True,
        ["mean"],
        extension="db",
        hidden_state_index=3,
    )

    assert filename == "ESM2-8_False_hs6_len2048_mean_var.pth"
    assert db_filename == "ESM2-8_True_hs3_len2048.db"


def test_embedding_filename_changes_with_max_length():
    short = get_embedding_filename(
        "ESM2-8",
        False,
        ["mean"],
        max_length=128,
    )
    long = get_embedding_filename(
        "ESM2-8",
        False,
        ["mean"],
        max_length=1024,
    )

    assert short != long


def test_legacy_filename_matches_the_published_download_names():
    # Synthyra/vector_embeddings stores names like AMPLIFY-120_False_mean_var.pth.gz, which
    # predate max_length joining the cache identity.
    assert legacy_embedding_filename("AMPLIFY-120", False, ["var", "mean"]) == "AMPLIFY-120_False_mean_var.pth"
    assert legacy_embedding_filename("ESM2-8", True, ["mean"], "db", 3) == "ESM2-8_True_hs3.db"


def test_download_falls_back_to_the_published_name_only_at_the_published_length(monkeypatch, tmp_path):
    args = EmbeddingArguments(
        embedding_pooling_types=["mean"],
        embedding_save_dir=str(tmp_path),
        max_length=PUBLISHED_EMBEDDING_MAX_LENGTH,
    )
    embedder = Embedder(args, ["MAAA"])

    requested = []

    def record_and_fail(repo_id, filename, repo_type):
        requested.append(filename)
        raise FileNotFoundError(filename)

    monkeypatch.setattr("src.protify.embedder.hf_hub_download", record_and_fail)
    embedder._download_embeddings("ESM2-8")

    assert requested == [
        "embeddings/ESM2-8_False_len2048_mean.pth.gz",
        "embeddings/ESM2-8_False_mean.pth.gz",
    ]

    # A shorter limit means the published file holds more of each sequence than this run
    # asked for, which is exactly what putting max_length in the name prevents.
    requested.clear()
    embedder.max_length = 512
    embedder.args.max_length = 512
    embedder._download_embeddings("ESM2-8")

    assert requested == ["embeddings/ESM2-8_False_len512_mean.pth.gz"]


def test_embedding_arguments_store_hidden_state_index():
    args = EmbeddingArguments(embedding_hidden_state_index=5)

    assert args.hidden_state_index == 5


def test_select_hidden_state_uses_final_state_by_default():
    final_state = torch.ones(2, 3, 4)  # (b=2, l=3, d=4)
    hidden_states = (  # two tensors, each (b, l, d)
        torch.zeros(2, 3, 4),
        torch.full((2, 3, 4), 2.0),
    )

    selected = select_hidden_state(final_state, hidden_states, -1)  # (b, l, d)

    assert selected is final_state


def test_select_hidden_state_uses_requested_tuple_index():
    final_state = torch.ones(2, 3, 4)  # (b=2, l=3, d=4)
    hidden_states = (  # two tensors, each (b, l, d)
        torch.zeros(2, 3, 4),
        torch.full((2, 3, 4), 2.0),
    )

    selected = select_hidden_state(final_state, hidden_states, 1)  # (b, l, d)

    assert torch.equal(selected, hidden_states[1])


def test_select_hidden_state_requires_hidden_states_for_non_default():
    final_state = torch.ones(2, 3, 4)  # (b=2, l=3, d=4)

    with pytest.raises(AssertionError):
        select_hidden_state(final_state, None, 0)


def test_parse_arguments_yaml_hidden_state_cli_override(tmp_path, monkeypatch):
    from src.protify.main import parse_arguments

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "data_names:",
                "  - DeepLoc-2",
                "model_names:",
                "  - ESM2-8",
                "embedding_hidden_state_index: 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--yaml_path",
            str(yaml_path),
            "--embedding_hidden_state_index",
            "4",
        ],
    )

    args = parse_arguments()

    assert args.embedding_hidden_state_index == 4


def test_hyperopt_apply_config_updates_and_restores_embedding_args():
    main_process = SimpleNamespace(
        probe_args=SimpleNamespace(
            hidden_size=128,
            head_size=64,
            pooling_types=["mean"],
        ),
        trainer_args=SimpleNamespace(),
        embedding_args=SimpleNamespace(
            pooling_types=["mean"],
            hidden_state_index=-1,
        ),
        full_args=SimpleNamespace(random_pair_flipping=False),
    )
    module = HyperoptModule(
        main_process,
        "ESM2-8",
        "DeepLoc-2",
        dataset=None,
        emb_dict=None,
        sweep_config={},
        results_list=[],
    )

    module.apply_config(
        {
            "embedding_pooling_types": "mean-var",
            "embedding_hidden_state_index": 6,
        }
    )
    assert main_process.embedding_args.pooling_types == ["mean", "var"]
    assert main_process.embedding_args.hidden_state_index == 6

    module.apply_config({})
    assert main_process.embedding_args.pooling_types == ["mean"]
    assert main_process.embedding_args.hidden_state_index == -1
