import os
import sys

import pytest
import torch


PROTIFY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(PROTIFY_ROOT)
REPO_ROOT = os.path.dirname(SRC_ROOT)
for path in (PROTIFY_ROOT, SRC_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


from src.protify.embedder import Embedder, EmbeddingArguments, get_embedding_filename


RUN_GPU_SMOKE = os.environ.get("PROTIFY_GPU_SMOKE") == "1"
RUN_GPU_6B_SMOKE = os.environ.get("PROTIFY_GPU_6B_SMOKE") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_GPU_SMOKE and not RUN_GPU_6B_SMOKE,
    reason="Set PROTIFY_GPU_SMOKE=1 to run model-loading GPU embedding smokes.",
)


@pytest.mark.parametrize(
    "model_name,hidden_state_index",
    [
        ("ESM2-8", -1),
        ("ESM2-8", 1),
        ("ESMC-300", -1),
        ("ESMC-300", 1),
    ],
)
def test_gpu_embedding_hidden_state_smoke(tmp_path, model_name, hidden_state_index):
    if not RUN_GPU_SMOKE:
        pytest.skip("Set PROTIFY_GPU_SMOKE=1 to run standard GPU embedding smokes.")
    assert torch.cuda.is_available(), "GPU smoke requires CUDA."

    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQ",
        "GASGDLV",
    ]
    args = EmbeddingArguments(
        embedding_batch_size=2,
        embedding_num_workers=0,
        matrix_embed=False,
        embedding_pooling_types=["mean"],
        save_embeddings=True,
        embed_dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        sql=False,
        embedding_save_dir=str(tmp_path),
        padding="longest",
        max_length=64,
        embedding_hidden_state_index=hidden_state_index,
        autocast=True,
    )

    embeddings = Embedder(args, sequences)(model_name)  # sequence -> (d,)
    filename = get_embedding_filename(
        model_name,
        False,
        ["mean"],
        hidden_state_index=hidden_state_index,
        max_length=64,
    )
    save_path = tmp_path / filename

    assert save_path.exists()
    assert set(embeddings) == set(sequences)
    for sequence in sequences:
        embedding = embeddings[sequence]  # (d,)
        assert embedding.ndim == 1
        assert embedding.numel() > 0


def test_gpu_embedding_esmc_6b_smoke(tmp_path):
    if not RUN_GPU_6B_SMOKE:
        pytest.skip("Set PROTIFY_GPU_6B_SMOKE=1 to run the ESMC-6B smoke.")
    assert torch.cuda.is_available(), "GPU smoke requires CUDA."

    sequences = ["MKTAYIAKQRQISFVKSHFSRQ"]
    args = EmbeddingArguments(
        embedding_batch_size=1,
        embedding_num_workers=0,
        matrix_embed=False,
        embedding_pooling_types=["mean"],
        save_embeddings=True,
        embed_dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        sql=False,
        embedding_save_dir=str(tmp_path),
        padding="longest",
        max_length=64,
        embedding_hidden_state_index=-1,
        autocast=True,
    )

    embeddings = Embedder(args, sequences)("ESMC-6B")  # sequence -> (d,)
    filename = get_embedding_filename("ESMC-6B", False, ["mean"], max_length=64)
    save_path = tmp_path / filename

    assert save_path.exists()
    assert set(embeddings) == set(sequences)
    embedding = embeddings[sequences[0]]  # (d,)
    assert embedding.ndim == 1
    assert embedding.numel() > 0
