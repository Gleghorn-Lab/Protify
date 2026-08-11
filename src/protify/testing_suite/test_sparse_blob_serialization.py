"""Coordinate-list blob storage and bfloat16 fidelity for embedding blobs."""

import struct

import torch

try:
    from src.protify.utils import (
        _COMPACT_VERSION,
        _SPARSE_VERSION,
        batch_tensor_to_blobs,
        embedding_blob_to_tensor,
        tensor_to_embedding_blob,
    )
except ImportError:
    try:
        from protify.utils import (
            _COMPACT_VERSION,
            _SPARSE_VERSION,
            batch_tensor_to_blobs,
            embedding_blob_to_tensor,
            tensor_to_embedding_blob,
        )
    except ImportError:
        from ..utils import (
            _COMPACT_VERSION,
            _SPARSE_VERSION,
            batch_tensor_to_blobs,
            embedding_blob_to_tensor,
            tensor_to_embedding_blob,
        )


def _sparse_vector(width: int, density: float, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """A codebook-shaped vector with a fixed fraction of non-zero entries."""
    torch.manual_seed(width)
    vector = torch.zeros(width, dtype=dtype)  # (w,)
    active = torch.randperm(width)[: int(density * width)]  # (nnz,)
    vector[active] = torch.rand(len(active)).to(dtype) + 0.1  # (nnz,)
    return vector  # (w,)


def test_sparse_flag_selects_the_coordinate_list_format() -> None:
    # Measured max-pooled density for ESMC SAE features at codebook width 131072.
    blob = tensor_to_embedding_blob(_sparse_vector(131072, 0.036), sparse=True)

    assert blob[0] == _SPARSE_VERSION


def test_embeddings_stay_dense_unless_asked() -> None:
    # A sparse tensor still stores dense when the model did not request coordinates.
    assert tensor_to_embedding_blob(torch.randn(1280))[0] == _COMPACT_VERSION
    assert tensor_to_embedding_blob(_sparse_vector(131072, 0.036))[0] == _COMPACT_VERSION


def test_sparse_blob_roundtrips_exactly() -> None:
    vector = _sparse_vector(65536, 0.087)  # (w,)

    recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(vector, sparse=True))  # (w,)

    assert recovered.dtype == vector.dtype
    assert recovered.shape == vector.shape
    assert torch.equal(recovered, vector)


def test_sparse_blob_shrinks_wide_codebooks() -> None:
    vector = _sparse_vector(131072, 0.036)  # (w,)

    blob = tensor_to_embedding_blob(vector, sparse=True)

    dense_bytes = vector.numel() * vector.element_size()
    assert len(blob) < 0.15 * dense_bytes


def test_dense_format_is_never_larger_than_the_payload() -> None:
    vector = _sparse_vector(8192, 0.5)  # (w,)

    blob = tensor_to_embedding_blob(vector)

    assert blob[0] == _COMPACT_VERSION
    assert len(blob) <= vector.numel() * vector.element_size() + 16


def test_bfloat16_sparse_blob_roundtrips() -> None:
    vector = _sparse_vector(65536, 0.05, dtype=torch.bfloat16)  # (w,)

    recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(vector, sparse=True))  # (w,)

    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, vector)


def test_batch_applies_one_format_to_every_row() -> None:
    width = 131072  # w
    batch = torch.zeros(3, width, dtype=torch.float16)  # (b, w)
    for row, count in enumerate((100, 5_000, 120_000)):
        batch[row, torch.randperm(width)[:count]] = 1.5

    blobs = batch_tensor_to_blobs(batch, sparse=True)

    assert [blob[0] for blob in blobs] == [_SPARSE_VERSION] * 3
    for row, blob in enumerate(blobs):
        assert torch.equal(embedding_blob_to_tensor(blob), batch[row])


def test_batch_of_dense_embeddings_matches_single_serialization() -> None:
    batch = torch.randn(4, 320, dtype=torch.float32)  # (b, d)

    blobs = batch_tensor_to_blobs(batch)

    assert blobs == [tensor_to_embedding_blob(batch[row]) for row in range(batch.shape[0])]


def test_sparse_matrix_embedding_roundtrips() -> None:
    torch.manual_seed(0)
    matrix = torch.zeros(7, 65536, dtype=torch.float16)  # (l, w)
    matrix[torch.randint(0, 7, (400,)), torch.randint(0, 65536, (400,))] = 2.0

    recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(matrix, sparse=True))  # (l, w)

    assert recovered.shape == matrix.shape
    assert torch.equal(recovered, matrix)


def test_bfloat16_keeps_values_below_the_float16_normal_range() -> None:
    # bfloat16 has a wider exponent than float16, so storing it as float16 bytes lost
    # small magnitudes. These are the values that used to come back changed.
    vector = torch.tensor([1.0e-6, 3.0e-8, 1.0e-30, 1.0, -2.5], dtype=torch.bfloat16)  # (d,)

    recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(vector))  # (d,)

    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, vector)


def test_sparse_bfloat16_keeps_small_magnitudes() -> None:
    vector = torch.zeros(8192, dtype=torch.bfloat16)  # (w,)
    vector[5] = 1.0e-6
    vector[9] = 1.0e-30

    recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(vector, sparse=True))  # (w,)

    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, vector)


def test_legacy_bfloat16_blobs_still_read() -> None:
    # Caches written before the fix tagged float16 bytes with dtype code 1.
    values = torch.tensor([1.0, -2.5, 0.5], dtype=torch.bfloat16)  # (d,)
    legacy = struct.pack('<BBii', _COMPACT_VERSION, 1, 1, 3) + values.half().numpy().tobytes()

    recovered = embedding_blob_to_tensor(legacy)  # (d,)

    assert recovered.dtype == torch.bfloat16
    assert torch.equal(recovered, values)


def test_negative_zero_survives_the_coordinate_list_format_as_zero() -> None:
    # The sparse writer keeps only non-zero entries, and negative zero is not one, so it
    # comes back as positive zero. The two compare equal, which is all callers rely on.
    vector = torch.zeros(4096, dtype=torch.float16)  # (w,)
    vector[7] = -0.0
    vector[11] = 2.5

    recovered = embedding_blob_to_tensor(tensor_to_embedding_blob(vector, sparse=True))  # (w,)

    assert torch.equal(recovered, vector)


def test_batch_and_single_serialization_agree_in_both_formats() -> None:
    width = 8192  # w
    batch = torch.zeros(4, width, dtype=torch.float16)  # (b, w)
    for row, count in enumerate((10, 1_000, 3_000, 7_000)):
        batch[row, torch.randperm(width)[:count]] = 1.25

    for sparse in (False, True):
        blobs = batch_tensor_to_blobs(batch, sparse=sparse)
        assert blobs == [
            tensor_to_embedding_blob(batch[row], sparse=sparse) for row in range(batch.shape[0])
        ]


def test_all_zero_embedding_roundtrips() -> None:
    vector = torch.zeros(4096, dtype=torch.float16)  # (w,)

    blob = tensor_to_embedding_blob(vector, sparse=True)

    assert blob[0] == _SPARSE_VERSION
    assert torch.equal(embedding_blob_to_tensor(blob), vector)
