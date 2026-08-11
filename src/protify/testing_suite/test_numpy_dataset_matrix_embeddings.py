"""Matrix embeddings feeding the numpy dataset the estimator and scikit probes consume.

These learners take one vector per sequence, so a per-residue cache has to be reduced on
the way in. The stored matrix keeps the leading and trailing special tokens, so it has
len(seq)+2 rows and cannot be reshaped against the sequence length.
"""

import types

import numpy as np
import pytest
import torch

try:
    from src.protify.data.data_mixin import DataMixin
except ImportError:
    try:
        from protify.data.data_mixin import DataMixin
    except ImportError:
        from ..data.data_mixin import DataMixin


POOLING_TYPES = ['max']
HIDDEN_STATE_INDEX = -1
HIDDEN_SIZE = 3  # d

SEQ_A = 'AAAA'
SEQ_B = 'CCCCCCC'


def _residue_matrix(seq: str, offset: float) -> torch.Tensor:
    """One cached matrix embedding, special tokens included."""
    rows = len(seq) + 2  # l
    values = torch.arange(rows * HIDDEN_SIZE, dtype=torch.float32) + offset
    return values.reshape(rows, HIDDEN_SIZE)  # (l, d)


@pytest.fixture
def matrix_mixin(tmp_path):
    """A DataMixin whose only embedding source is a two-protein matrix .pth cache."""
    from protify.embedder import get_embedding_filename

    embeddings = {SEQ_A: _residue_matrix(SEQ_A, 0.0), SEQ_B: _residue_matrix(SEQ_B, 100.0)}
    filename = get_embedding_filename('test-model', True, POOLING_TYPES, 'pth', HIDDEN_STATE_INDEX)
    torch.save(embeddings, tmp_path / filename)

    mixin = DataMixin()
    mixin._sql = False
    mixin._full = True
    mixin.embedding_args = types.SimpleNamespace(
        embedding_save_dir=str(tmp_path),
        pooling_types=POOLING_TYPES,
        hidden_state_index=HIDDEN_STATE_INDEX,
    )
    return mixin


def test_matrix_embeddings_average_to_one_row_per_sequence(matrix_mixin) -> None:
    # Sequences of different lengths have to land in the same matrix, which rules out any
    # layout that keeps a per-residue axis.
    X_train, X_valid, X_test = matrix_mixin.build_vector_numpy_dataset_from_embeddings(
        'test-model', [SEQ_A, SEQ_B], [SEQ_A], [SEQ_B]
    )

    assert X_train.shape == (2, HIDDEN_SIZE)
    assert X_valid.shape == (1, HIDDEN_SIZE)
    assert X_test.shape == (1, HIDDEN_SIZE)

    expected_a = _residue_matrix(SEQ_A, 0.0).mean(dim=0).numpy()  # (d,)
    expected_b = _residue_matrix(SEQ_B, 100.0).mean(dim=0).numpy()  # (d,)
    assert np.allclose(X_train[0], expected_a)
    assert np.allclose(X_train[1], expected_b)
    assert np.allclose(X_valid[0], expected_a)
    assert np.allclose(X_test[0], expected_b)


def test_matrix_pair_embeddings_concatenate_two_averaged_vectors(matrix_mixin) -> None:
    X_train, _, _ = matrix_mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model',
        [SEQ_A], [SEQ_B],
        [SEQ_A], [SEQ_B],
        [SEQ_A], [SEQ_B],
    )

    assert X_train.shape == (1, 2 * HIDDEN_SIZE)

    expected = np.concatenate([
        _residue_matrix(SEQ_A, 0.0).mean(dim=0).numpy(),  # (d,)
        _residue_matrix(SEQ_B, 100.0).mean(dim=0).numpy(),  # (d,)
    ])  # (2 * d,)
    assert np.allclose(X_train[0], expected)
