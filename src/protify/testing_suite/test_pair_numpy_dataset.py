"""Pair ordering in the numpy dataset the estimator and scikit probes consume.

Interaction is symmetric, but concatenating A with B is not, so swapping the two proteins
is a training augmentation. Doing it on validation or test instead makes a reported score
depend on the random state, which these tests pin down.
"""

import random
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

SEQ_A = 'AAAA'
SEQ_B = 'CCCC'


@pytest.fixture
def mixin(tmp_path):
    """A DataMixin whose only embedding source is a two-protein .pth cache."""
    from protify.embedder import get_embedding_filename

    # One distinguishing value per protein, so a swap is visible in the concatenated row.
    embeddings = {
        SEQ_A: torch.tensor([1.0, 2.0]),  # (d,), d = 2
        SEQ_B: torch.tensor([30.0, 40.0]),  # (d,)
    }
    filename = get_embedding_filename('test-model', False, POOLING_TYPES, 'pth', HIDDEN_STATE_INDEX)
    torch.save(embeddings, tmp_path / filename)

    mixin = DataMixin()
    mixin._sql = False
    mixin._full = False
    mixin.embedding_args = types.SimpleNamespace(
        embedding_save_dir=str(tmp_path),
        pooling_types=POOLING_TYPES,
        hidden_state_index=HIDDEN_STATE_INDEX,
    )
    return mixin


def build(mixin, flip_train_pairs, pair_count=64):
    """Build train, validation, and test matrices from the same repeated pair."""
    seqs_a = [SEQ_A] * pair_count
    seqs_b = [SEQ_B] * pair_count
    return mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model',
        seqs_a, seqs_b,
        seqs_a, seqs_b,
        seqs_a, seqs_b,
        flip_train_pairs=flip_train_pairs,
    )  # three arrays, each (pair_count, 2 * d)


def test_evaluation_splits_keep_dataset_pair_order_when_flipping(mixin):
    random.seed(0)
    _, X_valid, X_test = build(mixin, flip_train_pairs=True)  # (n, 2 * d) each

    forward = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,)
    assert np.array_equal(X_valid, np.tile(forward, (X_valid.shape[0], 1)))
    assert np.array_equal(X_test, np.tile(forward, (X_test.shape[0], 1)))


def test_evaluation_splits_are_independent_of_the_random_state(mixin):
    random.seed(0)
    _, first_valid, first_test = build(mixin, flip_train_pairs=True)
    random.seed(12345)
    _, second_valid, second_test = build(mixin, flip_train_pairs=True)

    assert np.array_equal(first_valid, second_valid)
    assert np.array_equal(first_test, second_test)


def test_training_pairs_swap_only_when_flipping_is_requested(mixin):
    random.seed(0)
    flipped_train, _, _ = build(mixin, flip_train_pairs=True)  # (n, 2 * d)
    reversed_row = np.array([30.0, 40.0, 1.0, 2.0])  # (2 * d,)
    swapped = (flipped_train == reversed_row).all(axis=1)  # (n,)
    assert swapped.any(), 'flipping requested but no training pair was swapped'
    assert not swapped.all(), 'flipping should leave some training pairs in dataset order'

    unflipped_train, _, _ = build(mixin, flip_train_pairs=False)
    forward = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,)
    assert np.array_equal(unflipped_train, np.tile(forward, (unflipped_train.shape[0], 1)))


def test_flipping_defaults_to_off(mixin):
    random.seed(0)
    seqs_a, seqs_b = [SEQ_A] * 64, [SEQ_B] * 64
    X_train, _, _ = mixin.build_pair_vector_numpy_dataset_from_embeddings(
        'test-model', seqs_a, seqs_b, seqs_a, seqs_b, seqs_a, seqs_b,
    )  # (n, 2 * d)

    forward = np.array([1.0, 2.0, 30.0, 40.0])  # (2 * d,)
    assert np.array_equal(X_train, np.tile(forward, (X_train.shape[0], 1)))
