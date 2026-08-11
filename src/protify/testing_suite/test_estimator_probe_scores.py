"""Score layout the estimator probes hand to the shared metric functions.

Those functions were written for neural probes, so they read their input as raw logits: the
single-label one softmaxes before computing AUC, and the multi-label one sigmoids before
thresholding at 0.5. Estimators produce calibrated probabilities instead, so these tests pin
the inverse transforms, and the multi-label case where a label never varies in training.
"""

import numpy as np
import pytest

try:
    from src.protify.probes.estimator_probe import (
        EstimatorProbeArguments,
        _predictions,
        build_estimator,
        train_estimator_probe,
    )
except ImportError:
    from protify.probes.estimator_probe import (
        EstimatorProbeArguments,
        _predictions,
        build_estimator,
        train_estimator_probe,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # x: (n, c)
    return 1.0 / (1.0 + np.exp(-x))  # (n, c)


def _softmax(x: np.ndarray) -> np.ndarray:
    # x: (n, c)
    shifted = np.exp(x - x.max(axis=1, keepdims=True))  # (n, c)
    return shifted / shifted.sum(axis=1, keepdims=True)  # (n, c)


@pytest.fixture
def separable_multilabel():
    """Two labels a shallow forest separates exactly, from 3 informative features."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))  # (n, d)
    y = np.stack([(X[:, 0] > 0).astype(int), (X[:, 1] > 0).astype(int)], axis=1)  # (n, c)
    return X, y


def test_multilabel_scores_invert_the_metric_sigmoid(separable_multilabel) -> None:
    X, y = separable_multilabel  # (n, d); (n, c)
    args = EstimatorProbeArguments(probe_type="random_forest", task_type="multilabel", num_labels=2)
    estimator = build_estimator(args).fit(X, y)

    scores = _predictions(estimator, X, "multilabel")  # (n, c)
    positives = np.stack([block[:, 1] for block in estimator.predict_proba(X)], axis=1)  # (n, c)

    assert scores.shape == y.shape
    assert np.allclose(_sigmoid(scores), positives, atol=1e-6)


def test_multilabel_metrics_do_not_collapse_to_all_positive(separable_multilabel) -> None:
    # Sigmoiding a probability puts every non-zero score above 0.5, which would make the
    # threshold metrics report an all-ones prediction on data the forest separates exactly.
    X, y = separable_multilabel  # (n, d); (n, c)
    args = EstimatorProbeArguments(probe_type="random_forest", task_type="multilabel", num_labels=2)
    _, valid_metrics, test_metrics = train_estimator_probe(args, X, y, X, y, X, y)

    assert test_metrics["mcc"] > 0.9
    assert test_metrics["accuracy"] > 0.9
    assert test_metrics["hamming_loss"] < 0.1
    assert valid_metrics["mcc"] == test_metrics["mcc"]


def test_a_label_constant_in_training_scores_as_negative() -> None:
    # num_labels counts labels across all three splits, so a label the training split never
    # sets still gets a column. Its estimator saw one class and cannot predict the other.
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(80, 3))  # (n_train, d)
    y_train = np.stack(
        [(X_train[:, 0] > 0).astype(int), np.zeros(80, dtype=int)], axis=1
    )  # (n_train, c)
    X_test = rng.normal(size=(20, 3))  # (n_test, d)
    y_test = np.stack([(X_test[:, 0] > 0).astype(int), np.ones(20, dtype=int)], axis=1)  # (n_test, c)

    args = EstimatorProbeArguments(probe_type="random_forest", task_type="multilabel", num_labels=2)
    estimator = build_estimator(args).fit(X_train, y_train)

    scores = _predictions(estimator, X_test, "multilabel")  # (n_test, c)

    assert scores.shape == y_test.shape
    assert np.allclose(_sigmoid(scores)[:, 1], 0.0, atol=1e-6)


def test_single_label_scores_invert_the_metric_softmax() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 3))  # (n, d)
    y = np.digitize(X[:, 0], [-0.5, 0.5])  # (n,); three ordered classes

    args = EstimatorProbeArguments(probe_type="random_forest", task_type="singlelabel", num_labels=3)
    estimator = build_estimator(args).fit(X, y)

    scores = _predictions(estimator, X, "singlelabel")  # (n, c)

    # Log probabilities sum to one after softmax, so the metric recovers them exactly rather
    # than distorting the one-vs-rest ranking the multiclass AUC reads.
    assert np.allclose(_softmax(scores), estimator.predict_proba(X), atol=1e-6)
