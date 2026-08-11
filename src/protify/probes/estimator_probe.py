"""Gradient-boosted and forest probes over pooled embeddings.

These probes sit beside the neural probes in `--probe_type` rather than behind the
separate `--use_scikit` branch, so they share the embedding cache, results table, plots,
and multi-run seed handling.

Axis-aligned learners suit sparse autoencoder features in particular. A max-pooled SAE
vector answers "did this protein ever express feature j, and how strongly", which is the
question a decision-tree split asks directly. Dense PLM embeddings work too, but the
defaults here are chosen for the wide, non-negative, mostly zero SAE case.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Any, Callable


# Metrics, transformers, and the estimator libraries load inside the functions that need
# them. `main.parse_arguments` reads ESTIMATOR_PROBE_TYPES before `entrypoint_setup` has
# configured the torch and TensorFlow environment, so importing them here would run those
# frameworks too early.


ESTIMATOR_PROBE_TYPES = ("xgboost", "lightgbm", "random_forest")

# Defaults for wide, sparse, non-negative features such as max-pooled SAE codebooks.
#
# The xgboost values were selected on validation MCC over seeds 42 to 44, fitting
# ESMC-300-SAE-l23-k64-c8192 max-pooled features on the solubility dataset (55536 train
# rows, 8192 features). They reach 0.4864 validation MCC against 0.4458 for XGBoost's own
# library defaults, a gap around ten times the 0.0044 seed standard deviation. That
# selection predates the move to the FastPLMs SAE runtime, whose residue standardization
# raised the same features from 23.7% to 36.6% non-zero; the defaults were not reselected
# afterwards. Most of the gap comes from many shallow-learning-rate trees under early
# stopping rather than from any single setting: sweeping `colsample_bytree` from 0.1 to
# 1.0 moved validation MCC by less than 0.01 and barely moved fit time. It stays low
# because the cost of scanning every column grows with codebook width, not because it
# was measured to help accuracy.
#
# The lightgbm and random_forest values mirror the same shape and are not separately
# measured. Override any of them with --scikit_model_args.
ESTIMATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "max_depth": 8,
        "min_child_weight": 16,
        "subsample": 1.0,
        "colsample_bytree": 0.3,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    },
    "lightgbm": {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "min_child_samples": 16,
        "colsample_bytree": 0.3,
        "reg_lambda": 1.0,
        "verbosity": -1,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
}


@dataclass
class EstimatorProbeArguments:
    """Configuration for one estimator probe fit."""

    probe_type: str
    task_type: str
    num_labels: int = 2
    seed: int = 42
    n_jobs: int = -1
    device: str = "cpu"
    overrides: dict[str, Any] = field(default_factory=dict)


# Task types an estimator probe can fit. Token-level tasks are excluded because these
# learners consume one pooled vector per sequence, not a per-residue sequence.
REGRESSION_TASK_TYPES = ("regression", "sigmoid_regression")
SUPPORTED_TASK_TYPES = REGRESSION_TASK_TYPES + ("singlelabel", "multilabel")


def is_estimator_probe(probe_type: str) -> bool:
    return probe_type in ESTIMATOR_PROBE_TYPES


def check_task_type(probe_type: str, task_type: str) -> None:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"--probe_type {probe_type} fits one pooled vector per sequence and cannot "
            f"handle task type {task_type!r}. Supported types are {SUPPORTED_TASK_TYPES}."
        )


def _load_estimator_classes(probe_type: str) -> tuple[type, type]:
    """Return the (classifier, regressor) pair for a probe type."""
    if probe_type == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier, XGBRegressor
    if probe_type == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        return LGBMClassifier, LGBMRegressor
    if probe_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        return RandomForestClassifier, RandomForestRegressor

    raise ValueError(f"{probe_type!r} is not an estimator probe; expected {ESTIMATOR_PROBE_TYPES}.")


def build_estimator(args: EstimatorProbeArguments) -> Any:
    """Construct an unfitted estimator with the tuned defaults and any overrides."""
    check_task_type(args.probe_type, args.task_type)
    classifier_cls, regressor_cls = _load_estimator_classes(args.probe_type)
    params = dict(ESTIMATOR_DEFAULTS[args.probe_type])
    params["random_state"] = args.seed

    if args.probe_type in ("lightgbm", "random_forest"):
        params["n_jobs"] = args.n_jobs
    if args.probe_type == "xgboost":
        params["n_jobs"] = args.n_jobs
        params["device"] = args.device
        if args.task_type == "singlelabel":
            params["objective"] = (
                "binary:logistic" if args.num_labels <= 2 else "multi:softprob"
            )

    if args.task_type == "multilabel" and args.probe_type != "random_forest":
        # XGBoost and LightGBM need an explicit wrapper for multi-output targets, and
        # early stopping needs a per-output eval set the wrapper cannot supply.
        from sklearn.multioutput import MultiOutputClassifier

        params.pop("early_stopping_rounds", None)
        params.update(args.overrides)
        return MultiOutputClassifier(classifier_cls(**params), n_jobs=1)

    # Overrides land last so every default, including the objective, stays adjustable.
    params.update(args.overrides)

    if args.task_type in REGRESSION_TASK_TYPES:
        return regressor_cls(**params)

    return classifier_cls(**params)


def _fit(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Any:
    """Fit an estimator, passing a validation set when it supports early stopping."""
    # X_train: (n_train, d); y_train: (n_train,) or (n_train, c)
    uses_early_stopping = getattr(estimator, "early_stopping_rounds", None) is not None
    if uses_early_stopping:
        estimator.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        estimator.fit(X_train, y_train)

    return estimator


def _predictions(estimator: Any, X: np.ndarray, task_type: str) -> np.ndarray:
    """Scores in the layout the shared metric functions expect."""
    # X: (n, d)
    if task_type in REGRESSION_TASK_TYPES:
        return estimator.predict(X)  # (n,)

    if task_type == "multilabel":
        # MultiOutputClassifier yields one (n, 2) array per label.
        per_label = estimator.predict_proba(X)
        return np.stack([column[:, 1] for column in per_label], axis=1)  # (n, c)

    probabilities = estimator.predict_proba(X)  # (n, c)
    if probabilities.shape[1] == 1:
        probabilities = np.concatenate([1.0 - probabilities, probabilities], axis=1)  # (n, 2)

    return probabilities  # (n, c)


def _metric_function(task_type: str) -> Callable[[Any], dict[str, float]]:
    try:
        from metrics import (
            compute_multi_label_classification_metrics,
            compute_regression_metrics,
            compute_single_label_classification_metrics,
        )
    except ImportError:
        from ..metrics import (
            compute_multi_label_classification_metrics,
            compute_regression_metrics,
            compute_single_label_classification_metrics,
        )

    if task_type in REGRESSION_TASK_TYPES:
        return compute_regression_metrics
    if task_type == "multilabel":
        return compute_multi_label_classification_metrics

    return compute_single_label_classification_metrics


def train_estimator_probe(
    args: EstimatorProbeArguments,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[Any, dict[str, float], dict[str, float]]:
    """Fit one estimator probe and score it on the validation and test splits."""
    # X_*: (n_split, d); y_*: (n_split,) or (n_split, c)
    from transformers import EvalPrediction

    try:
        from utils import print_message
    except ImportError:
        from ..utils import print_message

    estimator = build_estimator(args)
    print_message(
        f"Fitting {args.probe_type} probe on {X_train.shape[0]} samples "
        f"with {X_train.shape[1]} features "
        f"({100 * float(np.count_nonzero(X_train)) / X_train.size:.1f}% non-zero)"
    )
    estimator = _fit(estimator, X_train, y_train, X_valid, y_valid)

    compute_metrics = _metric_function(args.task_type)
    valid_metrics = compute_metrics(
        EvalPrediction(predictions=_predictions(estimator, X_valid, args.task_type), label_ids=y_valid)
    )
    test_metrics = compute_metrics(
        EvalPrediction(predictions=_predictions(estimator, X_test, args.task_type), label_ids=y_test)
    )

    return estimator, valid_metrics, test_metrics
