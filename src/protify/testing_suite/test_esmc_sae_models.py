"""ESMC SAE model naming, registry entries, and estimator probe construction."""

import pytest

try:
    from src.protify.base_models.esmc_sae import (
        DEFAULT_SAE_CODEBOOK_DIM,
        DEFAULT_SAE_K,
        ESMC_SAE_COVERAGE,
        SPARSE_ACTIVE_FRACTION,
        is_sae_model_name,
        parse_sae_model_name,
        resolve_sae_model_name,
    )
    from src.protify.base_models.get_base_models import BaseModelArguments
    from src.protify.base_models.supported_models import (
        all_presets_with_paths,
        currently_supported_models,
    )
    from src.protify.probes.estimator_probe import (
        ESTIMATOR_DEFAULTS,
        ESTIMATOR_PROBE_TYPES,
        EstimatorProbeArguments,
        build_estimator,
        is_estimator_probe,
    )
    from src.protify.probes.get_probe import normalize_probe_type
    from src.protify.base_models.get_base_models import get_base_model_for_training
except ImportError:
    from protify.base_models.esmc_sae import (
        DEFAULT_SAE_CODEBOOK_DIM,
        DEFAULT_SAE_K,
        ESMC_SAE_COVERAGE,
        SPARSE_ACTIVE_FRACTION,
        is_sae_model_name,
        parse_sae_model_name,
        resolve_sae_model_name,
    )
    from protify.base_models.get_base_models import BaseModelArguments
    from protify.base_models.supported_models import (
        all_presets_with_paths,
        currently_supported_models,
    )
    from protify.probes.estimator_probe import (
        ESTIMATOR_DEFAULTS,
        ESTIMATOR_PROBE_TYPES,
        EstimatorProbeArguments,
        build_estimator,
        is_estimator_probe,
    )
    from protify.probes.get_probe import normalize_probe_type
    from protify.base_models.get_base_models import get_base_model_for_training


SAE_ALIASES = ("ESMC-300-SAE", "ESMC-600-SAE", "ESMC-6B-SAE")


def test_every_alias_is_registered() -> None:
    for alias in SAE_ALIASES:
        assert alias in all_presets_with_paths
        assert alias in currently_supported_models


def test_aliases_resolve_to_the_published_depth_layer() -> None:
    assert resolve_sae_model_name("ESMC-300-SAE") == "ESMC-300-SAE-l23-k64-c8192"
    assert resolve_sae_model_name("ESMC-600-SAE") == "ESMC-600-SAE-l27-k64-c8192"
    assert resolve_sae_model_name("ESMC-6B-SAE") == "ESMC-6B-SAE-l60-k64-c8192"


def test_default_sparsity_and_width_match_the_resolved_name() -> None:
    resolved = resolve_sae_model_name("ESMC-300-SAE")

    assert f"-k{DEFAULT_SAE_K}-" in resolved
    assert resolved.endswith(f"-c{DEFAULT_SAE_CODEBOOK_DIM}")


def test_overrides_select_a_different_checkpoint() -> None:
    resolved = resolve_sae_model_name("ESMC-300-SAE", sae_layer=12, sae_codebook_dim=16384)

    assert resolved == "ESMC-300-SAE-l12-k64-c16384"


def test_unpublished_combination_names_the_valid_options() -> None:
    with pytest.raises(ValueError, match="only publishes k=64 at codebook widths"):
        resolve_sae_model_name("ESMC-300-SAE", sae_layer=12, sae_codebook_dim=8192)


def test_override_contradicting_a_qualified_name_raises() -> None:
    with pytest.raises(ValueError, match="already fixes its SAE checkpoint"):
        resolve_sae_model_name("ESMC-300-SAE-l23-k64-c8192", sae_layer=12)


def test_qualified_name_passes_through_unchanged() -> None:
    name = "ESMC-600-SAE-l27-k256-c65536"

    assert resolve_sae_model_name(name) == name


def test_parsing_a_qualified_name_yields_the_biohub_checkpoint() -> None:
    selection = parse_sae_model_name("ESMC-300-SAE-l23-k64-c8192")

    assert selection.repository == "biohub/ESMC-300M-sae-layer23-k64-codebook8192"
    assert ESMC_SAE_COVERAGE[selection.backbone].model_path == "Synthyra/ESMplusplus_small"
    assert selection.layer == 23
    assert selection.codebook_dim == 8192


def test_parsing_a_bare_alias_raises() -> None:
    with pytest.raises(ValueError, match="not a fully qualified"):
        parse_sae_model_name("ESMC-300-SAE")


def test_name_detection_ignores_ordinary_models() -> None:
    assert is_sae_model_name("ESMC-300-SAE")
    assert not is_sae_model_name("ESMC-300")
    assert not is_sae_model_name("ESM2-8")


def test_model_arguments_expand_aliases_and_leave_others_alone() -> None:
    args = BaseModelArguments(model_names=["ESMC-300-SAE", "ESM2-8"], sae_codebook_dim=16384)

    assert args.model_names == ["ESMC-300-SAE-l23-k64-c16384", "ESM2-8"]


def test_model_arguments_keep_the_resolved_name_in_entries() -> None:
    args = BaseModelArguments(model_names=["ESMC-600-SAE"])

    entries = list(args.model_entries())
    assert entries == [("ESMC-600-SAE-l27-k64-c8192", "ESMC-600-SAE-l27-k64-c8192", None)]


def test_sae_models_pool_inside_the_encoder() -> None:
    # Embedder reads `pools_internally` to skip building a Pooler over SAE outputs. That is
    # what lets 'sum' through: the generic Pooler owns a different name set and would reject
    # it before any embedding runs.
    try:
        from src.protify.base_models.esmc_sae import SAE_POOLING_TYPES, EsmcSaeForEmbedding
        from src.protify.pooler import Pooler
    except ImportError:
        from protify.base_models.esmc_sae import SAE_POOLING_TYPES, EsmcSaeForEmbedding
        from protify.pooler import Pooler

    assert EsmcSaeForEmbedding.pools_internally
    assert set(SAE_POOLING_TYPES) - set(Pooler(['mean']).pooling_options) == {'sum'}


def test_estimator_probe_types_are_recognized() -> None:
    for probe_type in ESTIMATOR_PROBE_TYPES:
        assert is_estimator_probe(probe_type)
    assert not is_estimator_probe("mlp")
    assert not is_estimator_probe("transformer")


def test_estimator_defaults_cover_every_estimator_probe_type() -> None:
    assert set(ESTIMATOR_DEFAULTS) == set(ESTIMATOR_PROBE_TYPES)


def test_probe_type_normalization_leaves_estimator_names_alone() -> None:
    for probe_type in ESTIMATOR_PROBE_TYPES:
        assert normalize_probe_type(probe_type) == probe_type


def test_xgboost_classifier_carries_the_measured_defaults() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(probe_type="xgboost", task_type="singlelabel", num_labels=2)
    )

    params = estimator.get_params()
    assert params["learning_rate"] == ESTIMATOR_DEFAULTS["xgboost"]["learning_rate"]
    assert params["max_depth"] == ESTIMATOR_DEFAULTS["xgboost"]["max_depth"]
    assert params["early_stopping_rounds"] == 50
    assert params["objective"] == "binary:logistic"


def test_multiclass_targets_switch_the_xgboost_objective() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(probe_type="xgboost", task_type="singlelabel", num_labels=7)
    )

    assert estimator.get_params()["objective"] == "multi:softprob"


def test_regression_builds_a_regressor() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(probe_type="xgboost", task_type="regression")
    )

    assert type(estimator).__name__ == "XGBRegressor"


def test_multilabel_wraps_the_classifier_and_drops_early_stopping() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(probe_type="xgboost", task_type="multilabel", num_labels=4)
    )

    assert type(estimator).__name__ == "MultiOutputClassifier"
    assert estimator.estimator.get_params()["early_stopping_rounds"] is None


def test_overrides_take_precedence_over_the_defaults() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(
            probe_type="xgboost", task_type="singlelabel", overrides={"max_depth": 3}
        )
    )

    assert estimator.get_params()["max_depth"] == 3


def test_seed_reaches_the_estimator() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(probe_type="lightgbm", task_type="singlelabel", seed=1234)
    )

    assert estimator.get_params()["random_state"] == 1234


def test_unknown_estimator_probe_type_raises() -> None:
    with pytest.raises(ValueError, match="is not an estimator probe"):
        build_estimator(EstimatorProbeArguments(probe_type="catboost", task_type="singlelabel"))


def test_sigmoid_regression_builds_a_regressor() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(probe_type="xgboost", task_type="sigmoid_regression")
    )

    assert type(estimator).__name__ == "XGBRegressor"


def test_tokenwise_tasks_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot handle task type"):
        build_estimator(EstimatorProbeArguments(probe_type="xgboost", task_type="tokenwise"))


def test_objective_override_survives() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(
            probe_type="xgboost",
            task_type="singlelabel",
            num_labels=2,
            overrides={"objective": "binary:logitraw"},
        )
    )

    assert estimator.get_params()["objective"] == "binary:logitraw"


def test_multilabel_overrides_reach_the_wrapped_estimator() -> None:
    estimator = build_estimator(
        EstimatorProbeArguments(
            probe_type="xgboost", task_type="multilabel", num_labels=4, overrides={"max_depth": 5}
        )
    )

    assert estimator.estimator.get_params()["max_depth"] == 5


def _sparse_storage_for(codebook_dim: int, k: int) -> bool:
    """The storage policy alone, without loading any checkpoint."""
    return k / codebook_dim <= SPARSE_ACTIVE_FRACTION


def test_storage_policy_matches_the_measured_densities() -> None:
    # Measured max-pooled density stayed under the break-even for these and exceeded it
    # for k=256 at codebook 16384, where it reached 56.5% on long proteins.
    assert _sparse_storage_for(16384, 16)
    assert _sparse_storage_for(16384, 64)
    assert not _sparse_storage_for(16384, 256)


def test_storage_policy_keeps_the_narrowest_codebook_dense_at_the_default_sparsity() -> None:
    # Codebook 8192 at k=64 split almost evenly between the formats on real proteins,
    # and dense was the better single choice.
    assert not _sparse_storage_for(8192, 64)


def test_storage_policy_follows_the_active_fraction_not_the_width_alone() -> None:
    # Widening the codebook at fixed k lowers the active fraction and turns storage
    # sparse; raising k at fixed width does the reverse.
    assert _sparse_storage_for(131072, 64)
    assert _sparse_storage_for(131072, 512)
    assert not _sparse_storage_for(8192, 512)


def test_sae_models_have_no_trainable_path() -> None:
    with pytest.raises(ValueError, match="no trainable path"):
        get_base_model_for_training("ESMC-300-SAE-l23-k64-c8192", num_labels=2)
