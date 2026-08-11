from dataclasses import replace

import pytest

try:
    from src.protify import probes as probes_package
    from src.protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
    from src.protify.probes.parallel_probe_plan import (
        ParallelProbeExecutionPlan,
        ParallelProbeGroup,
        ParallelProbeGroupEstimate,
        ParallelProbePlanEstimate,
        ParallelProbeRunSpec,
        ParallelProbeExecutionWave,
        ParallelProbeWaveSchedule,
        build_seed_run_specs,
        estimate_parallel_probe_group,
        estimate_parallel_probe_plan,
        group_parallel_probe_runs,
        linear_probe_batch_activation_count,
        linear_probe_forward_flop_count,
        linear_probe_parameter_count,
        linear_probe_parameter_count_for_spec,
        max_linear_probe_runs_for_estimated_peak_budget,
        max_linear_probe_runs_for_training_state_budget,
        plan_parallel_probe_runs,
        schedule_parallel_probe_execution_waves,
    )
except ImportError:
    try:
        from protify import probes as probes_package
        from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
        from protify.probes.parallel_probe_plan import (
            ParallelProbeExecutionPlan,
            ParallelProbeGroup,
            ParallelProbeGroupEstimate,
            ParallelProbePlanEstimate,
            ParallelProbeRunSpec,
            ParallelProbeExecutionWave,
            ParallelProbeWaveSchedule,
            build_seed_run_specs,
            estimate_parallel_probe_group,
            estimate_parallel_probe_plan,
            group_parallel_probe_runs,
            linear_probe_batch_activation_count,
            linear_probe_forward_flop_count,
            linear_probe_parameter_count,
            linear_probe_parameter_count_for_spec,
            max_linear_probe_runs_for_estimated_peak_budget,
            max_linear_probe_runs_for_training_state_budget,
            plan_parallel_probe_runs,
            schedule_parallel_probe_execution_waves,
        )
    except ImportError:
        from .. import probes as probes_package
        from ..probes.mlp_probe import MLPProbe, MLPProbeConfig
        from ..probes.parallel_probe_plan import (
            ParallelProbeExecutionPlan,
            ParallelProbeGroup,
            ParallelProbeGroupEstimate,
            ParallelProbePlanEstimate,
            ParallelProbeRunSpec,
            ParallelProbeExecutionWave,
            ParallelProbeWaveSchedule,
            build_seed_run_specs,
            estimate_parallel_probe_group,
            estimate_parallel_probe_plan,
            group_parallel_probe_runs,
            linear_probe_batch_activation_count,
            linear_probe_forward_flop_count,
            linear_probe_parameter_count,
            linear_probe_parameter_count_for_spec,
            max_linear_probe_runs_for_estimated_peak_budget,
            max_linear_probe_runs_for_training_state_budget,
            plan_parallel_probe_runs,
            schedule_parallel_probe_execution_waves,
        )


def _run_spec(run_id: str = 'run-0', seed: int = 7) -> ParallelProbeRunSpec:
    return ParallelProbeRunSpec(
        run_id=run_id,
        seed=seed,
        model_name='ESM2-35',
        data_name='EC',
        embedding_key='ESM2-35/EC/mean/v1',
        dataset_key='EC/split/v1',
        trainer_key='epochs=1|batch=8|lr=1e-4',
        probe_type='mlp',
        input_size=8,
        hidden_size=4,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
    )


def test_parallel_probe_plan_package_exports() -> None:
    assert "ParallelProbeExecutionPlan" in probes_package.__all__
    assert "ParallelProbeRunSpec" in probes_package.__all__
    assert "ParallelProbeGroup" in probes_package.__all__
    assert "ParallelProbeGroupEstimate" in probes_package.__all__
    assert "ParallelProbePlanEstimate" in probes_package.__all__
    assert "ParallelProbeExecutionWave" in probes_package.__all__
    assert "ParallelProbeWaveSchedule" in probes_package.__all__
    assert "build_seed_run_specs" in probes_package.__all__
    assert "estimate_parallel_probe_group" in probes_package.__all__
    assert "estimate_parallel_probe_plan" in probes_package.__all__
    assert "group_parallel_probe_runs" in probes_package.__all__
    assert "linear_probe_batch_activation_count" in probes_package.__all__
    assert "linear_probe_forward_flop_count" in probes_package.__all__
    assert "linear_probe_parameter_count" in probes_package.__all__
    assert "linear_probe_parameter_count_for_spec" in probes_package.__all__
    assert "max_linear_probe_runs_for_estimated_peak_budget" in probes_package.__all__
    assert "max_linear_probe_runs_for_training_state_budget" in probes_package.__all__
    assert "plan_parallel_probe_runs" in probes_package.__all__
    assert "schedule_parallel_probe_execution_waves" in probes_package.__all__
    assert probes_package.ParallelProbeExecutionPlan is ParallelProbeExecutionPlan
    assert probes_package.ParallelProbeRunSpec is ParallelProbeRunSpec
    assert probes_package.ParallelProbeGroup is ParallelProbeGroup
    assert probes_package.ParallelProbeGroupEstimate is ParallelProbeGroupEstimate
    assert probes_package.ParallelProbePlanEstimate is ParallelProbePlanEstimate
    assert probes_package.ParallelProbeExecutionWave is ParallelProbeExecutionWave
    assert probes_package.ParallelProbeWaveSchedule is ParallelProbeWaveSchedule
    assert probes_package.build_seed_run_specs is build_seed_run_specs
    assert probes_package.estimate_parallel_probe_group is estimate_parallel_probe_group
    assert probes_package.estimate_parallel_probe_plan is estimate_parallel_probe_plan
    assert probes_package.group_parallel_probe_runs is group_parallel_probe_runs
    assert probes_package.linear_probe_batch_activation_count is linear_probe_batch_activation_count
    assert probes_package.linear_probe_forward_flop_count is linear_probe_forward_flop_count
    assert probes_package.linear_probe_parameter_count is linear_probe_parameter_count
    assert probes_package.linear_probe_parameter_count_for_spec is linear_probe_parameter_count_for_spec
    assert probes_package.max_linear_probe_runs_for_estimated_peak_budget is (
        max_linear_probe_runs_for_estimated_peak_budget
    )
    assert probes_package.max_linear_probe_runs_for_training_state_budget is (
        max_linear_probe_runs_for_training_state_budget
    )
    assert probes_package.plan_parallel_probe_runs is plan_parallel_probe_runs
    assert probes_package.schedule_parallel_probe_execution_waves is (
        schedule_parallel_probe_execution_waves
    )


def test_build_seed_run_specs_groups_into_one_vectorized_group() -> None:
    specs = build_seed_run_specs(
        run_id_prefix='EC/ESM2-35',
        base_seed=42,
        num_runs=3,
        model_name='ESM2-35',
        data_name='EC',
        embedding_key='ESM2-35/EC/mean/v1',
        dataset_key='EC/split/v1',
        trainer_key='epochs=1|batch=8|lr=1e-4',
        probe_type='mlp',
        input_size=8,
        hidden_size=4,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
    )

    groups = group_parallel_probe_runs(specs)

    assert len(groups) == 1
    assert groups[0].eligible
    assert groups[0].can_vectorize
    assert groups[0].num_runs == 3
    assert groups[0].run_seeds == (42, 43, 44)
    assert groups[0].run_ids == ('EC/ESM2-35/seed-42', 'EC/ESM2-35/seed-43', 'EC/ESM2-35/seed-44')


def test_parallel_probe_plan_splits_incompatible_universes() -> None:
    first = _run_spec(run_id='first', seed=1)
    second = replace(first, run_id='second', seed=2)
    other_dataset = replace(
        first,
        run_id='other-dataset',
        seed=3,
        data_name='DeepLoc-2',
        dataset_key='DeepLoc-2/split/v1',
    )
    other_shape = replace(
        first,
        run_id='other-shape',
        seed=4,
        embedding_key='ESM2-35/EC/mean/v2',
        input_size=16,
    )
    other_batch_mode = replace(
        first,
        run_id='other-batch-mode',
        seed=5,
        batch_mode='run_specific',
    )
    other_index_strategy = replace(
        first,
        run_id='other-index-strategy',
        seed=6,
        batch_mode='run_specific',
        index_strategy='affine',
    )

    groups = group_parallel_probe_runs(
        (first, second, other_dataset, other_shape, other_batch_mode, other_index_strategy)
    )

    assert len(groups) == 5
    assert groups[0].can_vectorize
    assert groups[0].run_ids == ('first', 'second')
    assert groups[1].eligible
    assert not groups[1].can_vectorize
    assert groups[1].run_ids == ('other-dataset',)
    assert groups[2].eligible
    assert not groups[2].can_vectorize
    assert groups[2].run_ids == ('other-shape',)
    assert groups[3].eligible
    assert not groups[3].can_vectorize
    assert groups[3].run_ids == ('other-batch-mode',)
    assert groups[4].eligible
    assert not groups[4].can_vectorize
    assert groups[4].run_ids == ('other-index-strategy',)


def test_parallel_probe_plan_keeps_ineligible_runs_sequential() -> None:
    base = _run_spec()
    transformer = replace(base, run_id='transformer', seed=11, probe_type='transformer')
    tokenwise = replace(base, run_id='tokenwise', seed=12, tokenwise=True)
    matrix = replace(base, run_id='matrix', seed=13, matrix_embed=True)

    groups = group_parallel_probe_runs((transformer, tokenwise, matrix))

    assert len(groups) == 3
    for group in groups:
        assert not group.eligible
        assert not group.can_vectorize
        assert group.num_runs == 1
    assert transformer.ineligibility_reasons() == ('probe_type',)
    assert tokenwise.ineligibility_reasons() == ('tokenwise',)
    assert matrix.ineligibility_reasons() == ('matrix_embed',)


def test_parallel_probe_plan_treats_save_model_as_compatible_group_dimension() -> None:
    base = _run_spec(run_id='base', seed=1)
    save_a = replace(base, run_id='save-a', seed=2, save_model=True)
    save_b = replace(save_a, run_id='save-b', seed=3)

    groups = group_parallel_probe_runs((base, save_a, save_b))

    assert len(groups) == 2
    assert groups[0].eligible
    assert not groups[0].can_vectorize
    assert groups[0].run_ids == ('base',)
    assert groups[1].eligible
    assert groups[1].can_vectorize
    assert groups[1].run_ids == ('save-a', 'save-b')


def test_parallel_probe_plan_chunks_compatible_runs_by_max_group_size() -> None:
    specs = tuple(replace(_run_spec(run_id=f'run-{idx}', seed=idx), seed=100 + idx) for idx in range(5))

    groups = group_parallel_probe_runs(specs, max_parallel_group_size=2)
    plan = plan_parallel_probe_runs(specs, max_parallel_group_size=2)

    assert tuple(group.num_runs for group in groups) == (2, 2, 1)
    assert groups[0].run_ids == ('run-0', 'run-1')
    assert groups[1].run_ids == ('run-2', 'run-3')
    assert groups[2].run_ids == ('run-4',)
    assert plan.total_runs == 5
    assert plan.trainer_invocations == 3
    assert plan.invocation_reduction == 2
    assert plan.compression_ratio == pytest.approx(5.0 / 3.0)
    assert plan.vectorized_runs == 4
    assert plan.sequential_runs == 1


def test_parallel_probe_plan_supports_per_key_group_size_caps() -> None:
    first_specs = tuple(replace(_run_spec(run_id=f'first-{idx}', seed=idx), seed=100 + idx) for idx in range(5))
    second_specs = tuple(
        replace(
            _run_spec(run_id=f'second-{idx}', seed=idx),
            seed=200 + idx,
            model_name='ESM2-8',
            embedding_key='ESM2-8/EC/mean/v1',
        )
        for idx in range(3)
    )
    max_group_size_by_key = {
        first_specs[0].compatibility_key(): 2,
        second_specs[0].compatibility_key(): 1,
    }

    plan = plan_parallel_probe_runs(
        first_specs + second_specs,
        max_parallel_group_size=4,
        max_parallel_group_size_by_key=max_group_size_by_key,
    )

    assert tuple(group.num_runs for group in plan.groups) == (2, 2, 1, 1, 1, 1)
    assert plan.trainer_invocations == 6
    assert plan.vectorized_runs == 4
    assert plan.sequential_runs == 4


def test_parallel_probe_plan_rejects_invalid_max_group_size() -> None:
    specs = (_run_spec(run_id='run-0', seed=1),)

    with pytest.raises(AssertionError, match="max_parallel_group_size"):
        group_parallel_probe_runs(specs, max_parallel_group_size=0)

    with pytest.raises(AssertionError, match="max_parallel_group_size"):
        plan_parallel_probe_runs(specs, max_parallel_group_size=0)

    with pytest.raises(AssertionError, match="max_parallel_group_size_by_key"):
        group_parallel_probe_runs(
            specs,
            max_parallel_group_size_by_key={specs[0].compatibility_key(): 0},
        )


def test_linear_probe_parameter_count_matches_actual_model_parameters() -> None:
    config = MLPProbeConfig(
        input_size=8,
        hidden_size=4,
        dropout=0.0,
        num_labels=3,
        n_layers=2,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
    )
    model = MLPProbe(config)
    expected = sum(parameter.numel() for parameter in model.parameters())

    observed = linear_probe_parameter_count(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_labels=config.num_labels,
        n_layers=config.n_layers,
        pre_ln=config.pre_ln,
        use_bias=config.use_bias,
    )

    assert observed == expected


def test_parallel_probe_group_estimate_scales_parameter_bank_by_run_count() -> None:
    specs = tuple(replace(_run_spec(run_id=f'run-{idx}', seed=idx), seed=200 + idx) for idx in range(3))
    group = group_parallel_probe_runs(specs)[0]
    single_count = linear_probe_parameter_count_for_spec(specs[0])

    estimate = estimate_parallel_probe_group(group, dtype_bytes=4, optimizer_state_multiplier=2)

    assert estimate.parameter_count_known
    assert estimate.single_probe_parameter_count == single_count
    assert estimate.group_parameter_count == single_count * 3
    assert estimate.parameter_bytes == single_count * 3 * 4
    assert estimate.gradient_bytes == estimate.parameter_bytes
    assert estimate.optimizer_state_bytes == estimate.parameter_bytes * 2
    assert estimate.training_state_bytes == estimate.parameter_bytes * 4
    assert estimate.batch_activation_bytes == 0
    assert estimate.logit_bytes == 0
    assert estimate.run_specific_index_bytes == 0
    assert estimate.estimated_peak_bytes == estimate.training_state_bytes


def test_linear_probe_batch_activation_count_includes_projection_width() -> None:
    activation_count = linear_probe_batch_activation_count(
        input_size=8,
        hidden_size=4,
        num_labels=2,
        n_layers=1,
    )

    assert activation_count == 8 + (4 * 3) + 256 + 2


def test_linear_probe_forward_flop_count_includes_linear_matmuls() -> None:
    flop_count = linear_probe_forward_flop_count(
        input_size=8,
        hidden_size=4,
        num_labels=2,
        n_layers=1,
    )

    expected_macs = (8 * 4) + (4 * 4) + (4 * 256) + (256 * 2)
    assert flop_count == expected_macs * 2


def test_parallel_probe_group_estimate_includes_batch_activations_and_indices() -> None:
    specs = tuple(
        replace(
            _run_spec(run_id=f'run-{idx}', seed=idx),
            seed=200 + idx,
            batch_mode='run_specific',
        )
        for idx in range(3)
    )
    group = group_parallel_probe_runs(specs)[0]
    activation_count = linear_probe_batch_activation_count(
        input_size=specs[0].input_size,
        hidden_size=specs[0].hidden_size,
        num_labels=specs[0].num_labels,
        n_layers=specs[0].n_layers,
    )
    forward_flops = linear_probe_forward_flop_count(
        input_size=specs[0].input_size,
        hidden_size=specs[0].hidden_size,
        num_labels=specs[0].num_labels,
        n_layers=specs[0].n_layers,
    )

    estimate = estimate_parallel_probe_group(
        group,
        dtype_bytes=2,
        optimizer_state_multiplier=2,
        batch_size=5,
        dataset_size=11,
        include_run_specific_index=True,
        index_dtype_bytes=4,
    )

    assert estimate.batch_size == 5
    assert estimate.dataset_size == 11
    assert estimate.batch_activation_bytes == 5 * 3 * activation_count * 2
    assert estimate.logit_bytes == 5 * 3 * specs[0].num_labels * 2
    assert estimate.run_specific_index_bytes == 11 * 3 * 4
    assert estimate.single_probe_forward_flops_per_sample == forward_flops
    assert estimate.group_forward_flops_per_batch == 5 * 3 * forward_flops
    assert estimate.group_training_flops_per_batch == 5 * 3 * forward_flops * 3
    assert estimate.estimated_peak_bytes == (
        estimate.training_state_bytes
        + estimate.batch_activation_bytes
        + estimate.run_specific_index_bytes
    )


def test_parallel_probe_plan_estimate_reports_peak_and_unknown_groups() -> None:
    linear_a = _run_spec(run_id='linear-a', seed=1)
    linear_b = replace(linear_a, run_id='linear-b', seed=2)
    transformer = replace(linear_a, run_id='transformer', seed=3, probe_type='transformer')
    plan = plan_parallel_probe_runs((linear_a, linear_b, transformer))

    estimate = estimate_parallel_probe_plan(plan)
    summary = estimate.summary_dict()

    assert len(estimate.group_estimates) == 2
    assert estimate.unknown_group_count == 1
    assert summary["unknown_group_count"] == 1
    assert summary["total_parameter_count"] == estimate.group_estimates[0].group_parameter_count
    assert summary["peak_group_training_state_bytes"] == estimate.group_estimates[0].training_state_bytes
    assert summary["total_batch_activation_bytes"] == 0
    assert summary["peak_group_batch_activation_bytes"] == 0
    assert summary["total_run_specific_index_bytes"] == 0
    assert summary["peak_group_estimated_peak_bytes"] == estimate.group_estimates[0].training_state_bytes
    assert summary["training_flop_multiplier"] == 3
    assert summary["total_forward_flops_per_batch"] == 0
    assert summary["peak_group_forward_flops_per_batch"] == 0
    assert summary["total_training_flops_per_batch"] == 0
    assert summary["peak_group_training_flops_per_batch"] == 0
    assert estimate.total_training_state_bytes == estimate.group_estimates[0].training_state_bytes
    assert estimate.group_estimates[1].parameter_count_known is False
    assert estimate.group_estimates[1].training_state_bytes == 0


def test_parallel_probe_plan_estimate_reports_peak_batch_and_index_memory() -> None:
    first = replace(_run_spec(run_id='run-0', seed=1), batch_mode='run_specific')
    second = replace(first, run_id='run-1', seed=2)
    plan = plan_parallel_probe_runs((first, second))

    estimate = estimate_parallel_probe_plan(
        plan,
        dtype_bytes=4,
        optimizer_state_multiplier=2,
        batch_size=7,
        dataset_size=13,
        include_run_specific_index=True,
        index_dtype_bytes=8,
    )
    summary = estimate.summary_dict()
    group_estimate = estimate.group_estimates[0]

    assert summary["batch_size"] == 7
    assert summary["dataset_size"] == 13
    assert summary["include_run_specific_index"] is True
    assert summary["index_dtype_bytes"] == 8
    assert summary["total_batch_activation_bytes"] == group_estimate.batch_activation_bytes
    assert summary["peak_group_batch_activation_bytes"] == group_estimate.batch_activation_bytes
    assert summary["total_run_specific_index_bytes"] == 13 * 2 * 8
    assert summary["peak_group_estimated_peak_bytes"] == group_estimate.estimated_peak_bytes
    assert summary["total_forward_flops_per_batch"] == group_estimate.group_forward_flops_per_batch
    assert summary["peak_group_forward_flops_per_batch"] == group_estimate.group_forward_flops_per_batch
    assert summary["total_training_flops_per_batch"] == group_estimate.group_training_flops_per_batch
    assert summary["peak_group_training_flops_per_batch"] == group_estimate.group_training_flops_per_batch
    assert group_estimate.estimated_peak_bytes > group_estimate.training_state_bytes


def test_parallel_probe_budget_helper_returns_seed_count_for_training_state_budget() -> None:
    spec = _run_spec()
    single_count = linear_probe_parameter_count_for_spec(spec)
    bytes_per_run = single_count * 4 * 4
    budget = (bytes_per_run * 7) + (bytes_per_run // 2)

    max_runs = max_linear_probe_runs_for_training_state_budget(
        spec,
        memory_budget_bytes=budget,
        dtype_bytes=4,
        optimizer_state_multiplier=2,
    )

    assert max_runs == 7


def test_parallel_probe_peak_budget_helper_accounts_for_activations_and_indices() -> None:
    spec = replace(_run_spec(), batch_mode='run_specific')
    parameter_count = linear_probe_parameter_count_for_spec(spec)
    activation_count = linear_probe_batch_activation_count(
        input_size=spec.input_size,
        hidden_size=spec.hidden_size,
        num_labels=spec.num_labels,
        n_layers=spec.n_layers,
    )
    bytes_per_run = (parameter_count * 4 * 4) + (5 * activation_count * 4) + (11 * 8)
    budget = (bytes_per_run * 3) + (bytes_per_run // 2)

    max_runs = max_linear_probe_runs_for_estimated_peak_budget(
        spec,
        memory_budget_bytes=budget,
        batch_size=5,
        dataset_size=11,
        include_run_specific_index=True,
        dtype_bytes=4,
        optimizer_state_multiplier=2,
        index_dtype_bytes=8,
    )

    assert max_runs == 3


def test_parallel_probe_estimate_rejects_invalid_sizing_inputs() -> None:
    spec = _run_spec()
    group = ParallelProbeGroup(runs=(spec,), eligible=True)

    with pytest.raises(AssertionError, match="dtype_bytes"):
        estimate_parallel_probe_group(group, dtype_bytes=0)

    with pytest.raises(AssertionError, match="optimizer_state_multiplier"):
        estimate_parallel_probe_group(group, optimizer_state_multiplier=-1)

    with pytest.raises(AssertionError, match="training_flop_multiplier"):
        estimate_parallel_probe_group(group, training_flop_multiplier=0)

    with pytest.raises(AssertionError, match="batch_size"):
        estimate_parallel_probe_group(group, batch_size=-1)

    with pytest.raises(AssertionError, match="dataset_size"):
        estimate_parallel_probe_group(group, dataset_size=-1)

    with pytest.raises(AssertionError, match="index_dtype_bytes"):
        estimate_parallel_probe_group(group, index_dtype_bytes=0)

    with pytest.raises(AssertionError, match="memory_budget_bytes"):
        max_linear_probe_runs_for_training_state_budget(spec, memory_budget_bytes=0)

    with pytest.raises(AssertionError, match="batch_size"):
        max_linear_probe_runs_for_estimated_peak_budget(
            spec,
            memory_budget_bytes=1,
            batch_size=-1,
        )


def test_parallel_probe_group_rejects_mixed_eligible_runs() -> None:
    first = _run_spec(run_id='first', seed=1)
    incompatible = replace(first, run_id='incompatible', seed=2, hidden_size=12)

    with pytest.raises(AssertionError, match="incompatible"):
        ParallelProbeGroup(runs=(first, incompatible), eligible=True)


def test_build_seed_run_specs_rejects_empty_run_sets() -> None:
    with pytest.raises(AssertionError, match="num_runs must be positive"):
        build_seed_run_specs(
            run_id_prefix='empty',
            base_seed=1,
            num_runs=0,
            model_name='ESM2-35',
            data_name='EC',
            embedding_key='ESM2-35/EC/mean/v1',
            dataset_key='EC/split/v1',
            trainer_key='epochs=1|batch=8|lr=1e-4',
            probe_type='mlp',
            input_size=8,
            hidden_size=4,
            dropout=0.0,
            num_labels=2,
            n_layers=0,
            task_type='singlelabel',
            pre_ln=True,
            use_bias=True,
        )


def test_execution_plan_summarizes_vectorized_and_sequential_work() -> None:
    first = _run_spec(run_id='first', seed=1)
    second = replace(first, run_id='second', seed=2)
    third = replace(first, run_id='third', seed=3)
    other_model_a = replace(
        first,
        run_id='other-model-a',
        seed=4,
        model_name='ESM2-8',
        embedding_key='ESM2-8/EC/mean/v1',
    )
    other_model_b = replace(other_model_a, run_id='other-model-b', seed=5)
    transformer = replace(first, run_id='transformer', seed=6, probe_type='transformer')

    plan = plan_parallel_probe_runs((first, second, third, other_model_a, other_model_b, transformer))
    rows = plan.summary_rows()

    assert plan.total_runs == 6
    assert plan.trainer_invocations == 3
    assert plan.invocation_reduction == 3
    assert plan.compression_ratio == pytest.approx(2.0)
    assert plan.vectorized_runs == 5
    assert plan.sequential_runs == 1
    assert len(plan.vectorized_groups) == 2
    assert len(plan.sequential_groups) == 1
    assert rows[0][0] == 'vectorized'
    assert rows[0][1] == 3
    assert rows[0][2] == ('first', 'second', 'third')
    assert rows[0][3] == ()
    assert rows[1][0] == 'vectorized'
    assert rows[1][2] == ('other-model-a', 'other-model-b')
    assert rows[2][0] == 'sequential_fallback'
    assert rows[2][3] == ('probe_type',)


def test_execution_plan_prefers_largest_parallel_groups_first() -> None:
    singleton = _run_spec(run_id='singleton', seed=1)
    pair_a = replace(singleton, run_id='pair-a', seed=2, model_name='ESM2-8', embedding_key='ESM2-8/EC/mean/v1')
    pair_b = replace(pair_a, run_id='pair-b', seed=3)
    triple_a = replace(singleton, run_id='triple-a', seed=4, data_name='DeepLoc-2', dataset_key='DeepLoc-2/split/v1')
    triple_b = replace(triple_a, run_id='triple-b', seed=5)
    triple_c = replace(triple_a, run_id='triple-c', seed=6)

    plan = plan_parallel_probe_runs((singleton, pair_a, pair_b, triple_a, triple_b, triple_c))
    ordered = plan.execution_groups(prefer_largest_parallel=True)
    original = plan.execution_groups(prefer_largest_parallel=False)

    assert ordered[0].run_ids == ('triple-a', 'triple-b', 'triple-c')
    assert ordered[1].run_ids == ('pair-a', 'pair-b')
    assert ordered[2].run_ids == ('singleton',)
    assert original[0].run_ids == ('singleton',)


def test_execution_plan_rejects_empty_universes() -> None:
    with pytest.raises(AssertionError, match="at least one run"):
        plan_parallel_probe_runs(())


def test_wave_schedule_defaults_to_one_group_per_wave() -> None:
    first_a = _run_spec(run_id='first-a', seed=1)
    first_b = replace(first_a, run_id='first-b', seed=2)
    second_a = replace(
        first_a,
        run_id='second-a',
        seed=3,
        model_name='ESM2-8',
        embedding_key='ESM2-8/EC/mean/v1',
    )
    second_b = replace(second_a, run_id='second-b', seed=4)
    plan = plan_parallel_probe_runs((first_a, first_b, second_a, second_b))
    estimate = estimate_parallel_probe_plan(plan, batch_size=8)

    schedule = schedule_parallel_probe_execution_waves(estimate)
    summary = schedule.summary_dict()

    assert schedule.total_waves == 2
    assert schedule.total_groups == 2
    assert schedule.total_runs == 4
    assert schedule.over_memory_budget_wave_count == 0
    assert schedule.target_satisfied_wave_count == 0
    assert schedule.target_underfilled_wave_count == 0
    assert schedule.waves[0].group_indices == (0,)
    assert schedule.waves[1].group_indices == (1,)
    assert summary["max_groups_per_wave"] == 1
    assert summary["waves"][0]["group_indices"] == [0]
    assert summary["waves"][0]["group_run_counts"] == [2]
    assert summary["waves"][0]["group_run_ids"] == [["first-a", "first-b"]]


def test_wave_schedule_packs_groups_within_memory_budget_and_reports_target() -> None:
    first_a = _run_spec(run_id='first-a', seed=1)
    first_b = replace(first_a, run_id='first-b', seed=2)
    second_a = replace(
        first_a,
        run_id='second-a',
        seed=3,
        model_name='ESM2-8',
        embedding_key='ESM2-8/EC/mean/v1',
    )
    second_b = replace(second_a, run_id='second-b', seed=4)
    third_a = replace(
        first_a,
        run_id='third-a',
        seed=5,
        data_name='DeepLoc-2',
        dataset_key='DeepLoc-2/split/v1',
    )
    third_b = replace(third_a, run_id='third-b', seed=6)
    plan = plan_parallel_probe_runs((first_a, first_b, second_a, second_b, third_a, third_b))
    estimate = estimate_parallel_probe_plan(plan, batch_size=8)
    first_two_group_bytes = (
        estimate.group_estimates[0].estimated_peak_bytes
        + estimate.group_estimates[1].estimated_peak_bytes
    )
    first_two_group_flops = (
        estimate.group_estimates[0].group_training_flops_per_batch
        + estimate.group_estimates[1].group_training_flops_per_batch
    )

    schedule = schedule_parallel_probe_execution_waves(
        estimate,
        max_wave_peak_bytes=first_two_group_bytes,
        max_groups_per_wave=2,
        target_training_flops_per_wave=first_two_group_flops,
    )

    assert schedule.total_waves == 2
    assert schedule.total_groups == 3
    assert schedule.total_runs == 6
    assert schedule.peak_wave_estimated_peak_bytes == first_two_group_bytes
    assert schedule.target_satisfied_wave_count == 1
    assert schedule.target_underfilled_wave_count == 1
    assert schedule.over_memory_budget_wave_count == 0
    assert schedule.waves[0].group_indices == (0, 1)
    assert schedule.waves[0].trainer_invocations == 2
    assert schedule.waves[1].group_indices == (2,)


def test_wave_schedule_reports_over_budget_singleton_groups() -> None:
    first_a = _run_spec(run_id='first-a', seed=1)
    first_b = replace(first_a, run_id='first-b', seed=2)
    second_a = replace(
        first_a,
        run_id='second-a',
        seed=3,
        model_name='ESM2-8',
        embedding_key='ESM2-8/EC/mean/v1',
    )
    second_b = replace(second_a, run_id='second-b', seed=4)
    plan = plan_parallel_probe_runs((first_a, first_b, second_a, second_b))
    estimate = estimate_parallel_probe_plan(plan, batch_size=8)
    too_small_budget = estimate.group_estimates[0].estimated_peak_bytes - 1

    schedule = schedule_parallel_probe_execution_waves(
        estimate,
        max_wave_peak_bytes=too_small_budget,
        max_groups_per_wave=2,
    )

    assert schedule.total_waves == 2
    assert schedule.over_memory_budget_wave_count == 2


def test_wave_schedule_rejects_invalid_packing_args() -> None:
    specs = build_seed_run_specs(
        run_id_prefix='EC/ESM2-35',
        base_seed=42,
        num_runs=2,
        model_name='ESM2-35',
        data_name='EC',
        embedding_key='ESM2-35/EC/mean/v1',
        dataset_key='EC/split/v1',
        trainer_key='epochs=1|batch=8|lr=1e-4',
        probe_type='mlp',
        input_size=8,
        hidden_size=4,
        dropout=0.0,
        num_labels=2,
        n_layers=0,
        task_type='singlelabel',
        pre_ln=True,
        use_bias=True,
    )
    estimate = estimate_parallel_probe_plan(plan_parallel_probe_runs(specs))

    with pytest.raises(AssertionError, match="max_wave_peak_bytes"):
        schedule_parallel_probe_execution_waves(estimate, max_wave_peak_bytes=0)
    with pytest.raises(AssertionError, match="max_groups_per_wave"):
        schedule_parallel_probe_execution_waves(estimate, max_groups_per_wave=0)
    with pytest.raises(AssertionError, match="target_training_flops_per_wave"):
        schedule_parallel_probe_execution_waves(estimate, target_training_flops_per_wave=-1)
