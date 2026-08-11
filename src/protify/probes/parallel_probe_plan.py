from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from ..model_components.mlp import intermediate_correction_fn
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn
    except ImportError:
        from model_components.mlp import intermediate_correction_fn
from .get_probe import normalize_probe_type


ParallelProbeCompatibilityKey = Tuple[object, ...]


@dataclass(frozen=True)
class ParallelProbeRunSpec:
    """A declarative description of one probe run before execution."""

    run_id: str
    seed: int
    model_name: str
    data_name: str
    embedding_key: str
    dataset_key: str
    trainer_key: str
    probe_type: str
    input_size: int
    hidden_size: int
    dropout: float
    num_labels: int
    n_layers: int
    task_type: str
    pre_ln: bool
    use_bias: bool
    batch_mode: str = 'shared'
    index_strategy: str = 'permutation'
    tokenwise: bool = False
    matrix_embed: bool = False
    full_finetuning: bool = False
    save_model: bool = False

    def __post_init__(self) -> None:
        # Specs arrive from probe arguments, planning scripts, and saved manifests, so
        # the former probe name is resolved here rather than at every comparison.
        object.__setattr__(self, 'probe_type', normalize_probe_type(self.probe_type))

    def is_parallel_linear_eligible(self) -> bool:
        return (
            self.probe_type == 'mlp'
            and not self.tokenwise
            and not self.matrix_embed
            and not self.full_finetuning
        )

    def ineligibility_reasons(self) -> Tuple[str, ...]:
        reasons = []
        if self.probe_type != 'mlp':
            reasons.append('probe_type')
        if self.tokenwise:
            reasons.append('tokenwise')
        if self.matrix_embed:
            reasons.append('matrix_embed')
        if self.full_finetuning:
            reasons.append('full_finetuning')
        return tuple(reasons)

    def compatibility_key(self) -> ParallelProbeCompatibilityKey:
        assert self.is_parallel_linear_eligible(), (
            f"Run {self.run_id} is not eligible for parallel linear probe grouping: "
            f"{self.ineligibility_reasons()}"
        )
        return (
            self.model_name,
            self.data_name,
            self.embedding_key,
            self.dataset_key,
            self.trainer_key,
            self.probe_type,
            self.input_size,
            self.hidden_size,
            self.dropout,
            self.num_labels,
            self.n_layers,
            self.task_type,
            self.pre_ln,
            self.use_bias,
            self.batch_mode,
            self.index_strategy,
            self.save_model,
        )


@dataclass(frozen=True)
class ParallelProbeGroup:
    """A group of runs that can share one vectorized Trainer pass."""

    runs: Tuple[ParallelProbeRunSpec, ...]
    eligible: bool

    def __post_init__(self) -> None:
        assert len(self.runs) > 0, "ParallelProbeGroup requires at least one run."
        if not self.eligible:
            assert len(self.runs) == 1, "Ineligible runs must remain single-run groups."
            return

        reference_key = self.runs[0].compatibility_key()
        for run in self.runs:
            assert run.is_parallel_linear_eligible(), (
                f"Run {run.run_id} is not eligible for this parallel group: "
                f"{run.ineligibility_reasons()}"
            )
            assert run.compatibility_key() == reference_key, (
                f"Run {run.run_id} is incompatible with group key {reference_key}."
            )

    @property
    def can_vectorize(self) -> bool:
        return self.eligible and len(self.runs) > 1

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def run_ids(self) -> Tuple[str, ...]:
        return tuple(run.run_id for run in self.runs)

    @property
    def run_seeds(self) -> Tuple[int, ...]:
        return tuple(run.seed for run in self.runs)

    @property
    def compatibility_key(self) -> ParallelProbeCompatibilityKey:
        if self.eligible:
            return self.runs[0].compatibility_key()
        return ('sequential', self.runs[0].run_id)

    @property
    def execution_kind(self) -> str:
        if self.can_vectorize:
            return 'vectorized'
        if self.eligible:
            return 'eligible_singleton'
        return 'sequential_fallback'

    @property
    def fallback_reasons(self) -> Tuple[str, ...]:
        if self.can_vectorize:
            return ()
        if self.eligible:
            return ('group_size',)
        return self.runs[0].ineligibility_reasons()


@dataclass(frozen=True)
class ParallelProbeExecutionPlan:
    """A pure execution plan for a model/data/probe run universe."""

    groups: Tuple[ParallelProbeGroup, ...]

    @property
    def total_runs(self) -> int:
        return sum(group.num_runs for group in self.groups)

    @property
    def vectorized_groups(self) -> Tuple[ParallelProbeGroup, ...]:
        return tuple(group for group in self.groups if group.can_vectorize)

    @property
    def sequential_groups(self) -> Tuple[ParallelProbeGroup, ...]:
        return tuple(group for group in self.groups if not group.can_vectorize)

    @property
    def vectorized_runs(self) -> int:
        return sum(group.num_runs for group in self.vectorized_groups)

    @property
    def sequential_runs(self) -> int:
        return sum(group.num_runs for group in self.sequential_groups)

    @property
    def trainer_invocations(self) -> int:
        return len(self.groups)

    @property
    def invocation_reduction(self) -> int:
        return self.total_runs - self.trainer_invocations

    @property
    def compression_ratio(self) -> float:
        if self.trainer_invocations == 0:
            return 1.0
        return float(self.total_runs) / float(self.trainer_invocations)

    def execution_groups(self, prefer_largest_parallel: bool = True) -> Tuple[ParallelProbeGroup, ...]:
        if not prefer_largest_parallel:
            return self.groups
        indexed_groups = tuple(enumerate(self.groups))

        def sort_key(item: Tuple[int, ParallelProbeGroup]) -> Tuple[int, int, int]:
            index, group = item
            vectorized_rank = 0 if group.can_vectorize else 1
            return (vectorized_rank, -group.num_runs, index)

        return tuple(group for _index, group in sorted(indexed_groups, key=sort_key))

    def summary_rows(self) -> Tuple[Tuple[str, int, Tuple[str, ...], Tuple[str, ...]], ...]:
        rows = []
        for group in self.groups:
            rows.append(
                (
                    group.execution_kind,
                    group.num_runs,
                    group.run_ids,
                    group.fallback_reasons,
                )
            )
        return tuple(rows)

    def assert_nonempty(self) -> None:
        assert self.total_runs > 0, "ParallelProbeExecutionPlan requires at least one run."


@dataclass(frozen=True)
class ParallelProbeGroupEstimate:
    """Static resource estimate for one planned probe execution group."""

    execution_kind: str
    num_runs: int
    run_ids: Tuple[str, ...]
    parameter_count_known: bool
    single_probe_parameter_count: int
    group_parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    training_state_bytes: int
    batch_size: int
    batch_activation_bytes: int
    logit_bytes: int
    dataset_size: int
    run_specific_index_bytes: int
    estimated_peak_bytes: int
    single_probe_forward_flops_per_sample: int
    group_forward_flops_per_batch: int
    group_training_flops_per_batch: int


@dataclass(frozen=True)
class ParallelProbePlanEstimate:
    """Static resource estimate for a full parallel-probe execution plan."""

    group_estimates: Tuple[ParallelProbeGroupEstimate, ...]
    dtype_bytes: int
    optimizer_state_multiplier: int
    training_flop_multiplier: int = 3
    batch_size: int = 0
    dataset_size: int = 0
    include_run_specific_index: bool = False
    index_dtype_bytes: int = 8

    @property
    def total_parameter_count(self) -> int:
        return sum(estimate.group_parameter_count for estimate in self.group_estimates)

    @property
    def total_training_state_bytes(self) -> int:
        return sum(estimate.training_state_bytes for estimate in self.group_estimates)

    @property
    def peak_group_training_state_bytes(self) -> int:
        if len(self.group_estimates) == 0:
            return 0
        return max(estimate.training_state_bytes for estimate in self.group_estimates)

    @property
    def total_batch_activation_bytes(self) -> int:
        return sum(estimate.batch_activation_bytes for estimate in self.group_estimates)

    @property
    def peak_group_batch_activation_bytes(self) -> int:
        if len(self.group_estimates) == 0:
            return 0
        return max(estimate.batch_activation_bytes for estimate in self.group_estimates)

    @property
    def total_run_specific_index_bytes(self) -> int:
        return sum(estimate.run_specific_index_bytes for estimate in self.group_estimates)

    @property
    def peak_group_estimated_peak_bytes(self) -> int:
        if len(self.group_estimates) == 0:
            return 0
        return max(estimate.estimated_peak_bytes for estimate in self.group_estimates)

    @property
    def unknown_group_count(self) -> int:
        return sum(1 for estimate in self.group_estimates if not estimate.parameter_count_known)

    @property
    def total_forward_flops_per_batch(self) -> int:
        return sum(estimate.group_forward_flops_per_batch for estimate in self.group_estimates)

    @property
    def peak_group_forward_flops_per_batch(self) -> int:
        if len(self.group_estimates) == 0:
            return 0
        return max(estimate.group_forward_flops_per_batch for estimate in self.group_estimates)

    @property
    def total_training_flops_per_batch(self) -> int:
        return sum(estimate.group_training_flops_per_batch for estimate in self.group_estimates)

    @property
    def peak_group_training_flops_per_batch(self) -> int:
        if len(self.group_estimates) == 0:
            return 0
        return max(estimate.group_training_flops_per_batch for estimate in self.group_estimates)

    def summary_dict(self) -> Dict[str, object]:
        return {
            "total_parameter_count": self.total_parameter_count,
            "total_training_state_bytes": self.total_training_state_bytes,
            "peak_group_training_state_bytes": self.peak_group_training_state_bytes,
            "total_batch_activation_bytes": self.total_batch_activation_bytes,
            "peak_group_batch_activation_bytes": self.peak_group_batch_activation_bytes,
            "total_run_specific_index_bytes": self.total_run_specific_index_bytes,
            "peak_group_estimated_peak_bytes": self.peak_group_estimated_peak_bytes,
            "unknown_group_count": self.unknown_group_count,
            "dtype_bytes": self.dtype_bytes,
            "optimizer_state_multiplier": self.optimizer_state_multiplier,
            "training_flop_multiplier": self.training_flop_multiplier,
            "total_forward_flops_per_batch": self.total_forward_flops_per_batch,
            "peak_group_forward_flops_per_batch": self.peak_group_forward_flops_per_batch,
            "total_training_flops_per_batch": self.total_training_flops_per_batch,
            "peak_group_training_flops_per_batch": self.peak_group_training_flops_per_batch,
            "batch_size": self.batch_size,
            "dataset_size": self.dataset_size,
            "include_run_specific_index": self.include_run_specific_index,
            "index_dtype_bytes": self.index_dtype_bytes,
        }


@dataclass(frozen=True)
class ParallelProbeExecutionWave:
    """A static co-scheduling wave of planned Trainer invocations."""

    group_indices: Tuple[int, ...]
    group_run_counts: Tuple[int, ...]
    group_execution_kinds: Tuple[str, ...]
    group_run_ids: Tuple[Tuple[str, ...], ...]
    total_runs: int
    trainer_invocations: int
    concurrent_estimated_peak_bytes: int
    forward_flops_per_batch: int
    training_flops_per_batch: int

    def summary_dict(self) -> Dict[str, object]:
        return {
            "group_indices": list(self.group_indices),
            "group_run_counts": list(self.group_run_counts),
            "group_execution_kinds": list(self.group_execution_kinds),
            "group_run_ids": [list(run_ids) for run_ids in self.group_run_ids],
            "total_runs": self.total_runs,
            "trainer_invocations": self.trainer_invocations,
            "concurrent_estimated_peak_bytes": self.concurrent_estimated_peak_bytes,
            "forward_flops_per_batch": self.forward_flops_per_batch,
            "training_flops_per_batch": self.training_flops_per_batch,
        }


@dataclass(frozen=True)
class ParallelProbeWaveSchedule:
    """A static packing plan for co-scheduling parallel-probe groups."""

    waves: Tuple[ParallelProbeExecutionWave, ...]
    max_wave_peak_bytes: Optional[int] = None
    max_groups_per_wave: Optional[int] = 1
    target_training_flops_per_wave: int = 0

    @property
    def total_waves(self) -> int:
        return len(self.waves)

    @property
    def total_groups(self) -> int:
        return sum(wave.trainer_invocations for wave in self.waves)

    @property
    def total_runs(self) -> int:
        return sum(wave.total_runs for wave in self.waves)

    @property
    def peak_wave_estimated_peak_bytes(self) -> int:
        if len(self.waves) == 0:
            return 0
        return max(wave.concurrent_estimated_peak_bytes for wave in self.waves)

    @property
    def total_training_flops_per_batch(self) -> int:
        return sum(wave.training_flops_per_batch for wave in self.waves)

    @property
    def target_satisfied_wave_count(self) -> int:
        if self.target_training_flops_per_wave <= 0:
            return 0
        return sum(
            1 for wave in self.waves
            if wave.training_flops_per_batch >= self.target_training_flops_per_wave
        )

    @property
    def target_underfilled_wave_count(self) -> int:
        if self.target_training_flops_per_wave <= 0:
            return 0
        return self.total_waves - self.target_satisfied_wave_count

    @property
    def over_memory_budget_wave_count(self) -> int:
        if self.max_wave_peak_bytes is None:
            return 0
        return sum(
            1 for wave in self.waves
            if wave.concurrent_estimated_peak_bytes > self.max_wave_peak_bytes
        )

    def summary_dict(self) -> Dict[str, object]:
        return {
            "total_waves": self.total_waves,
            "total_groups": self.total_groups,
            "total_runs": self.total_runs,
            "max_wave_peak_bytes": self.max_wave_peak_bytes,
            "max_groups_per_wave": self.max_groups_per_wave,
            "target_training_flops_per_wave": self.target_training_flops_per_wave,
            "peak_wave_estimated_peak_bytes": self.peak_wave_estimated_peak_bytes,
            "total_training_flops_per_batch": self.total_training_flops_per_batch,
            "target_satisfied_wave_count": self.target_satisfied_wave_count,
            "target_underfilled_wave_count": self.target_underfilled_wave_count,
            "over_memory_budget_wave_count": self.over_memory_budget_wave_count,
            "waves": [wave.summary_dict() for wave in self.waves],
        }


def linear_probe_parameter_count(
        input_size: int,
        hidden_size: int,
        num_labels: int,
        n_layers: int,
        pre_ln: bool,
        use_bias: bool,
    ) -> int:
    assert input_size > 0, "input_size must be positive."
    assert hidden_size > 0, "hidden_size must be positive."
    assert num_labels > 0, "num_labels must be positive."
    assert n_layers >= 0, "n_layers must be non-negative."

    count = 0
    if pre_ln:
        count += input_size * 2
    count += input_size * hidden_size
    if use_bias:
        count += hidden_size

    for _layer_idx in range(n_layers):
        count += hidden_size * hidden_size
        if use_bias:
            count += hidden_size

    proj_dim = intermediate_correction_fn(2, num_labels)
    count += hidden_size * 2
    count += hidden_size * proj_dim
    if use_bias:
        count += proj_dim
    count += proj_dim * num_labels
    if use_bias:
        count += num_labels
    return count


def linear_probe_parameter_count_for_spec(spec: ParallelProbeRunSpec) -> int:
    assert spec.probe_type == 'mlp', "Only MLP probe specs have known parameter counts."
    return linear_probe_parameter_count(
        input_size=spec.input_size,
        hidden_size=spec.hidden_size,
        num_labels=spec.num_labels,
        n_layers=spec.n_layers,
        pre_ln=spec.pre_ln,
        use_bias=spec.use_bias,
    )


def linear_probe_batch_activation_count(
        input_size: int,
        hidden_size: int,
        num_labels: int,
        n_layers: int,
    ) -> int:
    assert input_size > 0, "input_size must be positive."
    assert hidden_size > 0, "hidden_size must be positive."
    assert num_labels > 0, "num_labels must be positive."
    assert n_layers >= 0, "n_layers must be non-negative."

    proj_dim = intermediate_correction_fn(2, num_labels)
    return input_size + (hidden_size * (n_layers + 2)) + proj_dim + num_labels


def linear_probe_forward_flop_count(
        input_size: int,
        hidden_size: int,
        num_labels: int,
        n_layers: int,
    ) -> int:
    assert input_size > 0, "input_size must be positive."
    assert hidden_size > 0, "hidden_size must be positive."
    assert num_labels > 0, "num_labels must be positive."
    assert n_layers >= 0, "n_layers must be non-negative."

    proj_dim = intermediate_correction_fn(2, num_labels)
    matmul_macs = input_size * hidden_size
    matmul_macs += n_layers * hidden_size * hidden_size
    matmul_macs += hidden_size * proj_dim
    matmul_macs += proj_dim * num_labels
    return 2 * matmul_macs


def estimate_parallel_probe_group(
        group: ParallelProbeGroup,
        dtype_bytes: int = 4,
        optimizer_state_multiplier: int = 2,
        training_flop_multiplier: int = 3,
        batch_size: int = 0,
        dataset_size: int = 0,
        include_run_specific_index: bool = False,
        index_dtype_bytes: int = 8,
    ) -> ParallelProbeGroupEstimate:
    assert dtype_bytes > 0, "dtype_bytes must be positive."
    assert optimizer_state_multiplier >= 0, "optimizer_state_multiplier must be non-negative."
    assert training_flop_multiplier > 0, "training_flop_multiplier must be positive."
    assert batch_size >= 0, "batch_size must be non-negative."
    assert dataset_size >= 0, "dataset_size must be non-negative."
    assert index_dtype_bytes > 0, "index_dtype_bytes must be positive."

    parameter_count_known = group.runs[0].probe_type == 'mlp'
    if parameter_count_known:
        single_probe_parameter_count = linear_probe_parameter_count_for_spec(group.runs[0])
        batch_activation_count = linear_probe_batch_activation_count(
            input_size=group.runs[0].input_size,
            hidden_size=group.runs[0].hidden_size,
            num_labels=group.runs[0].num_labels,
            n_layers=group.runs[0].n_layers,
        )
        single_probe_forward_flops_per_sample = linear_probe_forward_flop_count(
            input_size=group.runs[0].input_size,
            hidden_size=group.runs[0].hidden_size,
            num_labels=group.runs[0].num_labels,
            n_layers=group.runs[0].n_layers,
        )
    else:
        single_probe_parameter_count = 0
        batch_activation_count = 0
        single_probe_forward_flops_per_sample = 0
    group_parameter_count = single_probe_parameter_count * group.num_runs
    parameter_bytes = group_parameter_count * dtype_bytes
    gradient_bytes = parameter_bytes
    optimizer_state_bytes = parameter_bytes * optimizer_state_multiplier
    training_state_bytes = parameter_bytes + gradient_bytes + optimizer_state_bytes
    batch_activation_bytes = batch_size * group.num_runs * batch_activation_count * dtype_bytes
    if parameter_count_known:
        logit_bytes = batch_size * group.num_runs * group.runs[0].num_labels * dtype_bytes
    else:
        logit_bytes = 0
    if include_run_specific_index and group.eligible and group.runs[0].batch_mode == 'run_specific':
        run_specific_index_bytes = dataset_size * group.num_runs * index_dtype_bytes
    else:
        run_specific_index_bytes = 0
    estimated_peak_bytes = training_state_bytes + batch_activation_bytes + run_specific_index_bytes
    group_forward_flops_per_batch = batch_size * group.num_runs * single_probe_forward_flops_per_sample
    group_training_flops_per_batch = group_forward_flops_per_batch * training_flop_multiplier
    return ParallelProbeGroupEstimate(
        execution_kind=group.execution_kind,
        num_runs=group.num_runs,
        run_ids=group.run_ids,
        parameter_count_known=parameter_count_known,
        single_probe_parameter_count=single_probe_parameter_count,
        group_parameter_count=group_parameter_count,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        training_state_bytes=training_state_bytes,
        batch_size=batch_size,
        batch_activation_bytes=batch_activation_bytes,
        logit_bytes=logit_bytes,
        dataset_size=dataset_size,
        run_specific_index_bytes=run_specific_index_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        single_probe_forward_flops_per_sample=single_probe_forward_flops_per_sample,
        group_forward_flops_per_batch=group_forward_flops_per_batch,
        group_training_flops_per_batch=group_training_flops_per_batch,
    )


def estimate_parallel_probe_plan(
        plan: ParallelProbeExecutionPlan,
        dtype_bytes: int = 4,
        optimizer_state_multiplier: int = 2,
        training_flop_multiplier: int = 3,
        batch_size: int = 0,
        dataset_size: int = 0,
        include_run_specific_index: bool = False,
        index_dtype_bytes: int = 8,
    ) -> ParallelProbePlanEstimate:
    assert batch_size >= 0, "batch_size must be non-negative."
    assert dataset_size >= 0, "dataset_size must be non-negative."
    assert index_dtype_bytes > 0, "index_dtype_bytes must be positive."
    return ParallelProbePlanEstimate(
        group_estimates=tuple(
            estimate_parallel_probe_group(
                group,
                dtype_bytes=dtype_bytes,
                optimizer_state_multiplier=optimizer_state_multiplier,
                training_flop_multiplier=training_flop_multiplier,
                batch_size=batch_size,
                dataset_size=dataset_size,
                include_run_specific_index=include_run_specific_index,
                index_dtype_bytes=index_dtype_bytes,
            )
            for group in plan.groups
        ),
        dtype_bytes=dtype_bytes,
        optimizer_state_multiplier=optimizer_state_multiplier,
        training_flop_multiplier=training_flop_multiplier,
        batch_size=batch_size,
        dataset_size=dataset_size,
        include_run_specific_index=include_run_specific_index,
        index_dtype_bytes=index_dtype_bytes,
    )


def _build_execution_wave(
        items: Tuple[Tuple[int, ParallelProbeGroupEstimate], ...],
    ) -> ParallelProbeExecutionWave:
    assert len(items) > 0, "Execution waves require at least one group estimate."
    group_indices = tuple(index for index, _estimate in items)
    group_run_counts = tuple(estimate.num_runs for _index, estimate in items)
    group_execution_kinds = tuple(estimate.execution_kind for _index, estimate in items)
    group_run_ids = tuple(estimate.run_ids for _index, estimate in items)
    return ParallelProbeExecutionWave(
        group_indices=group_indices,
        group_run_counts=group_run_counts,
        group_execution_kinds=group_execution_kinds,
        group_run_ids=group_run_ids,
        total_runs=sum(estimate.num_runs for _index, estimate in items),
        trainer_invocations=len(items),
        concurrent_estimated_peak_bytes=sum(estimate.estimated_peak_bytes for _index, estimate in items),
        forward_flops_per_batch=sum(estimate.group_forward_flops_per_batch for _index, estimate in items),
        training_flops_per_batch=sum(estimate.group_training_flops_per_batch for _index, estimate in items),
    )


def schedule_parallel_probe_execution_waves(
        plan_estimate: ParallelProbePlanEstimate,
        max_wave_peak_bytes: Optional[int] = None,
        max_groups_per_wave: Optional[int] = 1,
        target_training_flops_per_wave: int = 0,
        prefer_largest_first: bool = True,
    ) -> ParallelProbeWaveSchedule:
    """Pack planned Trainer invocations into static co-scheduling waves.

    This does not start processes or train models. It produces a no-training
    launch matrix that can be validated later on workstation GPUs.
    """

    if max_wave_peak_bytes is not None:
        assert max_wave_peak_bytes > 0, "max_wave_peak_bytes must be positive when provided."
    if max_groups_per_wave is not None:
        assert max_groups_per_wave > 0, "max_groups_per_wave must be positive when provided."
    assert target_training_flops_per_wave >= 0, (
        "target_training_flops_per_wave must be non-negative."
    )

    indexed_estimates = tuple(enumerate(plan_estimate.group_estimates))
    if prefer_largest_first:
        ordered_estimates = tuple(
            sorted(
                indexed_estimates,
                key=lambda item: (
                    -item[1].estimated_peak_bytes,
                    -item[1].group_training_flops_per_batch,
                    item[0],
                ),
            )
        )
    else:
        ordered_estimates = indexed_estimates

    waves: List[List[Tuple[int, ParallelProbeGroupEstimate]]] = []
    wave_peak_bytes: List[int] = []
    for item in ordered_estimates:
        _group_index, estimate = item
        placed = False
        for wave_idx, wave_items in enumerate(waves):
            group_count_allowed = (
                max_groups_per_wave is None
                or len(wave_items) < max_groups_per_wave
            )
            if not group_count_allowed:
                continue
            memory_allowed = (
                max_wave_peak_bytes is None
                or wave_peak_bytes[wave_idx] + estimate.estimated_peak_bytes <= max_wave_peak_bytes
            )
            if not memory_allowed:
                continue
            wave_items.append(item)
            wave_peak_bytes[wave_idx] += estimate.estimated_peak_bytes
            placed = True
            break
        if not placed:
            waves.append([item])
            wave_peak_bytes.append(estimate.estimated_peak_bytes)

    execution_waves = tuple(_build_execution_wave(tuple(wave_items)) for wave_items in waves)
    return ParallelProbeWaveSchedule(
        waves=execution_waves,
        max_wave_peak_bytes=max_wave_peak_bytes,
        max_groups_per_wave=max_groups_per_wave,
        target_training_flops_per_wave=target_training_flops_per_wave,
    )


def max_linear_probe_runs_for_training_state_budget(
        spec: ParallelProbeRunSpec,
        memory_budget_bytes: int,
        dtype_bytes: int = 4,
        optimizer_state_multiplier: int = 2,
    ) -> int:
    assert memory_budget_bytes > 0, "memory_budget_bytes must be positive."
    assert dtype_bytes > 0, "dtype_bytes must be positive."
    assert optimizer_state_multiplier >= 0, "optimizer_state_multiplier must be non-negative."
    single_parameter_count = linear_probe_parameter_count_for_spec(spec)
    bytes_per_run = single_parameter_count * dtype_bytes * (2 + optimizer_state_multiplier)
    assert bytes_per_run > 0, "bytes_per_run must be positive."
    return max(1, memory_budget_bytes // bytes_per_run)


def max_linear_probe_runs_for_estimated_peak_budget(
        spec: ParallelProbeRunSpec,
        memory_budget_bytes: int,
        batch_size: int,
        dataset_size: int = 0,
        include_run_specific_index: bool = False,
        dtype_bytes: int = 4,
        optimizer_state_multiplier: int = 2,
        index_dtype_bytes: int = 8,
    ) -> int:
    assert memory_budget_bytes > 0, "memory_budget_bytes must be positive."
    assert batch_size >= 0, "batch_size must be non-negative."
    assert dataset_size >= 0, "dataset_size must be non-negative."
    assert dtype_bytes > 0, "dtype_bytes must be positive."
    assert optimizer_state_multiplier >= 0, "optimizer_state_multiplier must be non-negative."
    assert index_dtype_bytes > 0, "index_dtype_bytes must be positive."

    single_parameter_count = linear_probe_parameter_count_for_spec(spec)
    training_state_bytes_per_run = single_parameter_count * dtype_bytes * (2 + optimizer_state_multiplier)
    batch_activation_count = linear_probe_batch_activation_count(
        input_size=spec.input_size,
        hidden_size=spec.hidden_size,
        num_labels=spec.num_labels,
        n_layers=spec.n_layers,
    )
    activation_bytes_per_run = batch_size * batch_activation_count * dtype_bytes
    if include_run_specific_index and spec.batch_mode == 'run_specific':
        index_bytes_per_run = dataset_size * index_dtype_bytes
    else:
        index_bytes_per_run = 0
    bytes_per_run = training_state_bytes_per_run + activation_bytes_per_run + index_bytes_per_run
    assert bytes_per_run > 0, "bytes_per_run must be positive."
    return max(1, memory_budget_bytes // bytes_per_run)


def build_seed_run_specs(
        run_id_prefix: str,
        base_seed: int,
        num_runs: int,
        model_name: str,
        data_name: str,
        embedding_key: str,
        dataset_key: str,
        trainer_key: str,
        probe_type: str,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_labels: int,
        n_layers: int,
        task_type: str,
        pre_ln: bool,
        use_bias: bool,
        batch_mode: str = 'shared',
        index_strategy: str = 'permutation',
        tokenwise: bool = False,
        matrix_embed: bool = False,
        full_finetuning: bool = False,
        save_model: bool = False,
    ) -> Tuple[ParallelProbeRunSpec, ...]:
    assert num_runs > 0, "num_runs must be positive."
    specs = []
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        specs.append(
            ParallelProbeRunSpec(
                run_id=f"{run_id_prefix}/seed-{seed}",
                seed=seed,
                model_name=model_name,
                data_name=data_name,
                embedding_key=embedding_key,
                dataset_key=dataset_key,
                trainer_key=trainer_key,
                probe_type=probe_type,
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                num_labels=num_labels,
                n_layers=n_layers,
                task_type=task_type,
                pre_ln=pre_ln,
                use_bias=use_bias,
                batch_mode=batch_mode,
                index_strategy=index_strategy,
                tokenwise=tokenwise,
                matrix_embed=matrix_embed,
                full_finetuning=full_finetuning,
                save_model=save_model,
            )
        )
    return tuple(specs)


def _chunk_bucket(
        items: List[Tuple[int, ParallelProbeRunSpec]],
        max_parallel_group_size: Optional[int],
    ) -> Tuple[Tuple[int, Tuple[ParallelProbeRunSpec, ...]], ...]:
    if max_parallel_group_size is None:
        return ((items[0][0], tuple(spec for _index, spec in items)),)

    assert max_parallel_group_size > 0, "max_parallel_group_size must be positive when provided."
    chunks = []
    for start in range(0, len(items), max_parallel_group_size):
        chunk_items = items[start:start + max_parallel_group_size]
        chunks.append((chunk_items[0][0], tuple(spec for _index, spec in chunk_items)))
    return tuple(chunks)


def group_parallel_probe_runs(
        run_specs: Iterable[ParallelProbeRunSpec],
        max_parallel_group_size: Optional[int] = None,
        max_parallel_group_size_by_key: Optional[Dict[ParallelProbeCompatibilityKey, int]] = None,
    ) -> Tuple[ParallelProbeGroup, ...]:
    if max_parallel_group_size is not None:
        assert max_parallel_group_size > 0, "max_parallel_group_size must be positive when provided."
    if max_parallel_group_size_by_key is not None:
        for key, keyed_max_group_size in max_parallel_group_size_by_key.items():
            assert keyed_max_group_size > 0, (
                f"max_parallel_group_size_by_key value for {key} must be positive when provided."
            )
    buckets: Dict[ParallelProbeCompatibilityKey, List[Tuple[int, ParallelProbeRunSpec]]] = {}
    bucket_order: List[ParallelProbeCompatibilityKey] = []
    group_candidates: List[Tuple[int, ParallelProbeGroup]] = []

    for index, spec in enumerate(run_specs):
        if spec.is_parallel_linear_eligible():
            key = spec.compatibility_key()
            if key not in buckets:
                buckets[key] = []
                bucket_order.append(key)
            buckets[key].append((index, spec))
        else:
            group_candidates.append((index, ParallelProbeGroup(runs=(spec,), eligible=False)))

    for key in bucket_order:
        key_max_parallel_group_size = max_parallel_group_size
        if max_parallel_group_size_by_key is not None and key in max_parallel_group_size_by_key:
            keyed_max_group_size = max_parallel_group_size_by_key[key]
            if key_max_parallel_group_size is None:
                key_max_parallel_group_size = keyed_max_group_size
            else:
                key_max_parallel_group_size = min(key_max_parallel_group_size, keyed_max_group_size)
        for first_index, chunk_specs in _chunk_bucket(buckets[key], key_max_parallel_group_size):
            group_candidates.append((first_index, ParallelProbeGroup(runs=chunk_specs, eligible=True)))

    group_candidates.sort(key=lambda item: item[0])
    return tuple(group for _index, group in group_candidates)


def plan_parallel_probe_runs(
        run_specs: Iterable[ParallelProbeRunSpec],
        max_parallel_group_size: Optional[int] = None,
        max_parallel_group_size_by_key: Optional[Dict[ParallelProbeCompatibilityKey, int]] = None,
    ) -> ParallelProbeExecutionPlan:
    plan = ParallelProbeExecutionPlan(
        groups=group_parallel_probe_runs(
            run_specs,
            max_parallel_group_size=max_parallel_group_size,
            max_parallel_group_size_by_key=max_parallel_group_size_by_key,
        )
    )
    plan.assert_nonempty()
    return plan
