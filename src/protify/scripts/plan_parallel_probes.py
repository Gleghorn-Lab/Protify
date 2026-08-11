import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass

try:
    from probes.parallel_probe_plan import (
        ParallelProbeCompatibilityKey,
        ParallelProbeExecutionPlan,
        ParallelProbeExecutionWave,
        ParallelProbeGroup,
        ParallelProbeGroupEstimate,
        ParallelProbePlanEstimate,
        ParallelProbeRunSpec,
        ParallelProbeWaveSchedule,
        build_seed_run_specs,
        estimate_parallel_probe_plan,
        linear_probe_batch_activation_count,
        linear_probe_forward_flop_count,
        linear_probe_parameter_count,
        max_linear_probe_runs_for_estimated_peak_budget,
        max_linear_probe_runs_for_training_state_budget,
        plan_parallel_probe_runs,
        schedule_parallel_probe_execution_waves,
    )
    from probes.get_probe import normalize_probe_type
except ImportError:
    from protify.probes.parallel_probe_plan import (
        ParallelProbeCompatibilityKey,
        ParallelProbeExecutionPlan,
        ParallelProbeExecutionWave,
        ParallelProbeGroup,
        ParallelProbeGroupEstimate,
        ParallelProbePlanEstimate,
        ParallelProbeRunSpec,
        ParallelProbeWaveSchedule,
        build_seed_run_specs,
        estimate_parallel_probe_plan,
        linear_probe_batch_activation_count,
        linear_probe_forward_flop_count,
        linear_probe_parameter_count,
        max_linear_probe_runs_for_estimated_peak_budget,
        max_linear_probe_runs_for_training_state_budget,
        plan_parallel_probe_runs,
        schedule_parallel_probe_execution_waves,
    )
    from protify.probes.get_probe import normalize_probe_type


GroupSizeCaps = dict[ParallelProbeCompatibilityKey, int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a no-training parallel linear-probe execution plan for a model/dataset sweep."
    )
    parser.add_argument("--model_names", nargs="+", required=True)
    parser.add_argument("--data_names", nargs="+", required=True)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=8)
    parser.add_argument("--input_size", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, default=8192)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help=(
            "Planner-only label count for memory and FLOP estimates. "
            "Generated python -m main commands do not include this because Protify "
            "infers num_labels from the dataset."
        ),
    )
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--probe_hidden_sizes", nargs="+", type=int, default=None)
    parser.add_argument("--probe_dropouts", nargs="+", type=float, default=None)
    parser.add_argument("--probe_n_layers", nargs="+", type=int, default=None)
    parser.add_argument(
        "--task_type",
        choices=["singlelabel", "multilabel", "regression", "sigmoid_regression"],
        default="singlelabel",
    )
    parser.add_argument("--probe_type", choices=["mlp", "linear", "transformer", "lyra"], default="mlp")
    parser.add_argument("--no_pre_ln", dest="pre_ln", action="store_false", default=True)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--tokenwise", action="store_true")
    parser.add_argument("--matrix_embed", action="store_true")
    parser.add_argument("--full_finetuning", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--parallel_batch_mode", choices=["shared", "run_specific"], default="shared")
    parser.add_argument("--parallel_index_strategy", choices=["permutation", "affine"], default="permutation")
    parser.add_argument("--parallel_max_group_size", type=int, default=None)
    parser.add_argument("--parallel_max_grad_norm", type=float, default=0.0)
    parser.add_argument("--parallel_grad_clip_mode", choices=["none", "global", "per_run"], default="global")
    parser.add_argument("--training_state_budget_gb", type=float, default=None)
    parser.add_argument("--estimated_peak_budget_gb", type=float, default=None)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--train_dataset_size", type=int, default=0)
    parser.add_argument("--embedding_save_dir", type=str, default="embeddings")
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--embedding_num_workers", type=int, default=0)
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean", "var"])
    parser.add_argument("--embedding_hidden_state_index", type=int, default=-1)
    parser.add_argument(
        "--embed_dtype",
        choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
        default=None,
    )
    parser.add_argument("--sql", action="store_true")
    parser.add_argument("--download_embeddings", action="store_true")
    parser.add_argument("--dtype_bytes", type=int, default=4)
    parser.add_argument("--index_dtype_bytes", type=int, default=8)
    parser.add_argument("--optimizer_state_multiplier", type=int, default=2)
    parser.add_argument("--training_flop_multiplier", type=int, default=3)
    parser.add_argument("--wave_memory_budget_gb", type=float, default=None)
    parser.add_argument("--wave_max_groups", type=int, default=1)
    parser.add_argument("--wave_target_training_flops_per_batch", type=int, default=0)
    parser.add_argument("--gpu_peak_tflops", type=float, default=None)
    parser.add_argument("--gpu_memory_bandwidth_gbps", type=float, default=None)
    parser.add_argument("--gpu_indices", nargs="+", type=int, default=None)
    parser.add_argument("--gpu_assignment_mode", choices=["packed", "round_robin"], default="packed")
    parser.add_argument("--telemetry_dir", type=str, default="telemetry")
    parser.add_argument("--monitor_interval_seconds", type=float, default=1.0)
    parser.add_argument("--monitor_gpu_index", type=int, default=None)
    parser.add_argument("--json_indent", type=int, default=2)
    return validate_args(parser.parse_args(argv))


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    # Normalize before anything derives a key from it, so a plan built with the former
    # probe name matches one built with the current name.
    args.probe_type = normalize_probe_type(args.probe_type)
    assert len(args.model_names) > 0, "At least one model name is required."
    assert len(args.data_names) > 0, "At least one dataset name is required."
    assert args.num_runs > 0, "num_runs must be positive."
    assert args.input_size > 0, "input_size must be positive."
    assert args.hidden_size > 0, "hidden_size must be positive."
    assert 0.0 <= args.dropout < 1.0, "dropout must be in [0, 1)."
    assert args.num_labels > 0, "num_labels must be positive."
    assert args.n_layers >= 0, "n_layers must be non-negative."
    assert args.probe_batch_size >= 0, "probe_batch_size must be non-negative."
    assert args.train_dataset_size >= 0, "train_dataset_size must be non-negative."
    assert args.embedding_save_dir.strip() != "", "embedding_save_dir must be non-empty."
    assert args.embedding_batch_size > 0, "embedding_batch_size must be positive."
    assert args.embedding_num_workers >= 0, "embedding_num_workers must be non-negative."
    assert len(args.embedding_pooling_types) > 0, "embedding_pooling_types must be non-empty."
    for pooling_type in args.embedding_pooling_types:
        assert pooling_type.strip() != "", "embedding_pooling_types entries must be non-empty."
    assert args.dtype_bytes > 0, "dtype_bytes must be positive."
    assert args.index_dtype_bytes > 0, "index_dtype_bytes must be positive."
    assert args.optimizer_state_multiplier >= 0, "optimizer_state_multiplier must be non-negative."
    assert args.training_flop_multiplier > 0, "training_flop_multiplier must be positive."
    assert args.wave_max_groups > 0, "wave_max_groups must be positive."
    assert args.wave_target_training_flops_per_batch >= 0, (
        "wave_target_training_flops_per_batch must be non-negative."
    )
    assert args.telemetry_dir.strip() != "", "telemetry_dir must be non-empty."
    assert args.monitor_interval_seconds > 0.0, "monitor_interval_seconds must be positive."
    if args.monitor_gpu_index is not None:
        assert args.monitor_gpu_index >= 0, "monitor_gpu_index must be non-negative when provided."
    if args.gpu_indices is not None:
        assert len(args.gpu_indices) > 0, "gpu_indices must be non-empty when provided."
        for gpu_index in args.gpu_indices:
            assert gpu_index >= 0, "All gpu_indices values must be non-negative."
        assert len(set(args.gpu_indices)) == len(args.gpu_indices), "gpu_indices values must be unique."
    assert args.json_indent >= 0, "json_indent must be non-negative."
    if args.parallel_max_group_size is not None:
        assert args.parallel_max_group_size > 0, "parallel_max_group_size must be positive when provided."
    assert args.parallel_max_grad_norm >= 0.0, "parallel_max_grad_norm must be non-negative."
    if args.training_state_budget_gb is not None:
        assert args.training_state_budget_gb > 0.0, "training_state_budget_gb must be positive when provided."
    if args.estimated_peak_budget_gb is not None:
        assert args.estimated_peak_budget_gb > 0.0, "estimated_peak_budget_gb must be positive when provided."
    if args.wave_memory_budget_gb is not None:
        assert args.wave_memory_budget_gb > 0.0, "wave_memory_budget_gb must be positive when provided."
    if args.gpu_peak_tflops is not None:
        assert args.gpu_peak_tflops > 0.0, "gpu_peak_tflops must be positive when provided."
    if args.gpu_memory_bandwidth_gbps is not None:
        assert args.gpu_memory_bandwidth_gbps > 0.0, (
            "gpu_memory_bandwidth_gbps must be positive when provided."
        )
    if args.task_type == "singlelabel":
        assert args.num_labels > 1, "singlelabel plans require num_labels > 1."
    for hidden_size in probe_hidden_sizes(args):
        assert hidden_size > 0, "All probe hidden sizes must be positive."
    for dropout in probe_dropouts(args):
        assert 0.0 <= dropout < 1.0, "All probe dropouts must be in [0, 1)."
    for n_layers in probe_n_layers(args):
        assert n_layers >= 0, "All probe n_layers values must be non-negative."
    return args


@dataclass(frozen=True)
class ProbePlanConfig:
    hidden_size: int
    dropout: float
    n_layers: int

    @property
    def label(self) -> str:
        dropout_label = str(self.dropout).replace(".", "p")
        return f"h{self.hidden_size}_l{self.n_layers}_d{dropout_label}"

    def to_report(self) -> dict[str, object]:
        return {
            "label": self.label,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "n_layers": self.n_layers,
        }


def probe_config_parameter_count(
    args: argparse.Namespace,
    probe_config: ProbePlanConfig,
) -> int:
    if args.probe_type not in ("mlp", "linear"):
        return 0
    return linear_probe_parameter_count(
        input_size=args.input_size,
        hidden_size=probe_config.hidden_size,
        num_labels=args.num_labels,
        n_layers=probe_config.n_layers,
        pre_ln=args.pre_ln,
        use_bias=args.use_bias,
    )


def probe_config_recommendation(
    args: argparse.Namespace,
    probe_config: ProbePlanConfig,
) -> dict[str, object]:
    parameter_count = probe_config_parameter_count(args, probe_config)
    bytes_per_run = parameter_count * args.dtype_bytes * (2 + args.optimizer_state_multiplier)
    if args.probe_type in ("mlp", "linear"):
        batch_activation_count = linear_probe_batch_activation_count(
            input_size=args.input_size,
            hidden_size=probe_config.hidden_size,
            num_labels=args.num_labels,
            n_layers=probe_config.n_layers,
        )
        forward_flops_per_sample = linear_probe_forward_flop_count(
            input_size=args.input_size,
            hidden_size=probe_config.hidden_size,
            num_labels=args.num_labels,
            n_layers=probe_config.n_layers,
        )
    else:
        batch_activation_count = 0
        forward_flops_per_sample = 0
    batch_activation_bytes_per_run = args.probe_batch_size * batch_activation_count * args.dtype_bytes
    training_flops_per_batch_per_run = (
        args.probe_batch_size
        * forward_flops_per_sample
        * args.training_flop_multiplier
    )
    if args.parallel_batch_mode == "run_specific" and args.parallel_index_strategy == "permutation":
        run_specific_index_bytes_per_run = args.train_dataset_size * args.index_dtype_bytes
    else:
        run_specific_index_bytes_per_run = 0
    estimated_peak_bytes_per_run = (
        bytes_per_run
        + batch_activation_bytes_per_run
        + run_specific_index_bytes_per_run
    )
    requested_group_size = args.num_runs
    if args.parallel_max_group_size is not None:
        requested_group_size = min(requested_group_size, args.parallel_max_group_size)

    budget_bytes = None
    budget_limited_group_size = None
    estimated_peak_budget_bytes = None
    estimated_peak_budget_limited_group_size = None
    recommended_group_size = requested_group_size
    if args.training_state_budget_gb is not None:
        budget_bytes = int(args.training_state_budget_gb * (1024 ** 3))
        if bytes_per_run > 0:
            budget_limited_group_size = max(1, budget_bytes // bytes_per_run)
            recommended_group_size = min(requested_group_size, budget_limited_group_size)
        else:
            budget_limited_group_size = 1
            recommended_group_size = 1
    if args.estimated_peak_budget_gb is not None:
        estimated_peak_budget_bytes = int(args.estimated_peak_budget_gb * (1024 ** 3))
        if estimated_peak_bytes_per_run > 0:
            estimated_peak_budget_limited_group_size = max(
                1,
                estimated_peak_budget_bytes // estimated_peak_bytes_per_run,
            )
            recommended_group_size = min(
                recommended_group_size,
                estimated_peak_budget_limited_group_size,
            )
        else:
            estimated_peak_budget_limited_group_size = 1
            recommended_group_size = 1

    return {
        "label": probe_config.label,
        "hidden_size": probe_config.hidden_size,
        "dropout": probe_config.dropout,
        "n_layers": probe_config.n_layers,
        "parameter_count_known": args.probe_type in ("mlp", "linear"),
        "single_probe_parameter_count": parameter_count,
        "training_state_bytes_per_run": bytes_per_run,
        "batch_activation_bytes_per_run": batch_activation_bytes_per_run,
        "forward_flops_per_sample": forward_flops_per_sample,
        "training_flops_per_batch_per_run": training_flops_per_batch_per_run,
        "run_specific_index_bytes_per_run": run_specific_index_bytes_per_run,
        "estimated_peak_bytes_per_run": estimated_peak_bytes_per_run,
        "requested_group_size": requested_group_size,
        "training_state_budget_bytes": budget_bytes,
        "budget_limited_group_size": budget_limited_group_size,
        "estimated_peak_budget_bytes": estimated_peak_budget_bytes,
        "estimated_peak_budget_limited_group_size": estimated_peak_budget_limited_group_size,
        "recommended_group_size": recommended_group_size,
    }


def probe_hidden_sizes(args: argparse.Namespace) -> tuple[int, ...]:
    if args.probe_hidden_sizes is None:
        return (args.hidden_size,)
    return tuple(args.probe_hidden_sizes)


def probe_dropouts(args: argparse.Namespace) -> tuple[float, ...]:
    if args.probe_dropouts is None:
        return (args.dropout,)
    return tuple(args.probe_dropouts)


def probe_n_layers(args: argparse.Namespace) -> tuple[int, ...]:
    if args.probe_n_layers is None:
        return (args.n_layers,)
    return tuple(args.probe_n_layers)


def probe_plan_configs(args: argparse.Namespace) -> tuple[ProbePlanConfig, ...]:
    configs = []
    for hidden_size in probe_hidden_sizes(args):
        for dropout in probe_dropouts(args):
            for n_layers in probe_n_layers(args):
                configs.append(
                    ProbePlanConfig(
                        hidden_size=hidden_size,
                        dropout=dropout,
                        n_layers=n_layers,
                    )
                )
    return tuple(configs)


def trainer_key(args: argparse.Namespace, probe_config: ProbePlanConfig) -> str:
    return (
        f"batch_mode={args.parallel_batch_mode}|"
        f"index_strategy={args.parallel_index_strategy}|"
        f"max_group={args.parallel_max_group_size}|"
        f"max_grad_norm={args.parallel_max_grad_norm}|"
        f"grad_clip_mode={args.parallel_grad_clip_mode}|"
        f"probe={args.probe_type}|"
        f"hidden={probe_config.hidden_size}|"
        f"layers={probe_config.n_layers}|"
        f"dropout={probe_config.dropout}"
    )


def build_universe_run_specs(args: argparse.Namespace) -> tuple[ParallelProbeRunSpec, ...]:
    specs = []
    probe_configs = probe_plan_configs(args)
    include_probe_label = len(probe_configs) > 1
    embedding_kind = "matrix" if args.matrix_embed else "pooled"
    for model_name in args.model_names:
        for data_name in args.data_names:
            for probe_config in probe_configs:
                run_id_prefix = f"{data_name}/{model_name}"
                if include_probe_label:
                    run_id_prefix = f"{run_id_prefix}/{probe_config.label}"
                specs.extend(
                    build_seed_run_specs(
                        run_id_prefix=run_id_prefix,
                        base_seed=args.base_seed,
                        num_runs=args.num_runs,
                        model_name=model_name,
                        data_name=data_name,
                        embedding_key=f"{model_name}/{data_name}/{embedding_kind}",
                        dataset_key=f"{data_name}/default",
                        trainer_key=trainer_key(args, probe_config),
                        probe_type=args.probe_type,
                        input_size=args.input_size,
                        hidden_size=probe_config.hidden_size,
                        dropout=probe_config.dropout,
                        num_labels=args.num_labels,
                        n_layers=probe_config.n_layers,
                        task_type=args.task_type,
                        pre_ln=args.pre_ln,
                        use_bias=args.use_bias,
                        batch_mode=args.parallel_batch_mode,
                        index_strategy=args.parallel_index_strategy,
                        tokenwise=args.tokenwise,
                        matrix_embed=args.matrix_embed,
                        full_finetuning=args.full_finetuning,
                        save_model=args.save_model,
                    )
                )
    return tuple(specs)


def effective_max_group_size_for_spec(
    args: argparse.Namespace,
    spec: ParallelProbeRunSpec,
    parallel_max_group_size: int | None = None,
) -> int | None:
    candidate_group_sizes = []
    if parallel_max_group_size is None:
        parallel_max_group_size = args.parallel_max_group_size
    if parallel_max_group_size is not None:
        candidate_group_sizes.append(parallel_max_group_size)
    if spec.is_parallel_linear_eligible() and args.training_state_budget_gb is not None:
        training_state_budget_bytes = int(args.training_state_budget_gb * (1024 ** 3))
        candidate_group_sizes.append(
            max_linear_probe_runs_for_training_state_budget(
                spec,
                memory_budget_bytes=training_state_budget_bytes,
                dtype_bytes=args.dtype_bytes,
                optimizer_state_multiplier=args.optimizer_state_multiplier,
            )
        )
    if spec.is_parallel_linear_eligible() and args.estimated_peak_budget_gb is not None:
        estimated_peak_budget_bytes = int(args.estimated_peak_budget_gb * (1024 ** 3))
        candidate_group_sizes.append(
            max_linear_probe_runs_for_estimated_peak_budget(
                spec,
                memory_budget_bytes=estimated_peak_budget_bytes,
                batch_size=args.probe_batch_size,
                dataset_size=args.train_dataset_size,
                include_run_specific_index=(
                    args.parallel_batch_mode == "run_specific"
                    and args.parallel_index_strategy == "permutation"
                ),
                dtype_bytes=args.dtype_bytes,
                optimizer_state_multiplier=args.optimizer_state_multiplier,
                index_dtype_bytes=args.index_dtype_bytes,
            )
        )
    if len(candidate_group_sizes) == 0:
        return None
    return min(args.num_runs, min(candidate_group_sizes))


def effective_max_group_size_by_key(
    args: argparse.Namespace,
    run_specs: Sequence[ParallelProbeRunSpec],
    parallel_max_group_size: int | None = None,
) -> GroupSizeCaps | None:
    max_group_size_by_key = {}
    for spec in run_specs:
        if spec.is_parallel_linear_eligible():
            max_group_size = effective_max_group_size_for_spec(
                args,
                spec,
                parallel_max_group_size=parallel_max_group_size,
            )
            if max_group_size is not None:
                key = spec.compatibility_key()
                if key not in max_group_size_by_key or max_group_size < max_group_size_by_key[key]:
                    max_group_size_by_key[key] = max_group_size
    if len(max_group_size_by_key) == 0:
        return None
    return max_group_size_by_key


def group_size_cap_report(
    max_group_size_by_key: GroupSizeCaps | None,
) -> list[dict[str, object]]:
    if max_group_size_by_key is None:
        return []
    reports = []
    for key in sorted(max_group_size_by_key.keys(), key=lambda item: str(item)):
        reports.append(
            {
                "model_name": key[0],
                "data_name": key[1],
                "embedding_key": key[2],
                "dataset_key": key[3],
                "trainer_key": key[4],
                "probe_type": key[5],
                "input_size": key[6],
                "hidden_size": key[7],
                "dropout": key[8],
                "num_labels": key[9],
                "n_layers": key[10],
                "task_type": key[11],
                "pre_ln": key[12],
                "use_bias": key[13],
                "batch_mode": key[14],
                "index_strategy": key[15],
                "save_model": key[16],
                "max_group_size": max_group_size_by_key[key],
            }
        )
    return reports


def applied_group_size_cap(
    group: ParallelProbeGroup,
    max_group_size_by_key: GroupSizeCaps | None,
) -> int | None:
    if not group.eligible or max_group_size_by_key is None:
        return None
    key = group.compatibility_key
    if key not in max_group_size_by_key:
        return None
    return max_group_size_by_key[key]


def group_report(
    group: ParallelProbeGroup,
    estimate: ParallelProbeGroupEstimate,
    max_group_size_by_key: GroupSizeCaps | None = None,
) -> dict[str, object]:
    representative = group.runs[0]
    return {
        "execution_kind": group.execution_kind,
        "eligible": group.eligible,
        "can_vectorize": group.can_vectorize,
        "model_name": representative.model_name,
        "data_name": representative.data_name,
        "embedding_key": representative.embedding_key,
        "dataset_key": representative.dataset_key,
        "trainer_key": representative.trainer_key,
        "probe_type": representative.probe_type,
        "input_size": representative.input_size,
        "hidden_size": representative.hidden_size,
        "dropout": representative.dropout,
        "num_labels": representative.num_labels,
        "n_layers": representative.n_layers,
        "task_type": representative.task_type,
        "pre_ln": representative.pre_ln,
        "use_bias": representative.use_bias,
        "batch_mode": representative.batch_mode,
        "index_strategy": representative.index_strategy,
        "save_model": representative.save_model,
        "tokenwise": representative.tokenwise,
        "matrix_embed": representative.matrix_embed,
        "full_finetuning": representative.full_finetuning,
        "num_runs": group.num_runs,
        "applied_group_size_cap": applied_group_size_cap(group, max_group_size_by_key),
        "run_ids": list(group.run_ids),
        "run_seeds": list(group.run_seeds),
        "fallback_reasons": list(group.fallback_reasons),
        "parameter_count_known": estimate.parameter_count_known,
        "single_probe_parameter_count": estimate.single_probe_parameter_count,
        "group_parameter_count": estimate.group_parameter_count,
        "parameter_bytes": estimate.parameter_bytes,
        "gradient_bytes": estimate.gradient_bytes,
        "optimizer_state_bytes": estimate.optimizer_state_bytes,
        "training_state_bytes": estimate.training_state_bytes,
        "batch_size": estimate.batch_size,
        "batch_activation_bytes": estimate.batch_activation_bytes,
        "logit_bytes": estimate.logit_bytes,
        "dataset_size": estimate.dataset_size,
        "run_specific_index_bytes": estimate.run_specific_index_bytes,
        "estimated_peak_bytes": estimate.estimated_peak_bytes,
        "single_probe_forward_flops_per_sample": estimate.single_probe_forward_flops_per_sample,
        "group_forward_flops_per_batch": estimate.group_forward_flops_per_batch,
        "group_training_flops_per_batch": estimate.group_training_flops_per_batch,
    }


def group_size_sweep_candidates(args: argparse.Namespace) -> tuple[int, ...]:
    candidates = [1]
    size = 2
    while size < args.num_runs:
        candidates.append(size)
        size *= 2
    if args.num_runs not in candidates:
        candidates.append(args.num_runs)
    if args.parallel_max_group_size is not None and args.parallel_max_group_size not in candidates:
        candidates.append(args.parallel_max_group_size)
    return tuple(sorted(set(candidates)))


def group_size_sweep_report(
    args: argparse.Namespace,
    run_specs: Sequence[ParallelProbeRunSpec],
) -> list[dict[str, object]]:
    reports = []
    has_budget_cap = (
        args.training_state_budget_gb is not None
        or args.estimated_peak_budget_gb is not None
    )
    for group_size in group_size_sweep_candidates(args):
        max_group_size_by_key = effective_max_group_size_by_key(
            args,
            run_specs,
            parallel_max_group_size=group_size,
        )
        plan = plan_parallel_probe_runs(
            run_specs,
            max_parallel_group_size_by_key=max_group_size_by_key,
        )
        report = {
            "parallel_probe_max_group_size": group_size,
            "trainer_invocations": plan.trainer_invocations,
            "invocation_reduction": plan.invocation_reduction,
            "compression_ratio": plan.compression_ratio,
            "vectorized_runs": plan.vectorized_runs,
            "sequential_runs": plan.sequential_runs,
        }
        if has_budget_cap:
            effective_caps = [] if max_group_size_by_key is None else list(max_group_size_by_key.values())
            if len(effective_caps) == 0:
                max_effective_group_size = None
                budget_constrained = False
            else:
                max_effective_group_size = max(effective_caps)
                budget_constrained = min(effective_caps) < group_size
            report["max_effective_group_size"] = max_effective_group_size
            report["budget_constrained"] = budget_constrained
        reports.append(report)
    return reports


def _estimate_plan_for_args(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
) -> ParallelProbePlanEstimate:
    return estimate_parallel_probe_plan(
        plan,
        dtype_bytes=args.dtype_bytes,
        optimizer_state_multiplier=args.optimizer_state_multiplier,
        training_flop_multiplier=args.training_flop_multiplier,
        batch_size=args.probe_batch_size,
        dataset_size=args.train_dataset_size,
        include_run_specific_index=(
            args.parallel_batch_mode == "run_specific"
            and args.parallel_index_strategy == "permutation"
        ),
        index_dtype_bytes=args.index_dtype_bytes,
    )


def recommendation_group_size_candidates(args: argparse.Namespace) -> tuple[int, ...]:
    candidates = group_size_sweep_candidates(args)
    if args.parallel_max_group_size is not None:
        candidates = tuple(
            group_size for group_size in candidates
            if group_size <= args.parallel_max_group_size
        )
    assert len(candidates) > 0, "Recommendation candidate set must be non-empty."
    return candidates


def effective_group_size_summary(
    max_group_size_by_key: GroupSizeCaps | None,
    requested_group_size: int,
) -> dict[str, object]:
    if max_group_size_by_key is None:
        return {
            "min_effective_group_size": requested_group_size,
            "max_effective_group_size": requested_group_size,
            "budget_constrained": False,
        }
    effective_sizes = list(max_group_size_by_key.values())
    return {
        "min_effective_group_size": min(effective_sizes),
        "max_effective_group_size": max(effective_sizes),
        "budget_constrained": min(effective_sizes) < requested_group_size,
    }


def parallel_cli_args_for_group_size(
    args: argparse.Namespace,
    group_size: int,
) -> list[str]:
    cli_args = [
        "--num_runs",
        str(args.num_runs),
        "--parallel_probe_runs",
        "--parallel_probe_batch_mode",
        args.parallel_batch_mode,
        "--parallel_probe_index_strategy",
        args.parallel_index_strategy,
        "--parallel_probe_max_group_size",
        str(group_size),
    ]
    if args.training_state_budget_gb is not None:
        cli_args.extend([
            "--parallel_probe_training_state_budget_gb",
            str(args.training_state_budget_gb),
        ])
    if args.estimated_peak_budget_gb is not None:
        cli_args.extend([
            "--parallel_probe_estimated_peak_budget_gb",
            str(args.estimated_peak_budget_gb),
        ])
    if args.parallel_max_grad_norm > 0.0:
        cli_args.extend(["--parallel_probe_max_grad_norm", str(args.parallel_max_grad_norm)])
    if args.parallel_grad_clip_mode != "global":
        cli_args.extend(["--parallel_probe_grad_clip_mode", args.parallel_grad_clip_mode])
    return cli_args


def manifest_runner_template_args(
    args: argparse.Namespace,
    execute: bool,
    variant: str,
) -> list[str]:
    assert variant in ("parallel", "sequential", "both"), "variant must be parallel, sequential, or both."
    if args.wave_max_groups > 1 and variant == "parallel":
        wave_execution_mode = "concurrent"
    else:
        wave_execution_mode = "sequential"
    if execute:
        if variant == "both":
            output_path = f"{args.telemetry_dir}/manifest_runner_execute.report.json"
        else:
            output_path = f"{args.telemetry_dir}/manifest_runner_{variant}_execute.report.json"
    else:
        if variant == "both":
            output_path = f"{args.telemetry_dir}/manifest_runner_dry_run.report.json"
        else:
            output_path = f"{args.telemetry_dir}/manifest_runner_{variant}_dry_run.report.json"
    cli_args = [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        output_path,
        "--variant",
        variant,
        "--use_monitor",
        "--wave_execution_mode",
        wave_execution_mode,
    ]
    if execute:
        cli_args.append("--execute")
    return cli_args


def manifest_embedding_runner_template_args(
    args: argparse.Namespace,
    execute: bool,
) -> list[str]:
    if execute:
        output_path = f"{args.telemetry_dir}/manifest_runner_embeddings_execute.report.json"
    else:
        output_path = f"{args.telemetry_dir}/manifest_runner_embeddings_dry_run.report.json"
    cli_args = [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        output_path,
        "--phase",
        "embeddings",
        "--wave_execution_mode",
        "sequential",
    ]
    if execute:
        cli_args.append("--execute")
    return cli_args


def execution_recommendation_candidate(
    args: argparse.Namespace,
    run_specs: Sequence[ParallelProbeRunSpec],
    requested_group_size: int,
) -> dict[str, object]:
    max_group_size_by_key = effective_max_group_size_by_key(
        args,
        run_specs,
        parallel_max_group_size=requested_group_size,
    )
    plan = plan_parallel_probe_runs(
        run_specs,
        max_parallel_group_size_by_key=max_group_size_by_key,
    )
    estimate = _estimate_plan_for_args(args, plan)
    schedule = execution_wave_schedule(args, estimate)
    effective_summary = effective_group_size_summary(
        max_group_size_by_key,
        requested_group_size,
    )
    has_vectorized_work = plan.vectorized_runs > 0
    memory_fit = schedule.over_memory_budget_wave_count == 0
    target_fit = (
        args.wave_target_training_flops_per_batch == 0
        or schedule.target_underfilled_wave_count == 0
    )
    feasible = has_vectorized_work and memory_fit
    roofline = hardware_roofline_report(args, schedule)
    return {
        "parallel_probe_max_group_size": requested_group_size,
        "min_effective_group_size": effective_summary["min_effective_group_size"],
        "max_effective_group_size": effective_summary["max_effective_group_size"],
        "budget_constrained": effective_summary["budget_constrained"],
        "feasible": feasible,
        "has_vectorized_work": has_vectorized_work,
        "memory_fit": memory_fit,
        "target_fit": target_fit,
        "trainer_invocations": plan.trainer_invocations,
        "invocation_reduction": plan.invocation_reduction,
        "compression_ratio": plan.compression_ratio,
        "vectorized_runs": plan.vectorized_runs,
        "sequential_runs": plan.sequential_runs,
        "vectorized_group_count": len(plan.vectorized_groups),
        "eligible_singleton_group_count": sum(
            1 for group in plan.groups
            if group.eligible and not group.can_vectorize
        ),
        "ineligible_group_count": sum(1 for group in plan.groups if not group.eligible),
        "peak_group_estimated_peak_bytes": estimate.peak_group_estimated_peak_bytes,
        "peak_wave_estimated_peak_bytes": schedule.peak_wave_estimated_peak_bytes,
        "total_training_flops_per_batch": estimate.total_training_flops_per_batch,
        "peak_group_training_flops_per_batch": estimate.peak_group_training_flops_per_batch,
        "wave_count": schedule.total_waves,
        "target_underfilled_wave_count": schedule.target_underfilled_wave_count,
        "over_memory_budget_wave_count": schedule.over_memory_budget_wave_count,
        "roofline_available": roofline["available"],
        "roofline_total_seconds_per_batch": roofline["total_roofline_seconds_per_batch"],
        "roofline_peak_wave_seconds_per_batch": roofline["peak_wave_roofline_seconds_per_batch"],
        "roofline_compute_bound_wave_count": roofline["compute_bound_wave_count"],
        "roofline_memory_bound_wave_count": roofline["memory_bound_wave_count"],
    }


def _recommendation_sort_key(candidate):
    feasible_rank = 1 if candidate["feasible"] else 0
    memory_rank = 1 if candidate["memory_fit"] else 0
    target_rank = 1 if candidate["target_fit"] else 0
    return (
        feasible_rank,
        memory_rank,
        target_rank,
        candidate["compression_ratio"],
        candidate["vectorized_runs"],
        candidate["invocation_reduction"],
        -candidate["over_memory_budget_wave_count"],
        -candidate["target_underfilled_wave_count"],
        -candidate["peak_wave_estimated_peak_bytes"],
        candidate["parallel_probe_max_group_size"],
    )


def execution_recommendation_report(
    args: argparse.Namespace,
    run_specs: Sequence[ParallelProbeRunSpec],
) -> dict[str, object]:
    candidates = [
        execution_recommendation_candidate(args, run_specs, group_size)
        for group_size in recommendation_group_size_candidates(args)
    ]
    selected_candidate = max(candidates, key=_recommendation_sort_key)
    selected_group_size = selected_candidate["parallel_probe_max_group_size"]
    if selected_candidate["feasible"]:
        status = "recommended"
    elif selected_candidate["has_vectorized_work"]:
        status = "needs_adjustment"
    else:
        status = "not_recommended"
    selected = dict(selected_candidate)
    selected["status"] = status
    selected["parallel_cli_args"] = parallel_cli_args_for_group_size(args, selected_group_size)
    selected["manifest_runner_dry_run_args"] = manifest_runner_template_args(
        args,
        execute=False,
        variant="both",
    )
    selected["manifest_runner_embeddings_dry_run_args"] = manifest_embedding_runner_template_args(
        args,
        execute=False,
    )
    selected["manifest_runner_embeddings_execute_args"] = manifest_embedding_runner_template_args(
        args,
        execute=True,
    )
    selected["manifest_runner_execute_args"] = manifest_runner_template_args(
        args,
        execute=True,
        variant="both",
    )
    selected["manifest_runner_sequential_execute_args"] = manifest_runner_template_args(
        args,
        execute=True,
        variant="sequential",
    )
    selected["manifest_runner_parallel_execute_args"] = manifest_runner_template_args(
        args,
        execute=True,
        variant="parallel",
    )
    selected["selection_rule"] = (
        "Prefer feasible vectorized plans with memory-fit waves, target-fit waves, "
        "higher trainer-invocation compression, more vectorized runs, and lower peak wave memory."
    )
    return {
        "selected": selected,
        "candidates": candidates,
        "candidate_group_sizes": [
            candidate["parallel_probe_max_group_size"]
            for candidate in candidates
        ],
        "explicit_parallel_max_group_size_respected": args.parallel_max_group_size is not None,
    }


def wave_memory_budget_bytes(args: argparse.Namespace) -> int | None:
    if args.wave_memory_budget_gb is None:
        return None
    return int(args.wave_memory_budget_gb * (1024 ** 3))


def execution_wave_schedule(
    args: argparse.Namespace,
    estimate: ParallelProbePlanEstimate,
) -> ParallelProbeWaveSchedule:
    return schedule_parallel_probe_execution_waves(
        estimate,
        max_wave_peak_bytes=wave_memory_budget_bytes(args),
        max_groups_per_wave=args.wave_max_groups,
        target_training_flops_per_wave=args.wave_target_training_flops_per_batch,
    )


def execution_wave_report(schedule: ParallelProbeWaveSchedule) -> dict[str, object]:
    return schedule.summary_dict()


def gpu_peak_flops_per_second(args: argparse.Namespace) -> float | None:
    if args.gpu_peak_tflops is None:
        return None
    return float(args.gpu_peak_tflops) * 1_000_000_000_000.0


def gpu_memory_bandwidth_bytes_per_second(args: argparse.Namespace) -> float | None:
    if args.gpu_memory_bandwidth_gbps is None:
        return None
    return float(args.gpu_memory_bandwidth_gbps) * 1_000_000_000.0


def _max_present(values: Sequence[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    if len(present_values) == 0:
        return None
    return max(present_values)


def _roofline_bottleneck(
    compute_seconds: float | None,
    memory_seconds: float | None,
) -> str | None:
    if compute_seconds is None and memory_seconds is None:
        return None
    if compute_seconds is None:
        return "memory_only"
    if memory_seconds is None:
        return "compute_only"
    if compute_seconds > memory_seconds:
        return "compute"
    if memory_seconds > compute_seconds:
        return "memory"
    return "balanced"


def hardware_roofline_report(
    args: argparse.Namespace,
    schedule: ParallelProbeWaveSchedule,
) -> dict[str, object]:
    peak_flops_per_second = gpu_peak_flops_per_second(args)
    memory_bandwidth_bytes_per_second = gpu_memory_bandwidth_bytes_per_second(args)
    profile_available = (
        peak_flops_per_second is not None
        or memory_bandwidth_bytes_per_second is not None
    )
    if not profile_available:
        return {
            "available": False,
            "gpu_peak_tflops": None,
            "gpu_memory_bandwidth_gbps": None,
            "peak_flops_per_second": None,
            "memory_bandwidth_bytes_per_second": None,
            "total_roofline_seconds_per_batch": None,
            "peak_wave_roofline_seconds_per_batch": None,
            "compute_bound_wave_count": 0,
            "memory_bound_wave_count": 0,
            "balanced_wave_count": 0,
            "compute_only_wave_count": 0,
            "memory_only_wave_count": 0,
            "waves": [],
        }

    wave_reports = []
    roofline_values = []
    compute_bound_wave_count = 0
    memory_bound_wave_count = 0
    balanced_wave_count = 0
    compute_only_wave_count = 0
    memory_only_wave_count = 0
    for wave_index, wave in enumerate(schedule.waves):
        if peak_flops_per_second is None:
            compute_seconds = None
        else:
            compute_seconds = wave.training_flops_per_batch / peak_flops_per_second
        if memory_bandwidth_bytes_per_second is None:
            memory_seconds = None
        else:
            memory_seconds = (
                wave.concurrent_estimated_peak_bytes
                / memory_bandwidth_bytes_per_second
            )
        roofline_seconds = _max_present([compute_seconds, memory_seconds])
        bottleneck = _roofline_bottleneck(compute_seconds, memory_seconds)
        if roofline_seconds is not None:
            roofline_values.append(roofline_seconds)
        if bottleneck == "compute":
            compute_bound_wave_count += 1
        elif bottleneck == "memory":
            memory_bound_wave_count += 1
        elif bottleneck == "balanced":
            balanced_wave_count += 1
        elif bottleneck == "compute_only":
            compute_only_wave_count += 1
        elif bottleneck == "memory_only":
            memory_only_wave_count += 1
        wave_reports.append(
            {
                "wave_index": wave_index,
                "wave_id": f"wave-{wave_index + 1}",
                "trainer_invocations": wave.trainer_invocations,
                "total_runs": wave.total_runs,
                "training_flops_per_batch": wave.training_flops_per_batch,
                "concurrent_estimated_peak_bytes": wave.concurrent_estimated_peak_bytes,
                "compute_seconds_per_batch_lower_bound": compute_seconds,
                "memory_seconds_per_batch_lower_bound": memory_seconds,
                "roofline_seconds_per_batch_lower_bound": roofline_seconds,
                "roofline_bottleneck": bottleneck,
            }
        )

    if len(roofline_values) == 0:
        total_roofline_seconds_per_batch = None
        peak_wave_roofline_seconds_per_batch = None
    else:
        total_roofline_seconds_per_batch = sum(roofline_values)
        peak_wave_roofline_seconds_per_batch = max(roofline_values)

    return {
        "available": True,
        "gpu_peak_tflops": args.gpu_peak_tflops,
        "gpu_memory_bandwidth_gbps": args.gpu_memory_bandwidth_gbps,
        "peak_flops_per_second": peak_flops_per_second,
        "memory_bandwidth_bytes_per_second": memory_bandwidth_bytes_per_second,
        "total_roofline_seconds_per_batch": total_roofline_seconds_per_batch,
        "peak_wave_roofline_seconds_per_batch": peak_wave_roofline_seconds_per_batch,
        "compute_bound_wave_count": compute_bound_wave_count,
        "memory_bound_wave_count": memory_bound_wave_count,
        "balanced_wave_count": balanced_wave_count,
        "compute_only_wave_count": compute_only_wave_count,
        "memory_only_wave_count": memory_only_wave_count,
        "waves": wave_reports,
    }


def validation_readiness_report(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
    estimate: ParallelProbePlanEstimate,
    schedule: ParallelProbeWaveSchedule,
) -> dict[str, object]:
    vectorized_group_count = len(plan.vectorized_groups)
    eligible_singleton_group_count = sum(
        1 for group in plan.groups
        if group.eligible and not group.can_vectorize
    )
    ineligible_group_count = sum(1 for group in plan.groups if not group.eligible)
    vectorized_group_sizes = [group.num_runs for group in plan.vectorized_groups]
    warnings = []
    if vectorized_group_count == 0:
        warnings.append("no_vectorized_groups")
    if ineligible_group_count > 0:
        warnings.append("ineligible_groups_present")
    if eligible_singleton_group_count > 0:
        warnings.append("eligible_singleton_groups_present")
    if plan.compression_ratio < 2.0:
        warnings.append("low_invocation_compression")
    if estimate.unknown_group_count > 0:
        warnings.append("unknown_static_estimates")
    if schedule.over_memory_budget_wave_count > 0:
        warnings.append("waves_over_memory_budget")
    if schedule.target_underfilled_wave_count > 0:
        warnings.append("waves_under_target_training_flops")

    if vectorized_group_count == 0:
        status = "not_ready"
    elif schedule.over_memory_budget_wave_count > 0:
        status = "needs_adjustment"
    elif len(warnings) > 0:
        status = "ready_with_cautions"
    else:
        status = "ready"

    if len(vectorized_group_sizes) == 0:
        min_vectorized_group_size = 0
        max_vectorized_group_size = 0
    else:
        min_vectorized_group_size = min(vectorized_group_sizes)
        max_vectorized_group_size = max(vectorized_group_sizes)

    return {
        "status": status,
        "warnings": warnings,
        "total_runs": plan.total_runs,
        "vectorized_runs": plan.vectorized_runs,
        "sequential_runs": plan.sequential_runs,
        "vectorized_group_count": vectorized_group_count,
        "eligible_singleton_group_count": eligible_singleton_group_count,
        "ineligible_group_count": ineligible_group_count,
        "trainer_invocations": plan.trainer_invocations,
        "invocation_reduction": plan.invocation_reduction,
        "compression_ratio": plan.compression_ratio,
        "min_vectorized_group_size": min_vectorized_group_size,
        "max_vectorized_group_size": max_vectorized_group_size,
        "unknown_estimate_group_count": estimate.unknown_group_count,
        "peak_group_estimated_peak_bytes": estimate.peak_group_estimated_peak_bytes,
        "peak_group_training_flops_per_batch": estimate.peak_group_training_flops_per_batch,
        "wave_count": schedule.total_waves,
        "target_underfilled_wave_count": schedule.target_underfilled_wave_count,
        "over_memory_budget_wave_count": schedule.over_memory_budget_wave_count,
    }


def _append_flag(cli_args: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cli_args.append(flag)


def _base_group_cli_args(
    args: argparse.Namespace,
    group: ParallelProbeGroup,
) -> list[str]:
    representative = group.runs[0]
    cli_args = [
        "--model_names",
        representative.model_name,
        "--data_names",
        representative.data_name,
        "--probe_type",
        representative.probe_type,
        "--hidden_size",
        str(representative.hidden_size),
        "--dropout",
        str(representative.dropout),
        "--n_layers",
        str(representative.n_layers),
        "--probe_batch_size",
        str(args.probe_batch_size),
        "--seed",
        str(group.run_seeds[0]),
        "--num_runs",
        str(group.num_runs),
    ]
    _append_flag(cli_args, "--use_bias", representative.use_bias)
    _append_flag(cli_args, "--pre_ln", not representative.pre_ln)
    _append_flag(cli_args, "--tokenwise", representative.tokenwise)
    _append_flag(cli_args, "--matrix_embed", representative.matrix_embed)
    _append_flag(cli_args, "--full_finetuning", representative.full_finetuning)
    _append_flag(cli_args, "--save_model", representative.save_model)
    return cli_args


def sequential_group_cli_args(
    args: argparse.Namespace,
    group: ParallelProbeGroup,
) -> list[str]:
    return _base_group_cli_args(args, group)


def parallel_group_cli_args(
    args: argparse.Namespace,
    group: ParallelProbeGroup,
) -> list[str]:
    if not group.can_vectorize:
        return []
    cli_args = _base_group_cli_args(args, group)
    cli_args.extend(
        [
            "--parallel_probe_runs",
            "--parallel_probe_batch_mode",
            group.runs[0].batch_mode,
            "--parallel_probe_index_strategy",
            group.runs[0].index_strategy,
            "--parallel_probe_max_group_size",
            str(group.num_runs),
        ]
    )
    if args.training_state_budget_gb is not None:
        cli_args.extend([
            "--parallel_probe_training_state_budget_gb",
            str(args.training_state_budget_gb),
        ])
    if args.estimated_peak_budget_gb is not None:
        cli_args.extend([
            "--parallel_probe_estimated_peak_budget_gb",
            str(args.estimated_peak_budget_gb),
        ])
    if args.parallel_max_grad_norm > 0.0:
        cli_args.extend(["--parallel_probe_max_grad_norm", str(args.parallel_max_grad_norm)])
    if args.parallel_grad_clip_mode != "global":
        cli_args.extend(["--parallel_probe_grad_clip_mode", args.parallel_grad_clip_mode])
    return cli_args


def protify_command(cli_args: Sequence[str]) -> list[str]:
    return ["python", "-m", "main"] + list(cli_args)


def embedding_kind_for_args(args: argparse.Namespace) -> str:
    if args.matrix_embed:
        return "matrix"
    return "pooled"


def embedding_prerequisite_cli_args(
    args: argparse.Namespace,
    model_name: str,
    data_name: str,
) -> list[str]:
    cli_args = [
        "--model_names",
        model_name,
        "--data_names",
        data_name,
        "--save_embeddings",
        "--embedding_save_dir",
        args.embedding_save_dir,
        "--embedding_batch_size",
        str(args.embedding_batch_size),
        "--embedding_num_workers",
        str(args.embedding_num_workers),
        "--embedding_hidden_state_index",
        str(args.embedding_hidden_state_index),
    ]
    if not args.matrix_embed:
        cli_args.append("--embedding_pooling_types")
        cli_args.extend(args.embedding_pooling_types)
    if args.embed_dtype is not None:
        cli_args.extend(["--embed_dtype", args.embed_dtype])
    if args.sql:
        cli_args.append("--sql")
    if args.download_embeddings:
        cli_args.append("--download_embeddings")
    if args.matrix_embed:
        cli_args.append("--matrix_embed")
    return cli_args


def embedding_prerequisite_report(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
) -> dict[str, object]:
    embedding_kind = embedding_kind_for_args(args)
    probe_configs = probe_plan_configs(args)
    jobs = []
    job_index = 0
    for model_name in args.model_names:
        for data_name in args.data_names:
            job_index += 1
            cli_args = embedding_prerequisite_cli_args(args, model_name, data_name)
            jobs.append(
                {
                    "command_id": f"embedding-{job_index}",
                    "model_name": model_name,
                    "data_name": data_name,
                    "embedding_key": f"{model_name}/{data_name}/{embedding_kind}",
                    "embedding_kind": embedding_kind,
                    "embedding_save_dir": args.embedding_save_dir,
                    "embedding_batch_size": args.embedding_batch_size,
                    "embedding_num_workers": args.embedding_num_workers,
                    "embedding_pooling_types": (
                        [] if args.matrix_embed else list(args.embedding_pooling_types)
                    ),
                    "embedding_hidden_state_index": args.embedding_hidden_state_index,
                    "embed_dtype": args.embed_dtype,
                    "sql": args.sql,
                    "download_embeddings": args.download_embeddings,
                    "downstream_probe_config_count": len(probe_configs),
                    "downstream_seed_runs": args.num_runs * len(probe_configs),
                    "command_environment": {"_PROTIFY_EMBED_PHASE": "1"},
                    "command": protify_command(cli_args),
                }
            )

    job_count = len(jobs)
    if job_count == 0:
        probe_run_fanout = 0.0
        trainer_invocation_fanout = 0.0
    else:
        probe_run_fanout = float(plan.total_runs) / float(job_count)
        trainer_invocation_fanout = float(plan.trainer_invocations) / float(job_count)

    return {
        "required_before_probe_training": True,
        "embedding_jobs": jobs,
        "embedding_job_count": job_count,
        "downstream_probe_runs": plan.total_runs,
        "downstream_probe_trainer_invocations": plan.trainer_invocations,
        "probe_run_fanout_per_embedding_job": probe_run_fanout,
        "trainer_invocation_fanout_per_embedding_job": trainer_invocation_fanout,
        "probe_training_reuses_cached_embeddings": not args.full_finetuning,
        "parallel_probe_training_reuses_cached_embeddings": (
            plan.vectorized_runs > 0 and not args.full_finetuning
        ),
        "embedding_parallelization_recommendation": (
            "Precompute or download each model/dataset embedding once before probe sweeps. "
            "Treat embedding co-scheduling as a separate workstation experiment because PLM "
            "sizes, sequence lengths, and cache writes can dominate the resource profile."
        ),
        "co_schedule_embedding_and_probe_training": False,
    }


def gpu_assignment_by_group(
    args: argparse.Namespace,
    schedule: ParallelProbeWaveSchedule,
) -> dict[int, int]:
    if args.gpu_indices is None:
        return {}
    assignments = {}
    for wave_index, wave in enumerate(schedule.waves):
        for group_position, group_index in enumerate(wave.group_indices):
            if args.gpu_assignment_mode == "packed":
                gpu_index = args.gpu_indices[wave_index % len(args.gpu_indices)]
            else:
                gpu_index = args.gpu_indices[group_position % len(args.gpu_indices)]
            assignments[group_index] = gpu_index
    return assignments


def group_environment_for_gpu(assigned_gpu_index: int | None) -> dict[str, str]:
    if assigned_gpu_index is None:
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(assigned_gpu_index)}


def telemetry_file_path(
    args: argparse.Namespace,
    command_id: str,
    variant: str,
    suffix: str,
) -> str:
    return f"{args.telemetry_dir}/{command_id}_{variant}.{suffix}"


def monitor_command(
    args: argparse.Namespace,
    command_id: str,
    variant: str,
    command: Sequence[str],
    assigned_gpu_index: int | None = None,
) -> list[str]:
    if len(command) == 0:
        return []
    monitor_gpu_index = args.monitor_gpu_index
    if monitor_gpu_index is None:
        monitor_gpu_index = assigned_gpu_index
    monitor_args = [
        "python",
        "-m",
        "scripts.monitor_parallel_probe_hardware",
        "--output_jsonl",
        telemetry_file_path(args, command_id, variant, "jsonl"),
        "--summary_json",
        telemetry_file_path(args, command_id, variant, "summary.json"),
        "--interval_seconds",
        str(args.monitor_interval_seconds),
    ]
    if monitor_gpu_index is not None:
        monitor_args.extend(["--gpu_index", str(monitor_gpu_index)])
    monitor_args.extend(["--command", "--"])
    monitor_args.extend(command)
    return monitor_args


def group_launch_manifest(
    args: argparse.Namespace,
    group: ParallelProbeGroup,
    estimate: ParallelProbeGroupEstimate,
    group_index: int,
    assigned_gpu_index: int | None = None,
) -> dict[str, object]:
    representative = group.runs[0]
    command_id = f"group-{group_index + 1}"
    sequential_cli_args = sequential_group_cli_args(args, group)
    parallel_cli_args = parallel_group_cli_args(args, group)
    sequential_command = protify_command(sequential_cli_args)
    if len(parallel_cli_args) == 0:
        parallel_command = []
    else:
        parallel_command = protify_command(parallel_cli_args)
    environment = group_environment_for_gpu(assigned_gpu_index)
    return {
        "group_index": group_index,
        "command_id": command_id,
        "model_name": representative.model_name,
        "data_name": representative.data_name,
        "probe_type": representative.probe_type,
        "hidden_size": representative.hidden_size,
        "dropout": representative.dropout,
        "n_layers": representative.n_layers,
        "task_type": representative.task_type,
        "num_labels": representative.num_labels,
        "first_seed": group.run_seeds[0],
        "num_runs": group.num_runs,
        "run_seeds": list(group.run_seeds),
        "parallel_supported": group.can_vectorize,
        "assigned_gpu_index": assigned_gpu_index,
        "environment": environment,
        "sequential_cli_args": sequential_cli_args,
        "parallel_cli_args": parallel_cli_args,
        "sequential_command": sequential_command,
        "parallel_command": parallel_command,
        "sequential_monitor_command": monitor_command(
            args,
            command_id,
            "sequential",
            sequential_command,
            assigned_gpu_index=assigned_gpu_index,
        ),
        "parallel_monitor_command": monitor_command(
            args,
            command_id,
            "parallel",
            parallel_command,
            assigned_gpu_index=assigned_gpu_index,
        ),
        "estimated_peak_bytes": estimate.estimated_peak_bytes,
        "training_flops_per_batch": estimate.group_training_flops_per_batch,
    }


def wave_gpu_assignment_report(
    wave: ParallelProbeExecutionWave,
    group_manifests: Sequence[dict[str, object]],
    group_estimates: Sequence[ParallelProbeGroupEstimate],
    memory_budget_bytes: int | None = None,
) -> list[dict[str, object]]:
    gpu_reports_by_index = {}
    for group_index in wave.group_indices:
        group_manifest = group_manifests[group_index]
        assigned_gpu_index = group_manifest["assigned_gpu_index"]
        if assigned_gpu_index is None:
            continue
        if assigned_gpu_index not in gpu_reports_by_index:
            gpu_reports_by_index[assigned_gpu_index] = {
                "gpu_index": assigned_gpu_index,
                "group_indices": [],
                "command_ids": [],
                "group_count": 0,
                "total_runs": 0,
                "concurrent_estimated_peak_bytes": 0,
                "max_group_estimated_peak_bytes": 0,
                "training_flops_per_batch": 0,
                "memory_budget_bytes": memory_budget_bytes,
                "over_memory_budget": False,
            }
        gpu_report = gpu_reports_by_index[assigned_gpu_index]
        group_estimate = group_estimates[group_index]
        gpu_report["group_indices"].append(group_index)
        gpu_report["command_ids"].append(group_manifest["command_id"])
        gpu_report["group_count"] += 1
        gpu_report["total_runs"] += group_manifest["num_runs"]
        gpu_report["concurrent_estimated_peak_bytes"] += group_estimate.estimated_peak_bytes
        gpu_report["max_group_estimated_peak_bytes"] = max(
            gpu_report["max_group_estimated_peak_bytes"],
            group_estimate.estimated_peak_bytes,
        )
        gpu_report["training_flops_per_batch"] += group_estimate.group_training_flops_per_batch
        gpu_report["over_memory_budget"] = (
            memory_budget_bytes is not None
            and gpu_report["concurrent_estimated_peak_bytes"] > memory_budget_bytes
        )
    return [
        gpu_reports_by_index[gpu_index]
        for gpu_index in sorted(gpu_reports_by_index.keys())
    ]


def launch_manifest_report(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
    estimate: ParallelProbePlanEstimate,
    schedule: ParallelProbeWaveSchedule,
    embedding_prerequisites: dict[str, object] | None = None,
) -> dict[str, object]:
    gpu_assignments = gpu_assignment_by_group(args, schedule)
    gpu_memory_budget_bytes = None
    if args.gpu_indices is not None:
        gpu_memory_budget_bytes = wave_memory_budget_bytes(args)
    group_manifests = [
        group_launch_manifest(
            args,
            group,
            group_estimate,
            group_index,
            assigned_gpu_index=(
                gpu_assignments[group_index]
                if group_index in gpu_assignments
                else None
            ),
        )
        for group_index, (group, group_estimate) in enumerate(zip(plan.groups, estimate.group_estimates))
    ]
    wave_manifests = []
    total_gpu_assignment_count = 0
    total_gpu_over_memory_budget_count = 0
    gpu_over_memory_budget_wave_count = 0
    peak_gpu_estimated_peak_bytes = 0
    for wave_index, wave in enumerate(schedule.waves):
        gpu_assignment_reports = wave_gpu_assignment_report(
            wave,
            group_manifests,
            estimate.group_estimates,
            memory_budget_bytes=gpu_memory_budget_bytes,
        )
        gpu_over_memory_budget_count = sum(
            1 for gpu_report in gpu_assignment_reports
            if gpu_report["over_memory_budget"]
        )
        total_gpu_assignment_count += len(gpu_assignment_reports)
        total_gpu_over_memory_budget_count += gpu_over_memory_budget_count
        if gpu_over_memory_budget_count > 0:
            gpu_over_memory_budget_wave_count += 1
        for gpu_report in gpu_assignment_reports:
            peak_gpu_estimated_peak_bytes = max(
                peak_gpu_estimated_peak_bytes,
                gpu_report["concurrent_estimated_peak_bytes"],
            )
        wave_manifests.append(
            {
                "wave_index": wave_index,
                "wave_id": f"wave-{wave_index + 1}",
                "group_indices": list(wave.group_indices),
                "command_ids": [
                    group_manifests[group_index]["command_id"]
                    for group_index in wave.group_indices
                ],
                "concurrent_estimated_peak_bytes": wave.concurrent_estimated_peak_bytes,
                "training_flops_per_batch": wave.training_flops_per_batch,
                "gpu_memory_budget_bytes": gpu_memory_budget_bytes,
                "gpu_over_memory_budget_count": gpu_over_memory_budget_count,
                "gpu_over_memory_budget": gpu_over_memory_budget_count > 0,
                "gpu_assignments": gpu_assignment_reports,
            }
        )
    return {
        "entrypoint": "python -m main",
        "monitor_entrypoint": "python -m scripts.monitor_parallel_probe_hardware",
        "telemetry_dir": args.telemetry_dir,
        "monitor_interval_seconds": args.monitor_interval_seconds,
        "monitor_gpu_index": args.monitor_gpu_index,
        "embedding_prerequisites": (
            {} if embedding_prerequisites is None else embedding_prerequisites
        ),
        "gpu_indices": [] if args.gpu_indices is None else list(args.gpu_indices),
        "gpu_assignment_mode": args.gpu_assignment_mode,
        "gpu_memory_budget_bytes": gpu_memory_budget_bytes,
        "gpu_assignment_count": total_gpu_assignment_count,
        "gpu_over_memory_budget_count": total_gpu_over_memory_budget_count,
        "gpu_over_memory_budget_wave_count": gpu_over_memory_budget_wave_count,
        "peak_gpu_estimated_peak_bytes": peak_gpu_estimated_peak_bytes,
        "note": (
            "Command arrays are no-training preflight templates for later workstation runs. "
            "They assume the referenced datasets, models, embeddings, and normal Protify runtime "
            "configuration are available in the target environment."
        ),
        "groups": group_manifests,
        "waves": wave_manifests,
    }


def parallel_cli_args(args: argparse.Namespace) -> list[str]:
    cli_args = [
        "--num_runs",
        str(args.num_runs),
        "--parallel_probe_runs",
        "--parallel_probe_batch_mode",
        args.parallel_batch_mode,
        "--parallel_probe_index_strategy",
        args.parallel_index_strategy,
    ]
    if args.parallel_max_group_size is not None:
        cli_args.extend(["--parallel_probe_max_group_size", str(args.parallel_max_group_size)])
    if args.training_state_budget_gb is not None:
        cli_args.extend([
            "--parallel_probe_training_state_budget_gb",
            str(args.training_state_budget_gb),
        ])
    if args.estimated_peak_budget_gb is not None:
        cli_args.extend([
            "--parallel_probe_estimated_peak_budget_gb",
            str(args.estimated_peak_budget_gb),
        ])
    if args.parallel_max_grad_norm > 0.0:
        cli_args.extend(["--parallel_probe_max_grad_norm", str(args.parallel_max_grad_norm)])
    if args.parallel_grad_clip_mode != "global":
        cli_args.extend(["--parallel_probe_grad_clip_mode", args.parallel_grad_clip_mode])
    return cli_args


def compare_results_template_args(
    args: argparse.Namespace,
    runner_report_paths: Sequence[str],
    output_path: str,
) -> list[str]:
    cli_args = [
        "--sequential_results",
        "<sequential_results.tsv>",
        "--parallel_results",
        "<parallel_results.tsv>",
        "--output_path",
        output_path,
        "--launch_manifest",
        "<preflight.json>",
        "--runner_reports",
    ]
    cli_args.extend(runner_report_paths)
    cli_args.extend(
        [
            "--sequential_telemetry_summaries",
            f"{args.telemetry_dir}/*_sequential.summary.json",
            "--parallel_telemetry_summaries",
            f"{args.telemetry_dir}/*_parallel.summary.json",
            "--require_manifest_result_coverage",
            "--require_manifest_probe_result_coverage",
            "--require_complete_telemetry",
            "--require_successful_runner_reports",
        ]
    )
    return cli_args


def validation_comparison_report(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
    run_specs: Sequence[ParallelProbeRunSpec],
) -> dict[str, object]:
    speedup_formulas = {
        "sequential_total_seconds": "sequential.training_time_seconds",
        "parallel_total_seconds": "parallel.training_time_seconds",
        "sequential_seconds_per_run": "sequential.training_time_seconds / num_runs",
        "parallel_seconds_per_run": "parallel.parallel_probe_seconds_per_run",
        "wall_clock_speedup": "sequential_total_seconds / parallel_total_seconds",
        "per_run_speedup": "sequential_seconds_per_run / parallel_seconds_per_run",
        "trainer_invocation_speedup_ceiling": "sequential_trainer_invocations / parallel_trainer_invocations",
    }
    return {
        "sequential_cli_args": ["--num_runs", str(args.num_runs)],
        "parallel_cli_args": parallel_cli_args(args),
        "compare_conservative_args": compare_results_template_args(
            args,
            [f"{args.telemetry_dir}/manifest_runner_execute.report.json"],
            f"{args.telemetry_dir}/parallel_probe_compare_conservative.report.json",
        ),
        "compare_coscheduled_args": compare_results_template_args(
            args,
            [
                f"{args.telemetry_dir}/manifest_runner_sequential_execute.report.json",
                f"{args.telemetry_dir}/manifest_runner_parallel_execute.report.json",
            ],
            f"{args.telemetry_dir}/parallel_probe_compare_coscheduled.report.json",
        ),
        "sequential_trainer_invocations": plan.total_runs,
        "parallel_trainer_invocations": plan.trainer_invocations,
        "trainer_invocation_reduction": plan.invocation_reduction,
        "trainer_invocation_speedup_ceiling": plan.compression_ratio,
        "speedup_formulas": speedup_formulas,
        "hardware_metric_keys": [
            "gpu_utilization_percent",
            "sm_occupancy_percent",
            "memory_bandwidth_percent",
            "peak_gpu_memory_bytes",
            "cpu_utilization_percent",
            "dataloader_wait_seconds",
        ],
        "group_size_sweep": group_size_sweep_report(args, run_specs),
        "runtime_metric_keys": [
            "training_time_seconds",
            "parallel_probe_seconds_per_run",
            "parallel_probe_trainer_invocations",
            "parallel_probe_invocation_reduction",
            "parallel_probe_compression_ratio",
            "parallel_probe_group_runtime_records",
            "parallel_probe_run_records",
            "parallel_probe_valid_run_metrics",
            "parallel_probe_test_run_metrics",
            "parallel_probe_max_grad_norm",
            "parallel_probe_grad_clip_mode",
        ],
    }


def build_plan_report(args: argparse.Namespace) -> dict[str, object]:
    run_specs = build_universe_run_specs(args)
    probe_configs = probe_plan_configs(args)
    max_group_size_by_key = effective_max_group_size_by_key(args, run_specs)
    plan = plan_parallel_probe_runs(
        run_specs,
        max_parallel_group_size_by_key=max_group_size_by_key,
    )
    estimate = _estimate_plan_for_args(args, plan)
    estimate_summary = estimate.summary_dict()
    group_reports = [
        group_report(group, group_estimate, max_group_size_by_key=max_group_size_by_key)
        for group, group_estimate in zip(plan.groups, estimate.group_estimates)
    ]
    schedule = execution_wave_schedule(args, estimate)
    embedding_prerequisites = embedding_prerequisite_report(args, plan)
    launch_manifest = launch_manifest_report(
        args,
        plan,
        estimate,
        schedule,
        embedding_prerequisites=embedding_prerequisites,
    )
    return {
        "models": list(args.model_names),
        "datasets": list(args.data_names),
        "probe_config_count": len(probe_configs),
        "probe_configs": [probe_config.to_report() for probe_config in probe_configs],
        "probe_config_recommendations": [
            probe_config_recommendation(args, probe_config)
            for probe_config in probe_configs
        ],
        "num_runs_per_model_dataset": args.num_runs * len(probe_configs),
        "num_runs_per_model_dataset_probe": args.num_runs,
        "parallel_max_group_size": args.parallel_max_group_size,
        "parallel_max_grad_norm": args.parallel_max_grad_norm,
        "parallel_grad_clip_mode": args.parallel_grad_clip_mode,
        "training_state_budget_gb": args.training_state_budget_gb,
        "estimated_peak_budget_gb": args.estimated_peak_budget_gb,
        "training_flop_multiplier": args.training_flop_multiplier,
        "wave_memory_budget_gb": args.wave_memory_budget_gb,
        "wave_max_groups": args.wave_max_groups,
        "wave_target_training_flops_per_batch": args.wave_target_training_flops_per_batch,
        "gpu_peak_tflops": args.gpu_peak_tflops,
        "gpu_memory_bandwidth_gbps": args.gpu_memory_bandwidth_gbps,
        "gpu_indices": [] if args.gpu_indices is None else list(args.gpu_indices),
        "gpu_assignment_mode": args.gpu_assignment_mode,
        "telemetry_dir": args.telemetry_dir,
        "monitor_interval_seconds": args.monitor_interval_seconds,
        "monitor_gpu_index": args.monitor_gpu_index,
        "effective_group_size_caps": group_size_cap_report(max_group_size_by_key),
        "parallel_batch_mode": args.parallel_batch_mode,
        "parallel_index_strategy": args.parallel_index_strategy,
        "probe_batch_size": args.probe_batch_size,
        "train_dataset_size": args.train_dataset_size,
        "total_runs": plan.total_runs,
        "trainer_invocations": plan.trainer_invocations,
        "invocation_reduction": plan.invocation_reduction,
        "compression_ratio": plan.compression_ratio,
        "vectorized_runs": plan.vectorized_runs,
        "sequential_runs": plan.sequential_runs,
        "estimate": estimate_summary,
        "groups": group_reports,
        "execution_waves": execution_wave_report(schedule),
        "hardware_roofline": hardware_roofline_report(args, schedule),
        "embedding_prerequisites": embedding_prerequisites,
        "launch_manifest": launch_manifest,
        "execution_recommendation": execution_recommendation_report(args, run_specs),
        "validation_readiness": validation_readiness_report(args, plan, estimate, schedule),
        "validation_comparison": validation_comparison_report(args, plan, run_specs),
    }


def main() -> None:
    args = parse_args()
    report = build_plan_report(args)
    print(json.dumps(report, indent=args.json_indent))


if __name__ == "__main__":
    main()
