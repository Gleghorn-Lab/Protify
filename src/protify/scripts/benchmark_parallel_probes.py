import argparse
import json
import time
import torch
from collections.abc import Sequence
from torch.utils.data import DataLoader, TensorDataset

try:
    from probes.mlp_probe import MLPProbe, MLPProbeConfig
    from probes.parallel_linear_probe import ParallelLinearProbe, ParallelLinearProbeConfig
    from probes.parallel_probe_batches import ParallelRunDataset
    from probes.parallel_probe_plan import (
        ParallelProbeExecutionPlan,
        build_seed_run_specs,
        estimate_parallel_probe_plan,
        plan_parallel_probe_runs,
    )
except ImportError:
    from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
    from protify.probes.parallel_linear_probe import ParallelLinearProbe, ParallelLinearProbeConfig
    from protify.probes.parallel_probe_batches import ParallelRunDataset
    from protify.probes.parallel_probe_plan import (
        ParallelProbeExecutionPlan,
        build_seed_run_specs,
        estimate_parallel_probe_plan,
        plan_parallel_probe_runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sequential vs vectorized linear probe seeds.")
    parser.add_argument("--num_samples", type=int, default=8192)
    parser.add_argument("--input_size", type=int, default=320)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help=(
            "Synthetic label count used only to generate random benchmark labels. "
            "Real python -m main probe runs infer num_labels from the dataset."
        ),
    )
    parser.add_argument("--num_runs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task_type", choices=["singlelabel", "regression"], default="singlelabel")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data_on_device", action="store_true")
    parser.add_argument("--parallel_batch_mode", choices=["shared", "run_specific"], default="shared")
    parser.add_argument("--parallel_index_strategy", choices=["permutation", "affine"], default="permutation")
    parser.add_argument("--parallel_max_group_size", type=int, default=None)
    parser.add_argument("--training_flop_multiplier", type=int, default=3)
    parser.add_argument("--plan_only", action="store_true")
    return validate_args(parser.parse_args())


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    assert args.num_samples > 0, "num_samples must be positive."
    assert args.input_size > 0, "input_size must be positive."
    assert args.hidden_size > 0, "hidden_size must be positive."
    assert args.num_labels > 0, "num_labels must be positive."
    assert args.num_runs > 1, "Benchmark is intended for num_runs > 1."
    assert args.batch_size > 0, "batch_size must be positive."
    assert args.epochs > 0, "epochs must be positive."
    assert args.lr > 0.0, "lr must be positive."
    assert args.weight_decay >= 0.0, "weight_decay must be non-negative."
    assert 0.0 <= args.dropout < 1.0, "dropout must be in [0, 1)."
    assert args.training_flop_multiplier > 0, "training_flop_multiplier must be positive."
    if args.parallel_max_group_size is not None:
        assert args.parallel_max_group_size > 0, "parallel_max_group_size must be positive when provided."
    if args.task_type == "singlelabel":
        assert args.num_labels > 1, "singlelabel benchmarks require num_labels > 1."
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_dataset(args: argparse.Namespace, device: torch.device) -> TensorDataset:
    # n is sample count and d is pooled embedding width.
    data_device = device if args.data_on_device else torch.device("cpu")
    generator = torch.Generator(device=data_device)
    generator.manual_seed(args.seed)
    embeddings = torch.randn(  # (n, d)
        args.num_samples,
        args.input_size,
        generator=generator,
        device=data_device,
    )
    if args.task_type == "singlelabel":
        labels = torch.randint(  # (n,)
            args.num_labels,
            (args.num_samples,),
            generator=generator,
            device=data_device,
        )
    else:
        labels = torch.randn(args.num_samples, 1, generator=generator, device=data_device)  # (n, 1)
    return TensorDataset(embeddings, labels)


def make_loader(dataset: TensorDataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)


def run_seeds(args: argparse.Namespace) -> list[int]:
    return [args.seed + run_idx for run_idx in range(args.num_runs)]


def parallel_probe_trainer_key(args: argparse.Namespace) -> str:
    return (
        f"epochs={args.epochs}|"
        f"batch={args.batch_size}|"
        f"lr={args.lr}|"
        f"weight_decay={args.weight_decay}|"
        f"batch_mode={args.parallel_batch_mode}|"
        f"index_strategy={args.parallel_index_strategy}"
    )


def build_synthetic_plan(args: argparse.Namespace) -> ParallelProbeExecutionPlan:
    run_specs = build_seed_run_specs(
        run_id_prefix="synthetic/synthetic-model",
        base_seed=args.seed,
        num_runs=args.num_runs,
        model_name="synthetic-model",
        data_name="synthetic-data",
        embedding_key=f"synthetic:{args.num_samples}:{args.input_size}:{args.task_type}",
        dataset_key=f"synthetic:{args.num_samples}:{args.num_labels}",
        trainer_key=parallel_probe_trainer_key(args),
        probe_type="mlp",
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_labels=args.num_labels,
        n_layers=args.n_layers,
        task_type=args.task_type,
        pre_ln=True,
        use_bias=True,
        batch_mode=args.parallel_batch_mode,
        index_strategy=args.parallel_index_strategy,
    )
    return plan_parallel_probe_runs(run_specs, max_parallel_group_size=args.parallel_max_group_size)


def plan_summary_dict(
    plan: ParallelProbeExecutionPlan,
    args: argparse.Namespace | None = None,
) -> dict[str, object]:
    if args is None:
        estimate = estimate_parallel_probe_plan(plan)
    else:
        estimate = estimate_parallel_probe_plan(
            plan,
            training_flop_multiplier=args.training_flop_multiplier,
            batch_size=args.batch_size,
            dataset_size=args.num_samples,
            include_run_specific_index=(
                args.parallel_batch_mode == "run_specific"
                and args.parallel_index_strategy == "permutation"
            ),
        )
    estimate_summary = estimate.summary_dict()
    return {
        "total_runs": plan.total_runs,
        "trainer_invocations": plan.trainer_invocations,
        "invocation_reduction": plan.invocation_reduction,
        "compression_ratio": plan.compression_ratio,
        "vectorized_runs": plan.vectorized_runs,
        "sequential_runs": plan.sequential_runs,
        "total_parameter_count": estimate_summary["total_parameter_count"],
        "total_training_state_bytes": estimate_summary["total_training_state_bytes"],
        "peak_group_training_state_bytes": estimate_summary["peak_group_training_state_bytes"],
        "total_batch_activation_bytes": estimate_summary["total_batch_activation_bytes"],
        "peak_group_batch_activation_bytes": estimate_summary["peak_group_batch_activation_bytes"],
        "total_run_specific_index_bytes": estimate_summary["total_run_specific_index_bytes"],
        "peak_group_estimated_peak_bytes": estimate_summary["peak_group_estimated_peak_bytes"],
        "total_forward_flops_per_batch": estimate_summary["total_forward_flops_per_batch"],
        "peak_group_forward_flops_per_batch": estimate_summary["peak_group_forward_flops_per_batch"],
        "total_training_flops_per_batch": estimate_summary["total_training_flops_per_batch"],
        "peak_group_training_flops_per_batch": estimate_summary["peak_group_training_flops_per_batch"],
        "unknown_estimate_group_count": estimate_summary["unknown_group_count"],
    }


def benchmark_config_dict(args: argparse.Namespace) -> dict[str, object]:
    return {
        "num_samples": args.num_samples,
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "num_labels": args.num_labels,
        "num_runs": args.num_runs,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "task_type": args.task_type,
        "requested_device": args.device,
        "data_on_device": args.data_on_device,
        "parallel_batch_mode": args.parallel_batch_mode,
        "parallel_index_strategy": args.parallel_index_strategy,
        "parallel_max_group_size": args.parallel_max_group_size,
        "training_flop_multiplier": args.training_flop_multiplier,
    }


def speedup_formula_dict() -> dict[str, str]:
    return {
        "sequential_total_seconds": "sequential_seconds",
        "parallel_total_seconds": "parallel_seconds",
        "sequential_seconds_per_run": "sequential_seconds / num_runs",
        "parallel_seconds_per_run": "parallel_seconds / num_runs",
        "wall_clock_speedup": "sequential_seconds / parallel_seconds",
        "per_run_speedup": "sequential_seconds_per_run / parallel_seconds_per_run",
        "trainer_invocation_speedup_ceiling": "sequential_trainer_invocations / parallel_trainer_invocations",
    }


def benchmark_comparison_contract(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
) -> dict[str, object]:
    return {
        "sequential_trainer_invocations": plan.total_runs,
        "parallel_trainer_invocations": plan.trainer_invocations,
        "trainer_invocation_reduction": plan.invocation_reduction,
        "trainer_invocation_speedup_ceiling": plan.compression_ratio,
        "speedup_formulas": speedup_formula_dict(),
        "runtime_metric_keys": [
            "sequential_seconds",
            "parallel_seconds",
            "sequential_seconds_per_run",
            "parallel_seconds_per_run",
            "speedup",
            "per_run_speedup",
            "sequential_peak_memory_bytes",
            "parallel_peak_memory_bytes",
            "parallel_index_memory_bytes",
        ],
        "hardware_metric_keys": [
            "gpu_utilization_percent",
            "sm_occupancy_percent",
            "memory_bandwidth_percent",
            "peak_gpu_memory_bytes",
            "cpu_utilization_percent",
            "dataloader_wait_seconds",
        ],
    }


def plan_only_result(
    args: argparse.Namespace,
    plan: ParallelProbeExecutionPlan,
) -> dict[str, object]:
    result = benchmark_config_dict(args)
    result["plan_only"] = True
    result["plan"] = plan_summary_dict(plan, args)
    result["comparison"] = benchmark_comparison_contract(args, plan)
    return result


def parallel_seed_groups(args: argparse.Namespace) -> tuple[tuple[int, ...], ...]:
    seeds = run_seeds(args)
    if args.parallel_max_group_size is None:
        return (tuple(seeds),)
    groups = []
    for start in range(0, len(seeds), args.parallel_max_group_size):
        groups.append(tuple(seeds[start:start + args.parallel_max_group_size]))
    return tuple(groups)


def make_parallel_loader(
    args: argparse.Namespace,
    dataset: TensorDataset,
    seeds: Sequence[int],
) -> tuple[DataLoader, int]:
    # Shared batches contain embeddings (b, d). Run-specific batches contain
    # embeddings (b, r, d), where r is the number of seeds in this group.
    if args.parallel_batch_mode == "shared":
        return make_loader(dataset, args.batch_size, args.seed), 0
    parallel_dataset = ParallelRunDataset(
        dataset,
        run_seeds=seeds,
        independent_shuffles=True,
        index_strategy=args.parallel_index_strategy,
    )
    loader = DataLoader(parallel_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return loader, parallel_dataset.index_memory_bytes


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def move_batch(
    batch: Sequence[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Embeddings are (b, d) or (b, r, d). Labels are (b,), (b, 1),
    # (b, r), or (b, r, 1), depending on task and batch mode.
    embeddings, labels = batch
    moved_embeddings = embeddings.to(device, non_blocking=True)  # (b, d) or (b, r, d)
    moved_labels = labels.to(device, non_blocking=True)  # (b,), (b, 1), (b, r), or (b, r, 1)
    return moved_embeddings, moved_labels


def train_sequential(
    args: argparse.Namespace,
    dataset: TensorDataset,
    device: torch.device,
) -> float:
    start = time.perf_counter()
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(run_seed)
            config = MLPProbeConfig(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                dropout=args.dropout,
                num_labels=args.num_labels,
                n_layers=args.n_layers,
                task_type=args.task_type,
                pre_ln=True,
                use_bias=True,
            )
            model = MLPProbe(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader = make_loader(dataset, args.batch_size, run_seed)
        model.train()
        for _epoch in range(args.epochs):
            for batch in loader:
                embeddings, labels = move_batch(batch, device)  # (b, d), (b,) or (b, 1)
                optimizer.zero_grad(set_to_none=True)
                loss = model(embeddings=embeddings, labels=labels).loss  # ()
                loss.backward()
                optimizer.step()
    synchronize(device)
    return time.perf_counter() - start


def train_parallel(
    args: argparse.Namespace,
    dataset: TensorDataset,
    device: torch.device,
) -> tuple[float, int]:
    total_seconds = 0.0
    peak_index_memory_bytes = 0
    for seeds in parallel_seed_groups(args):
        config = ParallelLinearProbeConfig(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            num_labels=args.num_labels,
            n_layers=args.n_layers,
            task_type=args.task_type,
            pre_ln=True,
            use_bias=True,
            num_runs=len(seeds),
            run_seeds=list(seeds),
        )
        model = ParallelLinearProbe(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loader, index_memory_bytes = make_parallel_loader(args, dataset, seeds)
        peak_index_memory_bytes = max(peak_index_memory_bytes, index_memory_bytes)
        synchronize(device)
        start = time.perf_counter()
        model.train()
        for _epoch in range(args.epochs):
            for batch in loader:
                # Shared: (b, d) plus (b,) or (b, 1). Run-specific:
                # (b, r, d) plus (b, r) or (b, r, 1).
                embeddings, labels = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(embeddings=embeddings, labels=labels).loss  # ()
                loss.backward()
                optimizer.step()
        synchronize(device)
        total_seconds += time.perf_counter() - start
    return total_seconds, peak_index_memory_bytes


def main() -> None:
    args = parse_args()
    plan = build_synthetic_plan(args)
    if args.plan_only:
        print(json.dumps(plan_only_result(args, plan), indent=2))
        return

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dataset = make_dataset(args, device)
    sequential_seconds = train_sequential(args, dataset, device)
    if device.type == "cuda":
        sequential_peak_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        sequential_peak_memory = 0
    parallel_seconds, parallel_index_memory = train_parallel(args, dataset, device)
    if device.type == "cuda":
        parallel_peak_memory = torch.cuda.max_memory_allocated()
    else:
        parallel_peak_memory = 0

    result = benchmark_config_dict(args)
    result.update({
        "device": str(device),
        "plan_only": False,
        "sequential_seconds": sequential_seconds,
        "parallel_seconds": parallel_seconds,
        "sequential_seconds_per_run": sequential_seconds / args.num_runs,
        "parallel_seconds_per_run": parallel_seconds / args.num_runs,
        "speedup": sequential_seconds / parallel_seconds,
        "per_run_speedup": (sequential_seconds / args.num_runs) / (parallel_seconds / args.num_runs),
        "sequential_peak_memory_bytes": sequential_peak_memory,
        "parallel_peak_memory_bytes": parallel_peak_memory,
        "parallel_index_memory_bytes": parallel_index_memory,
        "plan": plan_summary_dict(plan, args),
        "comparison": benchmark_comparison_contract(args, plan),
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
