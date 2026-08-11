import pytest

try:
    from src.protify.scripts import plan_parallel_probes as preflight
except ImportError:
    try:
        from protify.scripts import plan_parallel_probes as preflight
    except ImportError:
        from ..scripts import plan_parallel_probes as preflight


def _args(*extra_args):
    argv = [
        "--model_names",
        "ESM2-35",
        "ESM2-8",
        "--data_names",
        "EC",
        "DeepLoc-2",
        "--input_size",
        "320",
        "--hidden_size",
        "64",
        "--num_labels",
        "3",
        "--num_runs",
        "3",
        "--n_layers",
        "0",
    ]
    argv.extend(extra_args)
    return preflight.parse_args(argv)


def test_parallel_probe_preflight_builds_cross_product_plan_report() -> None:
    args = _args("--parallel_max_group_size", "2")

    report = preflight.build_plan_report(args)

    assert report["models"] == ["ESM2-35", "ESM2-8"]
    assert report["datasets"] == ["EC", "DeepLoc-2"]
    assert report["probe_config_count"] == 1
    assert report["probe_config_recommendations"][0]["recommended_group_size"] == 2
    assert report["probe_config_recommendations"][0]["training_state_budget_bytes"] is None
    assert report["num_runs_per_model_dataset"] == 3
    assert report["num_runs_per_model_dataset_probe"] == 3
    assert report["parallel_max_group_size"] == 2
    assert report["parallel_max_grad_norm"] == pytest.approx(0.0)
    assert report["parallel_grad_clip_mode"] == "global"
    assert report["training_state_budget_gb"] is None
    assert report["estimated_peak_budget_gb"] is None
    assert report["training_flop_multiplier"] == 3
    assert report["wave_memory_budget_gb"] is None
    assert report["wave_max_groups"] == 1
    assert report["wave_target_training_flops_per_batch"] == 0
    assert report["gpu_peak_tflops"] is None
    assert report["gpu_memory_bandwidth_gbps"] is None
    assert report["gpu_indices"] == []
    assert report["gpu_assignment_mode"] == "packed"
    assert report["telemetry_dir"] == "telemetry"
    assert report["monitor_interval_seconds"] == pytest.approx(1.0)
    assert report["monitor_gpu_index"] is None
    embedding_prerequisites = report["embedding_prerequisites"]
    assert embedding_prerequisites["required_before_probe_training"] is True
    assert embedding_prerequisites["embedding_job_count"] == 4
    assert embedding_prerequisites["downstream_probe_runs"] == 12
    assert embedding_prerequisites["downstream_probe_trainer_invocations"] == 8
    assert embedding_prerequisites["probe_run_fanout_per_embedding_job"] == pytest.approx(3.0)
    assert embedding_prerequisites["trainer_invocation_fanout_per_embedding_job"] == pytest.approx(2.0)
    assert embedding_prerequisites["probe_training_reuses_cached_embeddings"] is True
    assert embedding_prerequisites["parallel_probe_training_reuses_cached_embeddings"] is True
    assert embedding_prerequisites["co_schedule_embedding_and_probe_training"] is False
    first_embedding_job = embedding_prerequisites["embedding_jobs"][0]
    assert first_embedding_job["command_id"] == "embedding-1"
    assert first_embedding_job["model_name"] == "ESM2-35"
    assert first_embedding_job["data_name"] == "EC"
    assert first_embedding_job["embedding_key"] == "ESM2-35/EC/pooled"
    assert first_embedding_job["embedding_kind"] == "pooled"
    assert first_embedding_job["embedding_save_dir"] == "embeddings"
    assert first_embedding_job["embedding_batch_size"] == 16
    assert first_embedding_job["embedding_num_workers"] == 0
    assert first_embedding_job["embedding_pooling_types"] == ["mean", "var"]
    assert first_embedding_job["embedding_hidden_state_index"] == -1
    assert first_embedding_job["embed_dtype"] is None
    assert first_embedding_job["sql"] is False
    assert first_embedding_job["download_embeddings"] is False
    assert first_embedding_job["downstream_probe_config_count"] == 1
    assert first_embedding_job["downstream_seed_runs"] == 3
    assert first_embedding_job["command_environment"] == {"_PROTIFY_EMBED_PHASE": "1"}
    assert first_embedding_job["command"] == [
        "python",
        "-m",
        "main",
        "--model_names",
        "ESM2-35",
        "--data_names",
        "EC",
        "--save_embeddings",
        "--embedding_save_dir",
        "embeddings",
        "--embedding_batch_size",
        "16",
        "--embedding_num_workers",
        "0",
        "--embedding_hidden_state_index",
        "-1",
        "--embedding_pooling_types",
        "mean",
        "var",
    ]
    assert report["probe_batch_size"] == 64
    assert report["train_dataset_size"] == 0
    assert report["total_runs"] == 12
    assert report["trainer_invocations"] == 8
    assert report["invocation_reduction"] == 4
    assert report["compression_ratio"] == pytest.approx(1.5)
    comparison = report["validation_comparison"]
    assert comparison["sequential_cli_args"] == ["--num_runs", "3"]
    assert comparison["parallel_cli_args"] == [
        "--num_runs",
        "3",
        "--parallel_probe_runs",
        "--parallel_probe_batch_mode",
        "shared",
        "--parallel_probe_index_strategy",
        "permutation",
        "--parallel_probe_max_group_size",
        "2",
    ]
    assert comparison["compare_conservative_args"] == [
        "--sequential_results",
        "<sequential_results.tsv>",
        "--parallel_results",
        "<parallel_results.tsv>",
        "--output_path",
        "telemetry/parallel_probe_compare_conservative.report.json",
        "--launch_manifest",
        "<preflight.json>",
        "--runner_reports",
        "telemetry/manifest_runner_execute.report.json",
        "--sequential_telemetry_summaries",
        "telemetry/*_sequential.summary.json",
        "--parallel_telemetry_summaries",
        "telemetry/*_parallel.summary.json",
        "--require_manifest_result_coverage",
        "--require_manifest_probe_result_coverage",
        "--require_complete_telemetry",
        "--require_successful_runner_reports",
    ]
    assert comparison["compare_coscheduled_args"] == [
        "--sequential_results",
        "<sequential_results.tsv>",
        "--parallel_results",
        "<parallel_results.tsv>",
        "--output_path",
        "telemetry/parallel_probe_compare_coscheduled.report.json",
        "--launch_manifest",
        "<preflight.json>",
        "--runner_reports",
        "telemetry/manifest_runner_sequential_execute.report.json",
        "telemetry/manifest_runner_parallel_execute.report.json",
        "--sequential_telemetry_summaries",
        "telemetry/*_sequential.summary.json",
        "--parallel_telemetry_summaries",
        "telemetry/*_parallel.summary.json",
        "--require_manifest_result_coverage",
        "--require_manifest_probe_result_coverage",
        "--require_complete_telemetry",
        "--require_successful_runner_reports",
    ]
    assert comparison["sequential_trainer_invocations"] == 12
    assert comparison["parallel_trainer_invocations"] == 8
    assert comparison["trainer_invocation_reduction"] == 4
    assert comparison["trainer_invocation_speedup_ceiling"] == pytest.approx(1.5)
    assert comparison["speedup_formulas"] == {
        "sequential_total_seconds": "sequential.training_time_seconds",
        "parallel_total_seconds": "parallel.training_time_seconds",
        "sequential_seconds_per_run": "sequential.training_time_seconds / num_runs",
        "parallel_seconds_per_run": "parallel.parallel_probe_seconds_per_run",
        "wall_clock_speedup": "sequential_total_seconds / parallel_total_seconds",
        "per_run_speedup": "sequential_seconds_per_run / parallel_seconds_per_run",
        "trainer_invocation_speedup_ceiling": "sequential_trainer_invocations / parallel_trainer_invocations",
    }
    assert comparison["hardware_metric_keys"] == [
        "gpu_utilization_percent",
        "sm_occupancy_percent",
        "memory_bandwidth_percent",
        "peak_gpu_memory_bytes",
        "cpu_utilization_percent",
        "dataloader_wait_seconds",
    ]
    assert comparison["runtime_metric_keys"] == [
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
    ]
    assert comparison["group_size_sweep"] == [
        {
            "parallel_probe_max_group_size": 1,
            "trainer_invocations": 12,
            "invocation_reduction": 0,
            "compression_ratio": pytest.approx(1.0),
            "vectorized_runs": 0,
            "sequential_runs": 12,
        },
        {
            "parallel_probe_max_group_size": 2,
            "trainer_invocations": 8,
            "invocation_reduction": 4,
            "compression_ratio": pytest.approx(1.5),
            "vectorized_runs": 8,
            "sequential_runs": 4,
        },
        {
            "parallel_probe_max_group_size": 3,
            "trainer_invocations": 4,
            "invocation_reduction": 8,
            "compression_ratio": pytest.approx(3.0),
            "vectorized_runs": 12,
            "sequential_runs": 0,
        },
    ]
    recommendation = report["execution_recommendation"]
    selected_recommendation = recommendation["selected"]
    assert recommendation["candidate_group_sizes"] == [1, 2]
    assert recommendation["explicit_parallel_max_group_size_respected"] is True
    assert selected_recommendation["status"] == "recommended"
    assert selected_recommendation["parallel_probe_max_group_size"] == 2
    assert selected_recommendation["min_effective_group_size"] == 2
    assert selected_recommendation["max_effective_group_size"] == 2
    assert selected_recommendation["budget_constrained"] is False
    assert selected_recommendation["memory_fit"] is True
    assert selected_recommendation["target_fit"] is True
    assert selected_recommendation["trainer_invocations"] == 8
    assert selected_recommendation["compression_ratio"] == pytest.approx(1.5)
    assert selected_recommendation["parallel_cli_args"] == comparison["parallel_cli_args"]
    assert selected_recommendation["manifest_runner_dry_run_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_dry_run.report.json",
        "--variant",
        "both",
        "--use_monitor",
        "--wave_execution_mode",
        "sequential",
    ]
    assert selected_recommendation["manifest_runner_embeddings_dry_run_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_embeddings_dry_run.report.json",
        "--phase",
        "embeddings",
        "--wave_execution_mode",
        "sequential",
    ]
    assert selected_recommendation["manifest_runner_embeddings_execute_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_embeddings_execute.report.json",
        "--phase",
        "embeddings",
        "--wave_execution_mode",
        "sequential",
        "--execute",
    ]
    assert selected_recommendation["manifest_runner_execute_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_execute.report.json",
        "--variant",
        "both",
        "--use_monitor",
        "--wave_execution_mode",
        "sequential",
        "--execute",
    ]
    assert selected_recommendation["manifest_runner_sequential_execute_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_sequential_execute.report.json",
        "--variant",
        "sequential",
        "--use_monitor",
        "--wave_execution_mode",
        "sequential",
        "--execute",
    ]
    assert selected_recommendation["manifest_runner_parallel_execute_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_parallel_execute.report.json",
        "--variant",
        "parallel",
        "--use_monitor",
        "--wave_execution_mode",
        "sequential",
        "--execute",
    ]
    assert report["vectorized_runs"] == 8
    assert report["sequential_runs"] == 4
    readiness = report["validation_readiness"]
    assert readiness["status"] == "ready_with_cautions"
    assert readiness["vectorized_runs"] == 8
    assert readiness["sequential_runs"] == 4
    assert readiness["vectorized_group_count"] == 4
    assert readiness["eligible_singleton_group_count"] == 4
    assert readiness["ineligible_group_count"] == 0
    assert readiness["compression_ratio"] == pytest.approx(1.5)
    assert readiness["min_vectorized_group_size"] == 2
    assert readiness["max_vectorized_group_size"] == 2
    assert readiness["unknown_estimate_group_count"] == 0
    assert readiness["target_underfilled_wave_count"] == 0
    assert readiness["over_memory_budget_wave_count"] == 0
    assert "eligible_singleton_groups_present" in readiness["warnings"]
    assert "low_invocation_compression" in readiness["warnings"]
    assert "no_vectorized_groups" not in readiness["warnings"]
    assert report["estimate"]["total_parameter_count"] > 0
    assert report["estimate"]["peak_group_training_state_bytes"] > 0
    assert report["estimate"]["peak_group_batch_activation_bytes"] > 0
    assert report["estimate"]["peak_group_estimated_peak_bytes"] > (
        report["estimate"]["peak_group_training_state_bytes"]
    )
    assert report["estimate"]["total_run_specific_index_bytes"] == 0
    assert report["estimate"]["unknown_group_count"] == 0
    assert report["estimate"]["total_forward_flops_per_batch"] > 0
    assert report["estimate"]["peak_group_forward_flops_per_batch"] > 0
    assert report["estimate"]["total_training_flops_per_batch"] == (
        report["estimate"]["total_forward_flops_per_batch"] * 3
    )
    roofline = report["hardware_roofline"]
    assert roofline["available"] is False
    assert roofline["waves"] == []
    assert len(report["groups"]) == 8
    assert report["groups"][0]["run_ids"] == ["EC/ESM2-35/seed-42", "EC/ESM2-35/seed-43"]
    assert report["groups"][1]["run_ids"] == ["EC/ESM2-35/seed-44"]
    assert report["groups"][0]["execution_kind"] == "vectorized"
    assert report["groups"][0]["eligible"] is True
    assert report["groups"][0]["can_vectorize"] is True
    assert report["groups"][0]["model_name"] == "ESM2-35"
    assert report["groups"][0]["data_name"] == "EC"
    assert report["groups"][0]["embedding_key"] == "ESM2-35/EC/pooled"
    assert report["groups"][0]["dataset_key"] == "EC/default"
    assert report["groups"][0]["probe_type"] == "mlp"
    assert report["groups"][0]["input_size"] == 320
    assert report["groups"][0]["hidden_size"] == 64
    assert report["groups"][0]["dropout"] == 0.2
    assert report["groups"][0]["num_labels"] == 3
    assert report["groups"][0]["n_layers"] == 0
    assert report["groups"][0]["task_type"] == "singlelabel"
    assert report["groups"][0]["pre_ln"] is True
    assert report["groups"][0]["use_bias"] is False
    assert report["groups"][0]["batch_mode"] == "shared"
    assert report["groups"][0]["index_strategy"] == "permutation"
    assert report["groups"][0]["save_model"] is False
    assert report["groups"][0]["tokenwise"] is False
    assert report["groups"][0]["matrix_embed"] is False
    assert report["groups"][0]["full_finetuning"] is False
    assert report["groups"][1]["execution_kind"] == "eligible_singleton"
    assert report["groups"][1]["can_vectorize"] is False
    assert report["groups"][0]["applied_group_size_cap"] == 2
    assert report["groups"][1]["applied_group_size_cap"] == 2
    assert report["groups"][0]["batch_size"] == 64
    assert report["groups"][0]["batch_activation_bytes"] > 0
    assert report["groups"][0]["single_probe_forward_flops_per_sample"] > 0
    assert report["groups"][0]["group_forward_flops_per_batch"] > 0
    assert report["groups"][0]["group_training_flops_per_batch"] == (
        report["groups"][0]["group_forward_flops_per_batch"] * 3
    )
    assert report["groups"][0]["estimated_peak_bytes"] > report["groups"][0]["training_state_bytes"]
    wave_plan = report["execution_waves"]
    assert wave_plan["total_waves"] == 8
    assert wave_plan["total_groups"] == 8
    assert wave_plan["total_runs"] == 12
    assert wave_plan["max_wave_peak_bytes"] is None
    assert wave_plan["max_groups_per_wave"] == 1
    assert wave_plan["target_training_flops_per_wave"] == 0
    assert wave_plan["target_satisfied_wave_count"] == 0
    assert wave_plan["target_underfilled_wave_count"] == 0
    assert wave_plan["over_memory_budget_wave_count"] == 0
    assert wave_plan["waves"][0]["group_indices"] == [0]
    assert wave_plan["waves"][0]["group_run_counts"] == [2]
    assert wave_plan["waves"][0]["group_run_ids"] == [["EC/ESM2-35/seed-42", "EC/ESM2-35/seed-43"]]
    manifest = report["launch_manifest"]
    assert manifest["entrypoint"] == "python -m main"
    assert manifest["monitor_entrypoint"] == "python -m scripts.monitor_parallel_probe_hardware"
    assert manifest["telemetry_dir"] == "telemetry"
    assert manifest["monitor_interval_seconds"] == pytest.approx(1.0)
    assert manifest["monitor_gpu_index"] is None
    assert manifest["gpu_indices"] == []
    assert manifest["gpu_assignment_mode"] == "packed"
    assert manifest["gpu_memory_budget_bytes"] is None
    assert manifest["gpu_assignment_count"] == 0
    assert manifest["gpu_over_memory_budget_count"] == 0
    assert manifest["gpu_over_memory_budget_wave_count"] == 0
    assert manifest["peak_gpu_estimated_peak_bytes"] == 0
    assert "no-training preflight" in manifest["note"]
    assert len(manifest["groups"]) == 8
    assert manifest["groups"][0]["command_id"] == "group-1"
    assert manifest["groups"][0]["parallel_supported"] is True
    assert manifest["groups"][0]["assigned_gpu_index"] is None
    assert manifest["groups"][0]["environment"] == {}
    assert manifest["groups"][0]["first_seed"] == 42
    assert manifest["groups"][0]["num_runs"] == 2
    assert manifest["groups"][0]["run_seeds"] == [42, 43]
    assert manifest["groups"][0]["sequential_cli_args"] == [
        "--model_names",
        "ESM2-35",
        "--data_names",
        "EC",
        "--probe_type",
        "mlp",
        "--hidden_size",
        "64",
        "--dropout",
        "0.2",
        "--n_layers",
        "0",
        "--probe_batch_size",
        "64",
        "--seed",
        "42",
        "--num_runs",
        "2",
    ]
    assert "--num_labels" not in manifest["groups"][0]["sequential_cli_args"]
    assert manifest["groups"][0]["sequential_command"] == [
        "python",
        "-m",
        "main",
    ] + manifest["groups"][0]["sequential_cli_args"]
    assert manifest["groups"][0]["parallel_cli_args"] == [
        "--model_names",
        "ESM2-35",
        "--data_names",
        "EC",
        "--probe_type",
        "mlp",
        "--hidden_size",
        "64",
        "--dropout",
        "0.2",
        "--n_layers",
        "0",
        "--probe_batch_size",
        "64",
        "--seed",
        "42",
        "--num_runs",
        "2",
        "--parallel_probe_runs",
        "--parallel_probe_batch_mode",
        "shared",
        "--parallel_probe_index_strategy",
        "permutation",
        "--parallel_probe_max_group_size",
        "2",
    ]
    assert "--num_labels" not in manifest["groups"][0]["parallel_cli_args"]
    assert manifest["groups"][0]["parallel_command"] == [
        "python",
        "-m",
        "main",
    ] + manifest["groups"][0]["parallel_cli_args"]
    assert manifest["groups"][0]["sequential_monitor_command"] == [
        "python",
        "-m",
        "scripts.monitor_parallel_probe_hardware",
        "--output_jsonl",
        "telemetry/group-1_sequential.jsonl",
        "--summary_json",
        "telemetry/group-1_sequential.summary.json",
        "--interval_seconds",
        "1.0",
        "--command",
        "--",
    ] + manifest["groups"][0]["sequential_command"]
    assert manifest["groups"][0]["parallel_monitor_command"] == [
        "python",
        "-m",
        "scripts.monitor_parallel_probe_hardware",
        "--output_jsonl",
        "telemetry/group-1_parallel.jsonl",
        "--summary_json",
        "telemetry/group-1_parallel.summary.json",
        "--interval_seconds",
        "1.0",
        "--command",
        "--",
    ] + manifest["groups"][0]["parallel_command"]
    assert manifest["groups"][1]["parallel_supported"] is False
    assert manifest["groups"][1]["parallel_cli_args"] == []
    assert manifest["groups"][1]["parallel_command"] == []
    assert manifest["groups"][1]["parallel_monitor_command"] == []
    assert manifest["waves"][0]["wave_id"] == "wave-1"
    assert manifest["waves"][0]["group_indices"] == [0]
    assert manifest["waves"][0]["command_ids"] == ["group-1"]
    assert manifest["waves"][0]["gpu_assignments"] == []


def test_parallel_probe_preflight_recommends_uncapped_seed_bank() -> None:
    args = _args()

    report = preflight.build_plan_report(args)

    recommendation = report["execution_recommendation"]
    selected = recommendation["selected"]
    assert recommendation["candidate_group_sizes"] == [1, 2, 3]
    assert recommendation["explicit_parallel_max_group_size_respected"] is False
    assert selected["status"] == "recommended"
    assert selected["parallel_probe_max_group_size"] == 3
    assert selected["trainer_invocations"] == 4
    assert selected["invocation_reduction"] == 8
    assert selected["compression_ratio"] == pytest.approx(3.0)
    assert selected["vectorized_runs"] == 12
    assert selected["sequential_runs"] == 0
    assert selected["parallel_cli_args"] == [
        "--num_runs",
        "3",
        "--parallel_probe_runs",
        "--parallel_probe_batch_mode",
        "shared",
        "--parallel_probe_index_strategy",
        "permutation",
        "--parallel_probe_max_group_size",
        "3",
    ]


def test_parallel_probe_preflight_packs_execution_waves_for_coscheduling() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--wave_max_groups",
        "2",
        "--wave_target_training_flops_per_batch",
        "1",
    )

    report = preflight.build_plan_report(args)

    wave_plan = report["execution_waves"]
    assert report["wave_max_groups"] == 2
    assert report["wave_target_training_flops_per_batch"] == 1
    assert wave_plan["total_waves"] == 4
    assert wave_plan["total_groups"] == 8
    assert wave_plan["total_runs"] == 12
    assert wave_plan["max_groups_per_wave"] == 2
    assert wave_plan["target_satisfied_wave_count"] == 4
    assert wave_plan["target_underfilled_wave_count"] == 0
    assert wave_plan["waves"][0]["trainer_invocations"] == 2
    assert wave_plan["waves"][0]["group_indices"] == [0, 2]
    assert wave_plan["waves"][0]["total_runs"] == 4
    assert report["launch_manifest"]["waves"][0]["command_ids"] == ["group-1", "group-3"]
    assert report["execution_recommendation"]["selected"]["manifest_runner_dry_run_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_dry_run.report.json",
        "--variant",
        "both",
        "--use_monitor",
        "--wave_execution_mode",
        "sequential",
    ]
    assert report["execution_recommendation"]["selected"]["manifest_runner_parallel_execute_args"] == [
        "--manifest_path",
        "<preflight.json>",
        "--output_path",
        "telemetry/manifest_runner_parallel_execute.report.json",
        "--variant",
        "parallel",
        "--use_monitor",
        "--wave_execution_mode",
        "concurrent",
        "--execute",
    ]


def test_parallel_probe_preflight_reports_static_hardware_roofline_estimates() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--wave_max_groups",
        "2",
        "--gpu_peak_tflops",
        "100",
        "--gpu_memory_bandwidth_gbps",
        "1000",
    )

    report = preflight.build_plan_report(args)

    roofline = report["hardware_roofline"]
    wave = report["execution_waves"]["waves"][0]
    roofline_wave = roofline["waves"][0]
    compute_seconds = wave["training_flops_per_batch"] / (100.0 * 1_000_000_000_000.0)
    memory_seconds = wave["concurrent_estimated_peak_bytes"] / (1000.0 * 1_000_000_000.0)
    expected_total_seconds = sum(
        max(
            planned_wave["training_flops_per_batch"] / (100.0 * 1_000_000_000_000.0),
            planned_wave["concurrent_estimated_peak_bytes"] / (1000.0 * 1_000_000_000.0),
        )
        for planned_wave in report["execution_waves"]["waves"]
    )

    assert report["gpu_peak_tflops"] == pytest.approx(100.0)
    assert report["gpu_memory_bandwidth_gbps"] == pytest.approx(1000.0)
    assert roofline["available"] is True
    assert roofline["peak_flops_per_second"] == pytest.approx(100.0 * 1_000_000_000_000.0)
    assert roofline["memory_bandwidth_bytes_per_second"] == pytest.approx(1000.0 * 1_000_000_000.0)
    assert roofline["total_roofline_seconds_per_batch"] == pytest.approx(expected_total_seconds)
    assert roofline["peak_wave_roofline_seconds_per_batch"] > 0.0
    assert roofline_wave["wave_id"] == "wave-1"
    assert roofline_wave["trainer_invocations"] == wave["trainer_invocations"]
    assert roofline_wave["total_runs"] == wave["total_runs"]
    assert roofline_wave["compute_seconds_per_batch_lower_bound"] == pytest.approx(compute_seconds)
    assert roofline_wave["memory_seconds_per_batch_lower_bound"] == pytest.approx(memory_seconds)
    assert roofline_wave["roofline_seconds_per_batch_lower_bound"] == pytest.approx(
        max(compute_seconds, memory_seconds)
    )
    assert roofline_wave["roofline_bottleneck"] in ("compute", "memory", "balanced")
    selected = report["execution_recommendation"]["selected"]
    assert selected["roofline_available"] is True
    assert selected["roofline_total_seconds_per_batch"] is not None
    assert selected["roofline_peak_wave_seconds_per_batch"] is not None


def test_parallel_probe_preflight_assigns_gpu_environment_by_wave() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--wave_max_groups",
        "2",
        "--gpu_indices",
        "0",
        "1",
    )

    report = preflight.build_plan_report(args)
    manifest = report["launch_manifest"]

    assert report["gpu_indices"] == [0, 1]
    assert report["gpu_assignment_mode"] == "packed"
    assert manifest["gpu_indices"] == [0, 1]
    assert manifest["gpu_assignment_mode"] == "packed"
    assert manifest["groups"][0]["assigned_gpu_index"] == 0
    assert manifest["groups"][0]["environment"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert manifest["groups"][2]["assigned_gpu_index"] == 0
    assert manifest["groups"][2]["environment"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert manifest["groups"][4]["assigned_gpu_index"] == 1
    assert manifest["groups"][4]["environment"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert "--gpu_index" in manifest["groups"][0]["parallel_monitor_command"]
    assert "0" in manifest["groups"][0]["parallel_monitor_command"]
    assert "--gpu_index" in manifest["groups"][4]["parallel_monitor_command"]
    assert "1" in manifest["groups"][4]["parallel_monitor_command"]
    wave_gpu_assignment = manifest["waves"][0]["gpu_assignments"][0]
    assert wave_gpu_assignment["gpu_index"] == 0
    assert wave_gpu_assignment["group_indices"] == [0, 2]
    assert wave_gpu_assignment["command_ids"] == ["group-1", "group-3"]
    assert wave_gpu_assignment["group_count"] == 2
    assert wave_gpu_assignment["total_runs"] == 4
    assert wave_gpu_assignment["memory_budget_bytes"] is None
    assert wave_gpu_assignment["over_memory_budget"] is False
    assert wave_gpu_assignment["concurrent_estimated_peak_bytes"] == (
        manifest["groups"][0]["estimated_peak_bytes"]
        + manifest["groups"][2]["estimated_peak_bytes"]
    )
    assert wave_gpu_assignment["max_group_estimated_peak_bytes"] == max(
        manifest["groups"][0]["estimated_peak_bytes"],
        manifest["groups"][2]["estimated_peak_bytes"],
    )
    assert wave_gpu_assignment["training_flops_per_batch"] == (
        manifest["groups"][0]["training_flops_per_batch"]
        + manifest["groups"][2]["training_flops_per_batch"]
    )
    assert manifest["gpu_assignment_count"] == 4
    assert manifest["gpu_over_memory_budget_count"] == 0
    assert manifest["gpu_over_memory_budget_wave_count"] == 0
    assert manifest["peak_gpu_estimated_peak_bytes"] == wave_gpu_assignment["concurrent_estimated_peak_bytes"]


def test_parallel_probe_preflight_round_robin_gpu_assignment_spreads_wave_groups() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--wave_max_groups",
        "2",
        "--gpu_indices",
        "0",
        "1",
        "--gpu_assignment_mode",
        "round_robin",
    )

    report = preflight.build_plan_report(args)
    manifest = report["launch_manifest"]

    assert report["gpu_assignment_mode"] == "round_robin"
    assert manifest["groups"][0]["assigned_gpu_index"] == 0
    assert manifest["groups"][2]["assigned_gpu_index"] == 1
    assert manifest["groups"][0]["environment"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert manifest["groups"][2]["environment"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert manifest["waves"][0]["gpu_assignments"] == [
        {
            "gpu_index": 0,
            "group_indices": [0],
            "command_ids": ["group-1"],
            "group_count": 1,
            "total_runs": 2,
            "concurrent_estimated_peak_bytes": manifest["groups"][0]["estimated_peak_bytes"],
            "max_group_estimated_peak_bytes": manifest["groups"][0]["estimated_peak_bytes"],
            "training_flops_per_batch": manifest["groups"][0]["training_flops_per_batch"],
            "memory_budget_bytes": None,
            "over_memory_budget": False,
        },
        {
            "gpu_index": 1,
            "group_indices": [2],
            "command_ids": ["group-3"],
            "group_count": 1,
            "total_runs": 2,
            "concurrent_estimated_peak_bytes": manifest["groups"][2]["estimated_peak_bytes"],
            "max_group_estimated_peak_bytes": manifest["groups"][2]["estimated_peak_bytes"],
            "training_flops_per_batch": manifest["groups"][2]["training_flops_per_batch"],
            "memory_budget_bytes": None,
            "over_memory_budget": False,
        },
    ]


def test_parallel_probe_preflight_reports_per_gpu_memory_budget_pressure() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--wave_max_groups",
        "2",
        "--wave_memory_budget_gb",
        "0.000001",
        "--gpu_indices",
        "0",
        "1",
    )

    report = preflight.build_plan_report(args)
    manifest = report["launch_manifest"]
    budget_bytes = int(0.000001 * (1024 ** 3))
    over_budget_assignments = [
        gpu_assignment
        for wave in manifest["waves"]
        for gpu_assignment in wave["gpu_assignments"]
        if gpu_assignment["over_memory_budget"]
    ]

    assert manifest["gpu_memory_budget_bytes"] == budget_bytes
    assert manifest["gpu_assignment_count"] > 0
    assert manifest["gpu_over_memory_budget_count"] == len(over_budget_assignments)
    assert manifest["gpu_over_memory_budget_count"] > 0
    assert manifest["gpu_over_memory_budget_wave_count"] > 0
    assert manifest["peak_gpu_estimated_peak_bytes"] > budget_bytes
    assert over_budget_assignments[0]["memory_budget_bytes"] == budget_bytes
    assert over_budget_assignments[0]["concurrent_estimated_peak_bytes"] > budget_bytes


def test_parallel_probe_preflight_launch_manifest_uses_monitor_settings() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--telemetry_dir",
        "run_telemetry",
        "--monitor_interval_seconds",
        "0.25",
        "--monitor_gpu_index",
        "3",
    )

    report = preflight.build_plan_report(args)
    manifest = report["launch_manifest"]
    command = manifest["groups"][0]["parallel_monitor_command"]

    assert report["telemetry_dir"] == "run_telemetry"
    assert report["monitor_interval_seconds"] == pytest.approx(0.25)
    assert report["monitor_gpu_index"] == 3
    assert manifest["telemetry_dir"] == "run_telemetry"
    assert manifest["monitor_interval_seconds"] == pytest.approx(0.25)
    assert manifest["monitor_gpu_index"] == 3
    assert "--gpu_index" in command
    assert "3" in command
    assert "run_telemetry/group-1_parallel.jsonl" in command
    assert "run_telemetry/group-1_parallel.summary.json" in command
    assert "0.25" in command


def test_parallel_probe_preflight_launch_manifest_includes_grad_clip_settings() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--parallel_max_grad_norm",
        "0.75",
        "--parallel_grad_clip_mode",
        "per_run",
    )

    report = preflight.build_plan_report(args)
    manifest = report["launch_manifest"]

    assert report["parallel_max_grad_norm"] == pytest.approx(0.75)
    assert report["parallel_grad_clip_mode"] == "per_run"
    assert "max_grad_norm=0.75" in report["groups"][0]["trainer_key"]
    assert "grad_clip_mode=per_run" in report["groups"][0]["trainer_key"]
    assert "--parallel_probe_max_grad_norm" in report["validation_comparison"]["parallel_cli_args"]
    assert "0.75" in report["validation_comparison"]["parallel_cli_args"]
    assert "--parallel_probe_grad_clip_mode" in report["validation_comparison"]["parallel_cli_args"]
    assert "per_run" in report["validation_comparison"]["parallel_cli_args"]
    assert "--parallel_probe_max_grad_norm" not in manifest["groups"][0]["sequential_cli_args"]
    assert "--parallel_probe_grad_clip_mode" not in manifest["groups"][0]["sequential_cli_args"]
    assert "--parallel_probe_max_grad_norm" in manifest["groups"][0]["parallel_cli_args"]
    assert "0.75" in manifest["groups"][0]["parallel_cli_args"]
    assert "--parallel_probe_grad_clip_mode" in manifest["groups"][0]["parallel_cli_args"]
    assert "per_run" in manifest["groups"][0]["parallel_cli_args"]


def test_parallel_probe_preflight_readiness_reports_underfilled_waves() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--wave_max_groups",
        "2",
        "--wave_target_training_flops_per_batch",
        "999999999999999",
    )

    report = preflight.build_plan_report(args)
    readiness = report["validation_readiness"]

    assert readiness["status"] == "ready_with_cautions"
    assert readiness["target_underfilled_wave_count"] == report["execution_waves"]["total_waves"]
    assert "waves_under_target_training_flops" in readiness["warnings"]


def test_parallel_probe_preflight_launch_manifest_preserves_boolean_probe_flags() -> None:
    args = _args(
        "--parallel_max_group_size",
        "2",
        "--use_bias",
        "--no_pre_ln",
        "--save_model",
    )

    report = preflight.build_plan_report(args)

    group = report["groups"][0]
    manifest_group = report["launch_manifest"]["groups"][0]
    assert group["use_bias"] is True
    assert group["pre_ln"] is False
    assert group["save_model"] is True
    assert "--use_bias" in manifest_group["sequential_cli_args"]
    assert "--pre_ln" in manifest_group["sequential_cli_args"]
    assert "--save_model" in manifest_group["sequential_cli_args"]
    assert "--use_bias" in manifest_group["parallel_cli_args"]
    assert "--pre_ln" in manifest_group["parallel_cli_args"]
    assert "--save_model" in manifest_group["parallel_cli_args"]


def test_parallel_probe_preflight_builds_probe_configuration_universe() -> None:
    args = _args(
        "--probe_hidden_sizes",
        "32",
        "64",
        "--probe_dropouts",
        "0.0",
        "0.2",
        "--probe_n_layers",
        "0",
        "1",
        "--parallel_max_group_size",
        "2",
    )

    report = preflight.build_plan_report(args)

    assert report["probe_config_count"] == 8
    assert report["num_runs_per_model_dataset"] == 24
    assert report["num_runs_per_model_dataset_probe"] == 3
    assert report["total_runs"] == 96
    assert report["trainer_invocations"] == 64
    assert report["invocation_reduction"] == 32
    assert report["vectorized_runs"] == 64
    assert report["sequential_runs"] == 32
    assert report["probe_configs"][0] == {
        "label": "h32_l0_d0p0",
        "hidden_size": 32,
        "dropout": 0.0,
        "n_layers": 0,
    }
    assert report["groups"][0]["run_ids"] == [
        "EC/ESM2-35/h32_l0_d0p0/seed-42",
        "EC/ESM2-35/h32_l0_d0p0/seed-43",
    ]
    assert report["groups"][2]["run_ids"] == [
        "EC/ESM2-35/h32_l1_d0p0/seed-42",
        "EC/ESM2-35/h32_l1_d0p0/seed-43",
    ]


def test_parallel_probe_preflight_reports_budget_limited_group_recommendations() -> None:
    args = _args(
        "--probe_hidden_sizes",
        "32",
        "64",
        "--num_runs",
        "16",
        "--parallel_max_group_size",
        "8",
        "--training_state_budget_gb",
        "0.00001",
    )

    report = preflight.build_plan_report(args)
    first = report["probe_config_recommendations"][0]
    second = report["probe_config_recommendations"][1]

    assert first["requested_group_size"] == 8
    assert second["requested_group_size"] == 8
    assert first["hidden_size"] == 32
    assert second["hidden_size"] == 64
    assert first["n_layers"] == 0
    assert first["training_state_budget_bytes"] == int(0.00001 * (1024 ** 3))
    assert first["estimated_peak_budget_bytes"] is None
    assert first["estimated_peak_budget_limited_group_size"] is None
    assert first["budget_limited_group_size"] >= 1
    assert first["recommended_group_size"] == min(8, first["budget_limited_group_size"])
    assert second["recommended_group_size"] == min(8, second["budget_limited_group_size"])
    assert first["single_probe_parameter_count"] < second["single_probe_parameter_count"]
    assert first["training_state_bytes_per_run"] < second["training_state_bytes_per_run"]
    assert first["batch_activation_bytes_per_run"] < second["batch_activation_bytes_per_run"]
    assert first["forward_flops_per_sample"] < second["forward_flops_per_sample"]
    assert first["training_flops_per_batch_per_run"] < second["training_flops_per_batch_per_run"]
    assert first["estimated_peak_bytes_per_run"] < second["estimated_peak_bytes_per_run"]


def test_parallel_probe_preflight_reports_estimated_peak_budget_recommendations() -> None:
    args = _args(
        "--num_runs",
        "16",
        "--parallel_max_group_size",
        "8",
        "--probe_batch_size",
        "64",
        "--train_dataset_size",
        "100",
        "--estimated_peak_budget_gb",
        "0.0001",
    )

    report = preflight.build_plan_report(args)
    recommendation = report["probe_config_recommendations"][0]

    assert report["estimated_peak_budget_gb"] == pytest.approx(0.0001)
    assert report["validation_comparison"]["parallel_cli_args"] == [
        "--num_runs",
        "16",
        "--parallel_probe_runs",
        "--parallel_probe_batch_mode",
        "shared",
        "--parallel_probe_index_strategy",
        "permutation",
        "--parallel_probe_max_group_size",
        "8",
        "--parallel_probe_estimated_peak_budget_gb",
        "0.0001",
    ]
    assert recommendation["requested_group_size"] == 8
    assert recommendation["training_state_budget_bytes"] is None
    assert recommendation["estimated_peak_budget_bytes"] == int(0.0001 * (1024 ** 3))
    assert recommendation["estimated_peak_budget_limited_group_size"] >= 1
    assert recommendation["recommended_group_size"] == min(
        8,
        recommendation["estimated_peak_budget_limited_group_size"],
    )
    assert max(group["num_runs"] for group in report["groups"]) <= recommendation["recommended_group_size"]
    assert len(report["effective_group_size_caps"]) == 4
    assert all(
        cap_report["max_group_size"] == recommendation["recommended_group_size"]
        for cap_report in report["effective_group_size_caps"]
    )
    expected_invocations_per_model_dataset = 16 // recommendation["recommended_group_size"]
    if 16 % recommendation["recommended_group_size"] != 0:
        expected_invocations_per_model_dataset += 1
    assert report["trainer_invocations"] == 4 * expected_invocations_per_model_dataset
    assert all(
        group["applied_group_size_cap"] == recommendation["recommended_group_size"]
        for group in report["groups"]
    )


def test_parallel_probe_preflight_applies_different_caps_per_probe_shape() -> None:
    args = _args(
        "--model_names",
        "ESM2-35",
        "--data_names",
        "EC",
        "--num_runs",
        "16",
        "--probe_hidden_sizes",
        "32",
        "256",
        "--probe_batch_size",
        "64",
        "--estimated_peak_budget_gb",
        "0.002",
    )

    report = preflight.build_plan_report(args)
    recommendations = report["probe_config_recommendations"]
    small_recommendation = recommendations[0]
    large_recommendation = recommendations[1]
    applied_caps = sorted({group["applied_group_size_cap"] for group in report["groups"]})

    assert small_recommendation["recommended_group_size"] > large_recommendation["recommended_group_size"]
    assert small_recommendation["hidden_size"] == 32
    assert large_recommendation["hidden_size"] == 256
    assert applied_caps == sorted(
        {
            small_recommendation["recommended_group_size"],
            large_recommendation["recommended_group_size"],
        }
    )
    assert len(report["effective_group_size_caps"]) == 2
    assert max(group["num_runs"] for group in report["groups"]) == small_recommendation["recommended_group_size"]
    assert min(group["applied_group_size_cap"] for group in report["groups"]) == (
        large_recommendation["recommended_group_size"]
    )
    sweep = report["validation_comparison"]["group_size_sweep"]
    unconstrained_candidate = sweep[-1]
    assert unconstrained_candidate["parallel_probe_max_group_size"] == 16
    assert unconstrained_candidate["trainer_invocations"] == report["trainer_invocations"]
    assert unconstrained_candidate["max_effective_group_size"] == small_recommendation["recommended_group_size"]
    assert unconstrained_candidate["budget_constrained"] is True


def test_parallel_probe_preflight_estimates_run_specific_permutation_index_memory() -> None:
    args = _args(
        "--parallel_batch_mode",
        "run_specific",
        "--parallel_index_strategy",
        "permutation",
        "--probe_batch_size",
        "7",
        "--train_dataset_size",
        "11",
    )

    report = preflight.build_plan_report(args)
    recommendation = report["probe_config_recommendations"][0]

    assert report["probe_batch_size"] == 7
    assert report["train_dataset_size"] == 11
    assert recommendation["run_specific_index_bytes_per_run"] == 11 * 8
    assert recommendation["estimated_peak_bytes_per_run"] > recommendation["training_state_bytes_per_run"]
    assert report["estimate"]["total_run_specific_index_bytes"] == 12 * 11 * 8
    assert report["estimate"]["peak_group_estimated_peak_bytes"] > (
        report["estimate"]["peak_group_training_state_bytes"]
    )
    assert report["groups"][0]["run_specific_index_bytes"] == 3 * 11 * 8


def test_parallel_probe_preflight_affine_run_specific_has_zero_index_memory() -> None:
    args = _args(
        "--parallel_batch_mode",
        "run_specific",
        "--parallel_index_strategy",
        "affine",
        "--train_dataset_size",
        "11",
    )

    report = preflight.build_plan_report(args)

    assert report["probe_config_recommendations"][0]["run_specific_index_bytes_per_run"] == 0
    assert report["estimate"]["total_run_specific_index_bytes"] == 0
    assert report["groups"][0]["run_specific_index_bytes"] == 0


def test_parallel_probe_preflight_reports_ineligible_probe_universe() -> None:
    args = _args("--probe_type", "transformer")

    report = preflight.build_plan_report(args)

    assert report["total_runs"] == 12
    assert report["trainer_invocations"] == 12
    assert report["invocation_reduction"] == 0
    assert report["vectorized_runs"] == 0
    assert report["sequential_runs"] == 12
    assert report["estimate"]["unknown_group_count"] == 12
    assert report["groups"][0]["fallback_reasons"] == ["probe_type"]
    assert report["groups"][0]["parameter_count_known"] is False
    readiness = report["validation_readiness"]
    assert readiness["status"] == "not_ready"
    assert readiness["vectorized_group_count"] == 0
    assert readiness["eligible_singleton_group_count"] == 0
    assert readiness["ineligible_group_count"] == 12
    assert readiness["min_vectorized_group_size"] == 0
    assert readiness["max_vectorized_group_size"] == 0
    assert readiness["unknown_estimate_group_count"] == 12
    assert "no_vectorized_groups" in readiness["warnings"]
    assert "ineligible_groups_present" in readiness["warnings"]
    assert "unknown_static_estimates" in readiness["warnings"]


def test_parallel_probe_preflight_rejects_invalid_args() -> None:
    with pytest.raises(AssertionError, match="input_size"):
        _args("--input_size", "0")

    with pytest.raises(AssertionError, match="parallel_max_group_size"):
        _args("--parallel_max_group_size", "0")

    with pytest.raises(AssertionError, match="parallel_max_grad_norm"):
        _args("--parallel_max_grad_norm", "-0.1")

    with pytest.raises(SystemExit):
        _args("--parallel_grad_clip_mode", "per_model")

    with pytest.raises(AssertionError, match="training_state_budget_gb"):
        _args("--training_state_budget_gb", "0")

    with pytest.raises(AssertionError, match="estimated_peak_budget_gb"):
        _args("--estimated_peak_budget_gb", "0")

    with pytest.raises(AssertionError, match="probe_batch_size"):
        _args("--probe_batch_size", "-1")

    with pytest.raises(AssertionError, match="train_dataset_size"):
        _args("--train_dataset_size", "-1")

    with pytest.raises(AssertionError, match="embedding_save_dir"):
        _args("--embedding_save_dir", "")

    with pytest.raises(AssertionError, match="embedding_batch_size"):
        _args("--embedding_batch_size", "0")

    with pytest.raises(AssertionError, match="embedding_num_workers"):
        _args("--embedding_num_workers", "-1")

    with pytest.raises(AssertionError, match="embedding_pooling_types"):
        _args("--embedding_pooling_types", "")

    with pytest.raises(AssertionError, match="index_dtype_bytes"):
        _args("--index_dtype_bytes", "0")

    with pytest.raises(AssertionError, match="training_flop_multiplier"):
        _args("--training_flop_multiplier", "0")

    with pytest.raises(AssertionError, match="wave_memory_budget_gb"):
        _args("--wave_memory_budget_gb", "0")

    with pytest.raises(AssertionError, match="wave_max_groups"):
        _args("--wave_max_groups", "0")

    with pytest.raises(AssertionError, match="wave_target_training_flops_per_batch"):
        _args("--wave_target_training_flops_per_batch", "-1")

    with pytest.raises(AssertionError, match="gpu_peak_tflops"):
        _args("--gpu_peak_tflops", "0")

    with pytest.raises(AssertionError, match="gpu_memory_bandwidth_gbps"):
        _args("--gpu_memory_bandwidth_gbps", "0")

    with pytest.raises(AssertionError, match="telemetry_dir"):
        _args("--telemetry_dir", "")

    with pytest.raises(AssertionError, match="monitor_interval_seconds"):
        _args("--monitor_interval_seconds", "0")

    with pytest.raises(AssertionError, match="monitor_gpu_index"):
        _args("--monitor_gpu_index", "-1")

    with pytest.raises(AssertionError, match="gpu_indices"):
        _args("--gpu_indices", "-1")

    with pytest.raises(AssertionError, match="gpu_indices"):
        _args("--gpu_indices", "0", "0")

    with pytest.raises(AssertionError, match="probe hidden"):
        _args("--probe_hidden_sizes", "0")

    with pytest.raises(AssertionError, match="probe dropouts"):
        _args("--probe_dropouts", "1.0")

    with pytest.raises(AssertionError, match="probe n_layers"):
        _args("--probe_n_layers", "-1")

    with pytest.raises(AssertionError, match="singlelabel"):
        preflight.parse_args(
            [
                "--model_names",
                "ESM2-35",
                "--data_names",
                "EC",
                "--input_size",
                "320",
                "--num_labels",
                "1",
            ]
        )


def test_parallel_probe_preflight_run_specs_include_dataset_and_model_universe() -> None:
    args = _args("--base_seed", "100")

    specs = preflight.build_universe_run_specs(args)

    assert len(specs) == 12
    assert specs[0].run_id == "EC/ESM2-35/seed-100"
    assert specs[0].embedding_key == "ESM2-35/EC/pooled"
    assert specs[0].dataset_key == "EC/default"
    assert specs[3].run_id == "DeepLoc-2/ESM2-35/seed-100"
    assert specs[6].run_id == "EC/ESM2-8/seed-100"


def test_parallel_probe_preflight_reports_matrix_embedding_prerequisites() -> None:
    args = _args(
        "--matrix_embed",
        "--embedding_save_dir",
        "cached_embeddings",
        "--embedding_batch_size",
        "8",
        "--embedding_num_workers",
        "2",
        "--embedding_hidden_state_index",
        "24",
        "--embed_dtype",
        "bf16",
        "--sql",
        "--download_embeddings",
    )

    report = preflight.build_plan_report(args)

    assert report["groups"][0]["embedding_key"] == "ESM2-35/EC/matrix"
    prerequisites = report["embedding_prerequisites"]
    assert prerequisites["probe_training_reuses_cached_embeddings"] is True
    assert prerequisites["parallel_probe_training_reuses_cached_embeddings"] is False
    first_job = prerequisites["embedding_jobs"][0]
    assert first_job["command_id"] == "embedding-1"
    assert first_job["embedding_key"] == "ESM2-35/EC/matrix"
    assert first_job["embedding_kind"] == "matrix"
    assert first_job["embedding_save_dir"] == "cached_embeddings"
    assert first_job["embedding_batch_size"] == 8
    assert first_job["embedding_num_workers"] == 2
    assert first_job["embedding_pooling_types"] == []
    assert first_job["embedding_hidden_state_index"] == 24
    assert first_job["embed_dtype"] == "bf16"
    assert first_job["sql"] is True
    assert first_job["download_embeddings"] is True
    assert first_job["command_environment"] == {"_PROTIFY_EMBED_PHASE": "1"}
    assert "--embedding_pooling_types" not in first_job["command"]
    assert "--matrix_embed" in first_job["command"]
    assert "--sql" in first_job["command"]
    assert "--download_embeddings" in first_job["command"]
