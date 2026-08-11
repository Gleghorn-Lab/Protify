import csv
import json
from types import SimpleNamespace

try:
    from src.protify.logger import MetricsLogger
except ImportError:
    try:
        from protify.logger import MetricsLogger
    except ImportError:
        from ..logger import MetricsLogger


def test_parallel_probe_seconds_per_run_survives_metric_time_filter(tmp_path) -> None:
    args = SimpleNamespace(
        log_dir=str(tmp_path / "logs"),
        results_dir=str(tmp_path / "results"),
        replay_path=None,
    )
    logger = MetricsLogger(args)
    logger.start_log_main()

    logger.log_metrics(
        "data",
        "model",
        {
            "parallel_probe_seconds_per_run": 1.25,
            "parallel_probe_group_runtime_records": [
                {
                    "group_number": 1,
                    "train_runtime_seconds": 2.0,
                    "seconds_per_run": 1.0,
                }
            ],
            "loader_seconds_debug": 99.0,
            "training_time_seconds": 5.0,
        },
        split_name="test",
    )

    stored = logger.logger_data_tracking["data"]["model"]
    assert stored["parallel_probe_seconds_per_run"] == 1.25
    assert stored["parallel_probe_group_runtime_records"] == [
        {
            "group_number": 1,
            "train_runtime_seconds": 2.0,
            "seconds_per_run": 1.0,
        }
    ]
    assert stored["training_time_seconds"] == 5.0
    assert "loader_seconds_debug" not in stored


def test_probe_result_identity_fields_are_written_to_results_tsv(tmp_path) -> None:
    args = SimpleNamespace(
        log_dir=str(tmp_path / "logs"),
        results_dir=str(tmp_path / "results"),
        replay_path=None,
        probe_type="mlp",
        hidden_size=64,
        dropout=0.2,
        n_layers=0,
        task_type="singlelabel",
        num_labels=3,
    )
    logger = MetricsLogger(args)
    logger.start_log_main()

    logger.log_metrics(
        "EC",
        "ESM2-35",
        {
            "test_loss_mean": 0.2,
        },
        split_name="test",
    )

    stored = logger.logger_data_tracking["EC"]["ESM2-35"]
    assert stored["probe_type"] == "mlp"
    assert stored["hidden_size"] == 64
    assert stored["dropout"] == 0.2
    assert stored["n_layers"] == 0
    assert stored["task_type"] == "singlelabel"
    assert stored["num_labels"] == 3

    with open(logger.results_file, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    metrics = json.loads(rows[1][1])
    assert metrics["probe_type"] == "mlp"
    assert metrics["hidden_size"] == 64
    assert metrics["dropout"] == 0.2
    assert metrics["n_layers"] == 0
    assert metrics["task_type"] == "singlelabel"
    assert metrics["num_labels"] == 3
