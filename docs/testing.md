# Testing

This page describes the Protify testing suite: where tests live, what they cover, and how to run them.

---

## Overview

Tests are under [src/protify/testing_suite/](../src/protify/testing_suite/). They cover metrics computation, pooling strategies, probe construction and forward passes, model components (MLP, attention, rotary embeddings), loss functions, blob serialization, data utilities, seed reproducibility, lazy-predict integration, packaged probe export, and probe attention behavior. Run them with pytest from the repository root.

---

## Test layout

| File | Description |
|------|-------------|
| **conftest.py** | Shared fixtures (tiny embeddings, masks, label arrays) and marker registration (`gpu`, `slow`). |
| **test_metrics.py** | Metric computation: softmax, scorers, threshold optimization, single/multi-label classification, regression, tokenwise, ROC/PR AUC, `get_compute_metrics` dispatch. |
| **test_pooler.py** | All pooling strategies (mean, max, norm, median, std, var, cls, eos, parti/PageRank), mask layouts, explicit boundary validation, and concatenation behavior. |
| **test_carbon.py** | CARBON and transitive Qwen source pinning, audited tokenizer parity, partial 6-mers, exact and 8-aligned padding, cache identity, checkpoint-vocabulary agreement, raw-sequence budgets across mixed model selections, LoRA target scoping, translation-aware length enforcement, lowercase DNA, GUI exposure, hidden-state selection, and embedding/training dispatch. |
| **test_probe_construction.py** | `get_probe` factory dispatch for all probe_type x tokenwise combinations, forward passes (singlelabel, regression, multilabel, sigmoid_regression), loss computation, config roundtrip. |
| **test_model_components.py** | `intermediate_correction_fn`, SwiGLU, `swiglu_ln_ffn`, RotaryEmbedding, MultiHeadAttention shapes and values. |
| **test_losses.py** | `get_loss_fct` dispatch, SoftBCELoss, SoftBCEWithLogitsLoss (forward, ignore_index, smooth_factor). |
| **test_blob_serialization.py** | `tensor_to_embedding_blob` / `embedding_blob_to_tensor` roundtrip for float32/float16/bfloat16, `batch_tensor_to_blobs`, legacy fallback. |
| **test_data_utils.py** | Amino acid, DNA, RNA, codon constant validation, translation mapping coverage, `pad_and_concatenate_dimer`. |
| **test_seed_utils.py** | `set_global_seed` / `get_global_seed`, reproducibility (torch, numpy, random), `seed_worker`, `dataloader_generator`. |
| **test_lazy_predict.py** | LazyClassifier and LazyRegressor fit, models, predictions, provide_models. |
| **test_packaged_probe_export.py** | Linear and transformer probe save/load roundtrip via PackagedProbeModel, PPI inference with/without token_type_ids. |
| **test_parallel_linear_probe.py** | Vectorized multi-seed MLP probe behavior, exported single-run parity, shared and run-specific batches, per-run metrics, ensemble metrics, gradient clipping, chunking, and budget-derived group sizing. |
| **test_parallel_probe_batches.py** | `ParallelRunDataset` deterministic per-run indexing, affine index strategy, tensor-cache fast paths, and collator shapes. |
| **test_parallel_probe_plan.py** | Static grouping and resource estimates for model/dataset/probe/seed universes. |
| **test_parallel_probe_preflight.py** | No-training launch-manifest generation, embedding prerequisites, execution waves, GPU placement metadata, validation templates, and helper-only `--num_labels` behavior. |
| **test_parallel_probe_compare.py** | Sequential-vs-parallel result comparison, metric parity, manifest coverage, telemetry coverage, runner-report gates, and validation verdicts. |
| **test_parallel_probe_benchmark.py** | Synthetic benchmark argument validation, plan summaries, run-specific loaders, and index-memory estimates. |
| **test_parallel_probe_hardware_monitor.py** | `nvidia-smi` monitor command construction, JSONL summary parsing, and dry-run wrapper behavior. |
| **test_parallel_probe_launch_manifest_runner.py** | Launch-manifest dry runs, execution planning, resume skips, wave concurrency, and over-budget execution gates. |
| **test_parallel_probe_logger.py** | Result TSV preservation for parallel-probe timing and identity fields. |
| **test_probe_attention.py** | Transformer attention with 2D/4D masks, s_max output, `resolve_attention_backend`. |
| **embedding_test.py** | Standalone embedding diagnostic script (excluded from pytest collection). |

---

## Markers

Tests use pytest markers registered in `conftest.py`:

- `@pytest.mark.gpu` -- requires a CUDA GPU. Skipped by default in CPU-only environments.
- `@pytest.mark.slow` -- tests that take >10 seconds (e.g. loading multiple models).

Filter tests by marker:

```bash
# CPU-only tests (skip gpu and slow)
py -m pytest src/protify/testing_suite -v -m "not gpu and not slow"

# GPU tests only
py -m pytest src/protify/testing_suite -v -m "gpu"

# All tests
py -m pytest src/protify/testing_suite -v
```

---

## How to run

From the **repository root** (so that `src` is on the path):

```bash
py -m pytest src/protify/testing_suite -v
```

To run a single file:

```bash
py -m pytest src/protify/testing_suite/test_metrics.py -v
```

To run with coverage (if you have pytest-cov):

```bash
py -m pytest src/protify/testing_suite -v --cov=src.protify --cov-report=term-missing
```

On Windows use `py`; on Linux/mac you can use `python` if preferred. Ensure the project dependencies are installed (`pip install -r requirements.txt`).

---

## Docker

Build the image first, then run pytest with the workspace mounted:

**Linux / macOS:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
```

**Windows:**
```bash
docker build -t protify-env:latest .
docker run --rm -v "%CD%":/workspace -w /workspace protify-env:latest py -m pytest src/protify/testing_suite -v
```

For GPU-dependent tests, add `--gpus all` to the run command:

```bash
docker run --rm --gpus all -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v
```

To run only CPU tests in Docker:

```bash
docker run --rm -v "${PWD}":/workspace -w /workspace protify-env:latest python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

### Parallel-probe focused suite

For reviews that touch vectorized seed-bank training, run the focused suite from `src/protify` inside Docker:

```bash
docker run --rm --gpus all -v "${PWD}":/workspace -e PYTHONPATH=/workspace -w /workspace/src/protify protify-env:latest python -m pytest testing_suite/test_parallel_probe_benchmark.py testing_suite/test_parallel_probe_batches.py testing_suite/test_parallel_probe_compare.py testing_suite/test_parallel_probe_hardware_monitor.py testing_suite/test_parallel_probe_launch_manifest_runner.py testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py testing_suite/test_parallel_probe_logger.py testing_suite/test_parallel_probe_preflight.py testing_suite/test_trainer_arguments.py -v
```

The current GH200 workstation validation for this focused suite passed with `180 passed`.

---

## See also

- [Getting started](getting_started.md) for installation and entry points
- [Probes and training](probes_and_training.md) for packaged probe export
- [Models and embeddings](models_and_embeddings.md) for embedder behavior
