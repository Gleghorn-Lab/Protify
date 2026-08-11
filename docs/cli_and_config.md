# Configuration

This page documents how configuration works: CLI argument groups, YAML config (base.yaml), merge behavior, and the distinction between `model_names` and `model_paths`/`model_types`. For a quick reference of where each option is used, see the argument tables below and [Getting started](getting_started.md).

---

## Overview

Configuration is a single namespace built by:

1. **Parsing CLI** with `parse_arguments()` in [main.py](../src/protify/main.py).
2. **Optionally loading a YAML file** when `--yaml_path` is set; the YAML is converted to a namespace and merged with the CLI result. **CLI overrides YAML** for any option that was explicitly set on the command line.
3. **Defaults** for ProteinGym, W&B sweep, and a few other options are filled when missing.

The same namespace (`full_args`) is used to build `DataArguments`, `BaseModelArguments`, `ProbeArguments`, `EmbeddingArguments`, and `TrainerArguments` inside `MainProcess.apply_current_settings()`.

---

## How it works

- **CLI-only:** Omit `--yaml_path`; every value comes from argparse defaults or from flags you pass.
- **YAML + CLI:** Pass `--yaml_path path/to/config.yaml`. The file is `yaml.safe_load`'ed; keys are merged into a namespace. Then CLI parsing runs; any CLI option overrides the YAML value. So you can override a few keys without editing the file (e.g. `--num_epochs 10`).
- **Store-true flags:** Merge logic treats store_true/store_false specially so that omitting a flag in YAML does not overwrite a CLI `--flag` or `--no-flag`.

The schema is defined by the union of [base.yaml](../src/protify/yamls/base.yaml) and all options in `parse_arguments()`. YAML can use type tags (e.g. `!!int`, `!!bool`) for clarity.

---

## CLI argument groups

### ID and API keys

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hf_username` | str | Synthyra | Hugging Face username. |
| `--hf_token` | str | None | Hugging Face token. |
| `--synthyra_api_key` | str | None | Synthyra API key for cloud GPU compute. Get yours at [synthyra.com](https://synthyra.com). |
| `--wandb_api_key` | str | None | Weights and Biases API key. |
| `--modal_token_id` | str | None | Modal token ID. |
| `--modal_token_secret` | str | None | Modal token secret. |
| `--modal_api_key` | str | None | Modal key as `token_id:token_secret`. |
| `--rebuild_modal` | flag | False | Force rebuild and deploy Modal backend before run. |
| `--delete_modal_embeddings` | flag | False | Delete embedding cache on Modal volume before submission. |

### Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hf_home` | str | None | Custom Hugging Face cache directory. |
| `--yaml_path` | str | None | Path to YAML config file. |
| `--log_dir` | str | logs | Log directory. |
| `--results_dir` | str | results | Results directory. |
| `--model_save_dir` | str | weights | Directory to save models. |
| `--embedding_save_dir` | str | embeddings | Directory for embeddings. |
| `--download_dir` | str | Synthyra/vector_embeddings | Directory for downloaded embeddings. |
| `--plots_dir` | str | plots | Directory for plots. |
| `--replay_path` | str | None | Path to replay log file. |

### Data

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--delimiter` | str | , | Delimiter for CSV/TSV from data_dirs. |
| `--col_names` | list | [seqs, labels] | Column names (legacy; often inferred). |
| `--max_length` | int | 2048 | Maximum tokenizer length, including special and model boundary tokens. For CARBON, the raw DNA budget is `6 * (max_length - 2)` base pairs. |
| `--padding` | choice | max_length | Padding strategy: `max_length` pads all sequences to `--max_length` (recommended for torch.compile + flex attention); `longest` pads to the longest sequence in each batch. |
| `--trim` | flag | False | If set, drop sequences longer than max_length; else truncate. |
| `--data_names` | list | [] | Dataset names (HuggingFace or preset e.g. standard_benchmark). |
| `--data_dirs` | list | [] | Local split directories with train and valid/test aliases; accepts tabular files or labeled FASTA. |
| `--aa_to_dna`, `--aa_to_rna`, `--dna_to_aa`, `--rna_to_aa`, `--codon_to_aa`, `--aa_to_codon` | flag | False | Sequence translation (only one may be True). |
| `--random_pair_flipping` | flag | False | Randomly swap paired inputs (e.g. PPI) on the training split only. Validation and test keep the dataset order so a score does not depend on the random state. |
| `--multi_column` | list | None | Sequence column names for multi-input tasks. |

### Base model

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_names` | list | None | Preset model names (e.g. ESM2-8). Mutually exclusive with model_paths/model_types. |
| `--model_paths` | list | None | Model paths (HF or local). Must pair with --model_types. Remote CARBON paths must be pinned as `org/repo@<full-commit-sha>`. |
| `--model_types` | list | None | Type keywords for each path (esm2, carbon, custom, etc.). |
| `--model_dtype` | choice | bf16 | fp32, fp16, bf16, float32, float16, bfloat16. |
| `--use_xformers` | flag | False | Use xformers attention for AMPLIFY. |
| `--sae_layer` | int | None | ESMC hidden-state layer the sparse autoencoder reads. Defaults to the depth layer each backbone publishes (23, 27, 60). |
| `--sae_k` | choice | None | Active SAE features per residue: 16, 32, 64, 128, 256, 512. Defaults to 64. |
| `--sae_codebook_dim` | choice | None | SAE codebook width, the embedding dimension per pooling type: 8192, 16384, 32768, 65536, 131072. Defaults to 8192. |

### Probe

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--probe_type` | choice | mlp | mlp, transformer, lyra, xgboost, lightgbm, random_forest. "linear" is the former name for mlp. |
| `--tokenwise` | flag | False | Token-wise prediction. |
| `--probe_n_jobs` | int | -1 | Workers for xgboost, lightgbm, and random_forest probes. |
| `--hidden_size` | int | 8192 | Hidden size for the MLP probe. |
| `--transformer_hidden_size` | int | 512 | Hidden size for transformer probe. |
| `--dropout` | float | 0.2 | Dropout rate. |
| `--n_layers` | int | 1 | Number of layers. |
| `--pre_ln` | flag | True | Pre-LayerNorm (store_false to disable). |
| `--classifier_size` | int | 4096 | Classifier feed-forward dimension. |
| `--transformer_dropout` | float | 0.1 | Transformer layer dropout. |
| `--classifier_dropout` | float | 0.2 | Classifier dropout. |
| `--head_size` | int | 128 | Attention head dimension. `n_heads` is derived as `hidden_size // head_size` and `hidden_size % head_size == 0` is asserted. |
| `--n_heads` | int | None | DEPRECATED. Use `--head_size`. If supplied, emits a `DeprecationWarning` and `head_size` is derived as `hidden_size // n_heads`. Conflicts with an explicit `--head_size` raise an assertion. |
| `--rotary` | flag | True | Use rotary embeddings (store_false to disable). |
| `--attention_backend` | choice | flex | kernels, flex, sdpa. |
| `--output_s_max` | flag | False | Return s_max from attention layers. |
| `--probe_pooling_types` | list | [mean, var] | Pooling types for probe. |
| `--use_bias` | flag | False | Use bias in Linear layers. |
| `--save_model` | flag | False | Save trained model. |
| `--push_raw_probe` | flag | False | With --save_model, push raw probe class to Hub (load with e.g. Class.from_pretrained(repo_id)) instead of packaged AutoModel. |
| `--production_model` | flag | False | Production model flag. |
| `--lora` | flag | False | Use LoRA. |
| `--lora_r` | int | 8 | LoRA rank. |
| `--lora_alpha` | float | 32.0 | LoRA alpha. |
| `--lora_dropout` | float | 0.01 | LoRA dropout. |
| `--sim_type` | choice | dot | dot, euclidean, cosine. |
| `--add_token_ids` | flag | False | Add token type embeddings for PPI. |

### Scikit

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scikit_n_iter` | int | 10 | Iterations for scikit tuning. |
| `--scikit_cv` | int | 3 | Cross-validation folds. |
| `--scikit_random_state` | int | None | Random state (None uses global seed). |
| `--scikit_model_name` | str | None | Scikit model name. |
| `--scikit_model_args` | str | None | JSON hyperparameters (skips tuning). |
| `--use_scikit` | flag | False | Use scikit-learn path. |
| `--n_jobs` | int | 1 | Processes for scikit. |

### Embedding

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embedding_batch_size` | int | 16 | Batch size for embedding. |
| `--embedding_num_workers` | int | 0 | DataLoader workers for embedding. |
| `--num_workers` | int | 0 | DataLoader workers for training. |
| `--download_embeddings` | flag | False | Download precomputed embeddings. |
| `--matrix_embed` | flag | False | Keep per-residue matrices (no pooling). |
| `--embedding_pooling_types` | list | [mean, var] | Pooling for vector embeddings. |
| `--embedding_hidden_state_index` | int | -1 | Hidden-state tuple index to pool from. `-1` uses `last_hidden_state` without materializing all layer states. |
| `--save_embeddings` | flag | False | Save computed embeddings. |
| `--embed_dtype` | choice | None | fp32/fp16/bf16 for embeddings (default: model_dtype). |
| `--sql` | flag | False | Store embeddings in SQLite. |
| `--read_scaler` | int | 100 | Read scaler for SQL. |

### Trainer

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_epochs` | int | 200 | Training epochs. |
| `--probe_batch_size` | int | 64 | Probe batch size. |
| `--base_batch_size` | int | 4 | Base model batch size. |
| `--probe_grad_accum` | int | 1 | Gradient accumulation steps (probe). |
| `--base_grad_accum` | int | 8 | Gradient accumulation steps (base). |
| `--lr` | float | 1e-4 | Learning rate (shared by probe and base phases unless `--base_lr` is set). |
| `--probe_lr` | float | None | Learning rate for the probe phase. If omitted, falls back to `--lr`. |
| `--base_lr` | float | None | Learning rate for the base-model phase. If omitted, falls back to `--lr`. |
| `--lr_scheduler` | str | cosine | Hugging Face `TrainingArguments.lr_scheduler_type`. |
| `--optimizer` | str | adamw_torch | Hugging Face `TrainingArguments.optim`. |
| `--weight_decay` | float | 0.00 | Weight decay. |
| `--patience` | int | 1 | Early-stopping patience (probe phase, and base phase unless `--base_patience` is set). |
| `--base_num_epochs` | int | None | Epoch count for the base-model phase of hybrid / full-finetuning training. If omitted, falls back to `--num_epochs`. |
| `--base_patience` | int | None | Early-stopping patience for the base-model phase. If omitted, falls back to `--patience`. |
| `--seed` | int | None | Random seed (None: time-based). |
| `--deterministic` | flag | False | Deterministic mode. |
| `--full_finetuning` | flag | False | Full model finetuning. |
| `--hybrid_probe` | flag | False | Hybrid probe then finetune. |
| `--num_runs` | int | 1 | Number of seeds; report mean and std. |
| `--parallel_probe_runs` | flag | False | For eligible pooled MLP probes, train all `--num_runs` seeds in one vectorized trainer pass. |
| `--parallel_probe_batch_mode` | choice | shared | With `--parallel_probe_runs`, use `shared` minibatches across runs or `run_specific` deterministic per-run training permutations. |
| `--parallel_probe_index_strategy` | choice | permutation | With `--parallel_probe_batch_mode run_specific`, use materialized per-run `permutation` indices or memory-free deterministic `affine` bijections. |
| `--parallel_probe_max_group_size` | int | None | Optional cap on seeds per vectorized probe bank; larger `--num_runs` values are chunked into multiple parallel Trainer invocations. |
| `--parallel_probe_training_state_budget_gb` | float | None | Optional static trainable-state budget in GiB for deriving a parallel-probe group-size cap. |
| `--parallel_probe_estimated_peak_budget_gb` | float | None | Optional static estimated peak-memory budget in GiB for deriving a parallel-probe group-size cap. |
| `--parallel_probe_max_grad_norm` | float | 0.0 | Parallel-probe gradient clipping max norm; default disables global clipping across seed banks. |
| `--parallel_probe_grad_clip_mode` | choice | global | Gradient clipping mode for parallel probe banks: `none`, `global`, or `per_run`. With the default max norm of `0.0`, all modes are no-ops until a positive norm is set. |
| `--parallel_probe_ensemble_average_mode` | choice | logits | Average mode for reported seed-bank ensemble metrics: `logits` or `probabilities`. |

For a no-training preflight over a planned model/dataset/probe sweep, use `python -m scripts.plan_parallel_probes`. It reports grouping, Trainer invocation reduction, static probe-bank resource estimates, embedding prerequisite commands, launch-manifest command templates, an `execution_recommendation` block, and a sequential-vs-parallel validation comparison block without loading datasets, models, embeddings, or Trainer. The `embedding_prerequisites` block emits one `_PROTIFY_EMBED_PHASE=1` command per model/dataset cache, records cache fanout into downstream probe runs, and keeps embedding work separate from probe launch waves because PLM sizes and cache writes can dominate resource use. Shape those commands with `--embedding_save_dir`, `--embedding_batch_size`, `--embedding_num_workers`, `--embedding_pooling_types`, `--embedding_hidden_state_index`, `--embed_dtype`, `--sql`, and `--download_embeddings`. The launch manifest includes full accepted `python -m main` command arrays plus monitor-wrapped command arrays for future workstation runs; use `--telemetry_dir`, `--monitor_interval_seconds`, and `--monitor_gpu_index` to shape those telemetry templates. The preflight accepts plural probe-shape flags such as `--probe_hidden_sizes`, `--probe_dropouts`, and `--probe_n_layers` to model probe-configuration sweeps; add `--probe_batch_size` and `--train_dataset_size` to include batch-activation, matmul FLOP, and run-specific permutation-index estimates, add `--training_state_budget_gb` or `--estimated_peak_budget_gb` to apply conservative per-probe group-size caps, and add `--parallel_max_grad_norm` plus `--parallel_grad_clip_mode per_run` to template independence-preserving gradient clipping. For future workstation co-scheduling tests across model/dataset/probe groups, add `--wave_max_groups`, `--wave_memory_budget_gb`, and `--wave_target_training_flops_per_batch` to emit an `execution_waves` launch matrix. Add `--gpu_peak_tflops` and `--gpu_memory_bandwidth_gbps` to emit a static `hardware_roofline` block with per-wave compute and memory lower bounds for the target GPU profile. Add `--gpu_indices 0 1 ...` to place planned groups through manifest `CUDA_VISIBLE_DEVICES` environment metadata; `--gpu_assignment_mode packed` keeps groups in the same wave on one GPU, while `round_robin` spreads groups within each wave. Wave entries include per-GPU estimated peak bytes, memory-budget fit, and FLOPs so placement can be reviewed before execution. The recommendation block selects a concrete group-size cap and manifest-runner template from static candidate plans while respecting any explicit `--parallel_max_group_size` as a hard cap; the runner templates include `--output_path` values under `--telemetry_dir` for durable dry-run and execution reports. Co-scheduled plans include separate `manifest_runner_sequential_execute_args` and `manifest_runner_parallel_execute_args`; the parallel template uses concurrent waves when requested, while the sequential template stays sequential for baseline timing. The validation block also includes `compare_conservative_args` and `compare_coscheduled_args` so later result TSVs, runner reports, telemetry summaries, manifest coverage, probe-config coverage, and saved comparison report JSON can be checked without rebuilding the command by hand.

`--num_labels` is intentionally not a `python -m main` argument. Normal Protify runs infer label count from the loaded dataset and store it on `probe_args`. The synthetic benchmark and no-training planner have their own `--num_labels` flag only because they do not load datasets; launch-manifest `python -m main` commands generated by the planner omit it.

Use `python -m scripts.run_parallel_probe_launch_manifest --manifest_path <preflight.json>` to dry-run those launch-manifest commands by wave. It only runs commands when `--execute` is explicitly passed; add `--output_path <runner-report.json>` to save the dry-run or execution report. The runner defaults to `--phase probes`; use `--phase embeddings` to plan or execute the embed-only prerequisite commands, or `--phase all` to include the embedding prerequisite wave before probe waves. Add `--wave_execution_mode concurrent` to launch all parallel commands in each planned wave together during future workstation runs. Concurrent launch with `--variant sequential` or `--variant both` is blocked by default because it can contaminate baseline timing; use `--allow_baseline_concurrency` only for deliberate throughput tests. If selected waves are over the preflight GPU memory budget, execution is blocked unless `--allow_over_budget` is explicitly passed. Add `--skip_completed` to omit commands whose monitor summary JSON already exists when resuming a partially completed manifest.

After workstation runs exist, use `python -m scripts.compare_parallel_probe_runs --sequential_results <seq.tsv> --parallel_results <par.tsv>` to compare wall-clock speedups and metric parity without rerunning training. Add `--output_path <compare-report.json>` to save the JSON report. If both files include seed-ensemble metrics, the report also compares `sequential_probe_ensemble_test_*` against `parallel_probe_ensemble_test_*`. The JSON summary includes `validation_verdict`; tune it with thresholds such as `--min_wall_clock_speedup`, `--min_per_run_speedup`, `--min_manifest_speedup_efficiency`, `--require_manifest_result_coverage`, `--require_manifest_probe_result_coverage`, `--require_ensemble_metrics`, `--require_complete_telemetry`, `--require_successful_runner_reports`, `--min_parallel_gpu_utilization_percent`, and `--min_gpu_utilization_gain_percent`. Add `--launch_manifest <preflight.json>` to report planned model/dataset result coverage, exact probe-config coverage from standard TSV probe identity fields, and observed wall-clock speedup as a fraction of the preflight trainer-invocation speedup ceiling. Add `--runner_reports <manifest_runner_execute.report.json>` to summarize saved manifest-runner dry-run or execution reports; add `--require_successful_runner_reports` to require at least one saved execution report with no over-budget block, unknown selected command ids, missing wave command ids, nonzero command returns, runner failures, or missing executed command results. When `--launch_manifest` is also provided, runner-report validation checks exact `(command_id, variant)` coverage against every planned sequential and parallel probe command. Embedding-prerequisite commands from `--phase embeddings` or `--phase all` runner reports are summarized but ignored for probe manifest coverage. Executed commands and resume skips with `skip_reason='completed_summary_exists'` count as covered; empty-command skips do not. Add sequential plus parallel telemetry summary JSON files to compare utilization summaries from the hardware monitor and report telemetry coverage against planned monitor commands.

Use `python -m scripts.monitor_parallel_probe_hardware --output_jsonl <samples.jsonl> --command -- <protify command>` on the workstation to collect `nvidia-smi` utilization telemetry around those runs.

### Balanced regression metrics (EpHod-style)

Applied only for regression tasks. See [Probes and Training > Balanced regression metrics](probes_and_training.md#balanced-regression-metrics) for details.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--balanced_regression_metrics` / `--no_balanced_regression_metrics` | flag | True | Toggle computation. |
| `--balanced_weight_method` | choice | `bin_inv` | `none`, `bin_inv`, `bin_inv_sqrt`, `LDS_inv`, `LDS_inv_sqrt`, `LDS_extreme`. |
| `--balanced_bin_borders` | list[float] | None | Explicit bin borders (e.g. `5 9` for pH). None uses training-label tertiles. |
| `--balanced_n_resamples` | int | 100 | Weight-bootstrap draws for resampled Pearson/Spearman. |
| `--balanced_lds_bins` | int | 100 | LDS histogram bin count. |
| `--balanced_lds_ks` | int | 5 | LDS Gaussian kernel size. |
| `--balanced_lds_sigma` | float | 2.0 | LDS Gaussian sigma. |

### ProteinGym

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--proteingym` | flag | False | Run ProteinGym zero-shot. |
| `--dms_ids` | list | [all] | DMS assay IDs or "all". |
| `--mode` | choice | benchmark | benchmark, indels, multiples, singles. |
| `--scoring_method` | choice | masked_marginal | masked_marginal, mutant_marginal, wildtype_marginal, pll, global_log_prob. |
| `--scoring_window` | choice | optimal | optimal, sliding. |
| `--pg_batch_size` | int | 32 | Batch size for ProteinGym. |
| `--compare_scoring_methods` | flag | False | Compare scoring methods. |
| `--score_only` | flag | False | Skip scoring; run benchmark on existing CSVs. |

### W&B sweep

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb_hyperopt` | flag | False | Run W&B hyperparameter sweep. |
| `--wandb_project` | str | (from env/args) | W&B project name. |
| `--wandb_entity` | str | (from env/args) | W&B entity. |
| `--sweep_config_path` | str | yamls/sweep.yaml | Sweep YAML path. |
| `--sweep_count` | int | 10 | Number of trials. |
| `--sweep_method` | choice | bayes | bayes, grid, random. |
| `--sweep_metric_cls` | str | eval_loss | Classification metric to optimize. |
| `--sweep_metric_reg` | str | eval_loss | Regression metric to optimize. |
| `--sweep_goal` | choice | minimize | maximize, minimize. |

---

## YAML config (base.yaml)

The file [src/protify/yamls/base.yaml](../src/protify/yamls/base.yaml) is organized by section. Key names match CLI long options (without the leading dashes). Types can be explicit with YAML tags (e.g. `!!int`, `!!bool`). Example structure:

```yaml
# ID
hf_username: Synthyra
hf_token: null
# ...

# Paths
log_dir: logs
results_dir: results
# ...

# DataArguments
delimiter: ','
max_length: 1024
data_names: [DeepLoc-2]
data_dirs: []
# ...

# BaseModelArguments
model_names: [ESM2-8]
# model_paths: [...]
# model_types: [...]

# ProbeArguments
probe_type: mlp
tokenwise: false
# ...

# EmbeddingArguments, TrainerArguments, ScikitArguments, etc.
```

Anything you can set via CLI can be set in YAML; CLI overrides when both are present.

---

## model_names vs model_paths and model_types

- **model_names:** A list of preset names (e.g. `ESM2-8`, `ProtT5`, or `standard` to expand to a standard set). Resolved via [supported models](models_and_embeddings.md) and `get_base_model`/`get_tokenizer`. Use this for built-in HuggingFace models.
- **model_paths + model_types:** For custom or local models. `model_paths` is a list of paths (HuggingFace IDs or local dirs); `model_types` must be the same length and each element is a dispatch keyword (e.g. `esm2`, `carbon`, `custom`). Remote CARBON paths must use `org/repo@<full-commit-sha>` so model weights are immutable. The audited tokenizer implementation and Qwen vocabulary revision are pinned in Protify. You cannot mix: either set `model_names` or set both `model_paths` and `model_types`.

---

## Examples

### Probe-only with two models and one dataset

```bash
py -m src.protify.main --model_names ESM2-8 ESM2-35 --data_names DeepLoc-2 --num_epochs 5 --results_dir my_results
```

### ProteinGym zero-shot on all substitution assays

```bash
py -m src.protify.main --proteingym --model_names ESM2-150 --dms_ids all --mode benchmark
```

### W&B sweep (after setting W&B credentials)

```bash
py -m src.protify.main --yaml_path src/protify/yamls/base.yaml --use_wandb_hyperopt --sweep_count 5 --wandb_project my_project --wandb_entity my_entity
```

### YAML with CLI overrides

```bash
py -m src.protify.main --yaml_path my_config.yaml --num_epochs 20 --lr 5e-5 --save_embeddings
```

### Docker

Run from the repository root on your host with the workspace mounted and working directory set to `/workspace/src/protify`. Linux/mac example:

```bash
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --model_names ESM2-8 ESM2-35 --data_names DeepLoc-2 --num_epochs 5 --results_dir my_results
```

ProteinGym zero-shot:

```bash
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --proteingym --model_names ESM2-150 --dms_ids all --mode benchmark
```

YAML with overrides:

```bash
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --yaml_path yamls/base.yaml --num_epochs 20 --lr 5e-5 --save_embeddings
```

On Windows use `-v "%CD%":/workspace` and `py -m main` instead of `python -m main`. Ensure the image is built first: `docker build -t protify-env:latest .`

---

## See also

- [Getting started](getting_started.md) for first runs
- [Data](data.md) for data options and supported datasets
- [Models and embeddings](models_and_embeddings.md) for base model and embedding options
- [Probes and training](probes_and_training.md) for probe and trainer options
- [Hyperparameter optimization](hyperparameter_optimization.md) for W&B sweep details
