# Getting Started

This guide covers installation, entry points, and your first CLI and YAML runs. For full argument reference, see [Configuration](cli_and_config.md).

---

## Overview

Protify can be run from the command line (with or without a YAML config file) or from the Tk GUI. All paths use the same `MainProcess` pipeline: load config, load data, compute or load embeddings, train probes (or run full finetuning, hybrid, or scikit), then write results and generate plots.

---

## Installation

### From pip

```bash
pip install Protify
```

### Local development (clone and install)

From the repository root:

```bash
git clone https://github.com/Gleghorn-Lab/Protify.git
cd Protify
git submodule update --init --recursive
py -m pip install -r requirements.txt
```

Then run from the repo root so that `src` is on the Python path:

```bash
py -m src.protify.main --help
```

Alternatively, from `src/protify` you can run `py -m main` and `py -m gui` (see [Entry points](#entry-points)).

### Docker

Run all commands from the **repository root** on your host; you do not need to `cd src/protify`. Mount the project at `/workspace` and set the container working directory to `/workspace/src/protify` so that `py -m main` and `py -m gui` work without changing the module path.

**Linux / macOS:**
```bash
docker build -t protify-env:latest .
docker run --rm -it --gpus all -v "${PWD}":/workspace -w /workspace/src/protify protify-env:latest python -m main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 2
```

**Windows:**
```bash
docker build -t protify-env:latest .
docker run --rm -it --gpus all -v "%CD%":/workspace -w /workspace/src/protify protify-env:latest py -m main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 2
```

Paths like `--log_dir` and `--results_dir` are relative to `/workspace/src/protify`. To write outputs at project root, use e.g. `--log_dir /workspace/logs` and `--results_dir /workspace/results`.

### Optional: xformers for AMPLIFY

If you use the AMPLIFY model with `--use_xformers`:

```bash
pip install xformers
```

---

## Entry points

| Entry point | Purpose |
|------------|--------|
| `py -m src.protify.main` | CLI (and optional YAML). Run from repo root. |
| `py -m src.protify.gui` | Tk GUI. Run from repo root. |
| `py -m main` | Same as main, when run from `src/protify`. |
| `py -m gui` | Same as gui, when run from `src/protify`. |

Use `py` on Windows; you can use `python` if it points to the same interpreter. **In Docker:** use `-w /workspace/src/protify` and then `py -m main` or `py -m gui` (no `src.protify` prefix needed).

---

## How a run flows

When you pass datasets (`--data_names` or `--data_dirs`), the default flow is:

1. **Parse arguments** (CLI and optionally merge YAML).
2. **Apply settings:** Build `DataArguments`, `BaseModelArguments`, `ProbeArguments`, `EmbeddingArguments`, `TrainerArguments` from the combined config.
3. **Load data:** `get_datasets()` loads from HuggingFace and/or local dirs and normalizes columns.
4. **Embeddings:** `save_embeddings_to_disk()` computes (or downloads/reads) embeddings per model and saves to disk (or SQLite).
5. **Train:** One of: `run_nn_probes()` (default), `run_full_finetuning()`, `run_hybrid_probes()`, or `run_scikit_scheme()` (or W&B hyperopt).
6. **Write results:** Metrics are written to a TSV in `results_dir`.
7. **Plots:** `create_plots()` generates radar, bar, and heatmap PNGs in `plots_dir`.
8. **End log:** Session log is finalized (e.g. system info appended).

If you pass `--proteingym` and no datasets, only the ProteinGym zero-shot scoring path runs (see [ProteinGym](proteingym.md)).

---

## First CLI run

Minimal probe-only run (one model, one dataset, two epochs):

```bash
py -m src.protify.main --model_names ESM2-8 --data_names DeepLoc-2 --num_epochs 2
```

The reference config in `src/protify/yamls/base.yaml` runs the recommended stack instead: ESMC 300M read through its sparse autoencoder, max-pooled, into a gradient-boosted probe.

```bash
py -m src.protify.main --yaml_path src/protify/yamls/base.yaml --data_names DeepLoc-2
```

Command-line defaults:

- `--log_dir logs`, `--results_dir results`, `--plots_dir plots`
- `--embedding_save_dir embeddings`, `--model_save_dir weights`
- Probe: mlp; embeddings: mean and variance pooling; no saving of embeddings unless you set `--save_embeddings`

Results appear in `results/` (TSV of metrics) and `logs/` (session log). Plots are written to `plots/` after the run.

---

## First YAML run

1. Copy or edit the bundled config:

   ```bash
   # From repo root; edit paths if needed
   notepad src\protify\yamls\base.yaml
   ```

2. Set at least:
   - `data_names` (e.g. `['DeepLoc-2']`)
   - `model_names` (e.g. `['ESM2-8']`)
   - Other sections (paths, probe, trainer) have defaults in the file.

3. Run:

   ```bash
   py -m src.protify.main --yaml_path src/protify/yamls/base.yaml
   ```

CLI flags override YAML. For example:

```bash
py -m src.protify.main --yaml_path src/protify/yamls/base.yaml --num_epochs 5
```

---

## Where results go

| Output | Default path | Controlled by |
|--------|--------------|---------------|
| Session log | `logs/{random_id}.log` | `--log_dir` |
| Metrics TSV | `results/{random_id}.tsv` | `--results_dir` |
| Plots | `plots/{tsv_stem}/` | `--plots_dir` |
| Saved models | `weights/` | `--model_save_dir` |
| Embeddings | `embeddings/` | `--embedding_save_dir` |

The `random_id` is generated at session start (or from `PROTIFY_JOB_ID` / replay path). See [Logging and replay](logging_and_replay.md).

---

## See also

- [Configuration](cli_and_config.md) for all CLI and YAML options
- [Data](data.md) for datasets and `data_dirs`
- [Models and embeddings](models_and_embeddings.md) for base models and embedding options
- [Probes and training](probes_and_training.md) for probe types and training flows
- [GUI](gui.md) for the Tk interface
