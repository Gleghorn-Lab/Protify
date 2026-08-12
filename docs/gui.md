# GUI

This page documents the Tk-based GUI: structure (tabs, settings_vars, full_args), the recommended flow (Start session, Get data, Get embeddings, Run trainer), and how background tasks keep the UI responsive. For Modal submission from the GUI, see [Modal](modal.md).

---

## Overview

The GUI is the same **MainProcess** as the CLI, run with `GUI=True` and an initially empty namespace. All configuration comes from the widgets: there is no YAML or argparse. Each tab updates a shared `full_args` namespace and, when the user clicks the relevant button, builds the appropriate *Arguments objects (DataArguments, BaseModelArguments, ProbeArguments, EmbeddingArguments, TrainerArguments) and calls the same methods as the CLI (get_datasets, save_embeddings_to_disk, run_nn_probes, etc.). Long-running work runs in a thread pool via **run_in_background** so the UI stays responsive.

---

## How it works

1. **Entry:** `py -m src.protify.gui` (or `py -m gui` from `src/protify`) creates `tk.Tk()` and `GUI(root)` (MainProcess with empty args and GUI=True), then `root.mainloop()`.
2. **Tabs** update `self.settings_vars` (Tk variables) and, when the user applies or runs something, write into `self.full_args`. Buttons then build *Arguments from `full_args.__dict__` and call the corresponding MainProcess methods (optionally via run_in_background).
3. **BackgroundTask** and **check_task_queue:** Blocking work (session start, get data, get embeddings, run trainer, ProteinGym, scikit, replay, generate plots, Modal deploy/submit/poll/fetch) is run in a `ThreadPoolExecutor`. The GUI polls the task queue every 100 ms and updates the UI when tasks complete.

---

## Tabs

| Tab | Purpose |
|-----|---------|
| **Info** | API keys (HF, W&B, Modal, Synthyra), paths, W&B sweep options, data transform flags. "Start session" creates dirs, sets log_dir/results_dir, starts the session log. |
| **Model** | Select models, including all three CARBON presets, or configure paths + types. "Apply" sets full_args.model_names (or model_paths/model_types) and builds BaseModelArguments. |
| **Data** | Dataset names or local dirs, max_length, trim, delimiter, col_names, translation flags, multi_column. "Get data" sets data_args and calls get_datasets(). |
| **Embedding** | Batch size, workers, download_embeddings, matrix_embed, pooling types, save_embeddings, sql. "Get embeddings" sets embedding_args and calls save_embeddings_to_disk(). |
| **Probe** | Probe type, tokenwise, sizes, dropout, pre_ln, rotary, attention_backend, probe_pooling_types, Save Model, Push Raw Probe, LoRA, sim_type. "Configure probe" updates full_args and builds ProbeArguments. |
| **Trainer** | Epochs, batch sizes, lr, weight_decay, patience, full_finetuning, hybrid_probe, num_runs. "Run trainer" sets trainer_args and runs the same branches as CLI (W&B hyperopt, full finetuning, hybrid, or run_nn_probes). |
| **Modal** | Deploy backend, submit remote run, poll status, cancel, fetch logs/results/plots. Uses run_in_background for deploy, submit, poll, cancel, fetch. |
| **ProteinGym** | DMS IDs, mode, scoring method, etc. "Run" calls run_proteingym_zero_shot() (or compare_scoring_methods) in background. |
| **Scikit** | use_scikit and scikit options. "Run" calls run_scikit_scheme() in background. |
| **Replay** | replay_path. "Start replay" parses log with LogReplayer, creates a new MainProcess(replay_args, GUI=False), runs run_replay in background. |
| **Viz** | results file and output dir. "Generate plots" calls create_plots() in background. |

---

## settings_vars and full_args

- **settings_vars:** A dict of Tk variables (StringVar, IntVar, BooleanVar, etc.) bound to widgets. When the user changes a widget, the variable updates; when applying, the code reads from these variables and sets `full_args.*`.
- **full_args:** A namespace (or dict-like) holding the current configuration. It is the same schema as CLI + YAML. When you click "Apply" or "Run", the relevant subset of full_args is used to construct DataArguments, BaseModelArguments, ProbeArguments, EmbeddingArguments, or TrainerArguments, and then the corresponding MainProcess method is called.

There is no separate "load config file" in the GUI; each tab pushes its state into full_args when the user acts.

---

## Recommended flow

1. **Start session** (Info tab): Set API keys and paths, click "Start session" to create dirs and start the log.
2. **Get data** (Data tab): Set data_names or data_dirs and options, click "Get data" to load datasets.
3. **Get embeddings** (Embedding tab): Set embedding options, click "Get embeddings" to compute or load embeddings.
4. **Run trainer** (Trainer tab): Set probe and trainer options, click "Run trainer" to run probe-only, full finetuning, hybrid, or W&B hyperopt.

Alternatively, run **ProteinGym** (ProteinGym tab) or **Scikit** (Scikit tab) instead of the trainer. Use **Modal** tab to submit the current config to Modal and fetch results. Use **Replay** to re-run a prior session from its log file. Use **Viz** to generate comparison plots from an existing results TSV.

---

## Background tasks

- **run_in_background(self, target, *args, **kwargs):** Submits `target(*args, **kwargs)` to a thread pool and appends the future to a queue. The GUI continues to respond.
- **check_task_queue():** Runs every 100 ms (via `master.after(100, check_task_queue)`). Checks completed futures, handles results or exceptions, and updates the UI (e.g. enable buttons, show status). This keeps the main thread free for Tk events.

---

## See also

- [Modal](modal.md) for the Modal tab and backend
- [Getting started](getting_started.md) for how to launch the GUI
- [Configuration](cli_and_config.md) for the meaning of options (same as GUI)
- [Logging and replay](logging_and_replay.md) for session log and replay
