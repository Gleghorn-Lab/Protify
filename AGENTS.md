# Protify

## Purpose and Sources

This is the standalone Synthyra Protify checkout. Keep its public training, evaluation, packaging, and cloud interfaces independent of the parent synth workspace.

- `docs/getting_started.md` and `docs/cli_and_config.md`: entrypoints and configuration
- `docs/probes_and_training.md`: probe behavior
- `docs/testing.md`: test scopes and working directories
- `src/protify/base_models/supported_models.py`: authoritative model registry
- `src/protify/data/supported_datasets.py`: authoritative dataset registry

## Architectural Invariants

- CLI, YAML, GUI, and cloud dispatch share the same `MainProcess` pipeline.
- Keep heavy model and dataset loading lazy where the registries and factories are lazy.
- `src/protify/fastplms/` is a vendored repository with its own guidance. Do not mix FastPLMs changes into a Protify task.
- `--parallel_probe_runs` applies only to compatible pooled, sequence-level MLP probes. Matrix or tokenwise probes, transformer probes, Lyra probes, PPI run-specific datasets, and full fine-tuning use the sequential fallback.
- A resolved model name carries everything that changes its embeddings. ESMC sparse autoencoder aliases expand to names such as `ESMC-300-SAE-l23-k64-c8192`, so the layer, sparsity, and codebook width reach the embedding cache filename and the results table without any separate cache key.
- FastPLMs owns the sparse autoencoder runtime, through `ESMplusplusModel.load_sae_models` and `compute_sae=True`, which returns one sparse `(valid_tokens, codebook_dim)` tensor per attached layer. `src/protify/base_models/esmc_sae.py` owns only what is Protify's: the published-coverage registry, model-name resolution, pooling to one vector per protein, and the storage policy. Do not reimplement the SAE encoder here.
- SAE models return one vector per sequence. They accept `max`, `mean`, and `sum` only, and reject `--matrix_embed`, because the dense per-residue codebook tensor does not fit the cache. Pooling drops the leading and trailing special tokens, matching the Biohub reference workflow.
- Pair datasets swap protein A and B only on the training split, and only under `--random_pair_flipping`. Swapping a validation or test pair would make a reported score depend on the random state.
- Coordinate-list blob storage is selected by the model, through `EsmcSaeForEmbedding.sparse_storage`, not per embedding. Every other model writes dense and pays nothing for the feature. bfloat16 keeps its own bit pattern under dtype code 3; code 1 is the older float16-byte encoding and is read but never written.
- `probe_type` is normalized once in `ProbeArguments` and `ParallelProbeRunSpec`. `mlp` is the current name; `linear` is accepted because saved probe configs, exported repositories, and existing YAML carry it, and `MLPProbeConfig.model_type` stays `linear_probe`.
- Estimator probes (`xgboost`, `lightgbm`, `random_forest`) run through `run_estimator_probes`, share the embedding cache and results table with the neural probes, and skip embedding standardization because it cannot change a tree's splits and would densify sparse features.
- The shared metric functions read their input as raw logits, because neural probes produce logits: the single-label one softmaxes and the multi-label one sigmoids before thresholding at 0.5. `estimator_probe._predictions` therefore returns log probabilities and log odds, never the probabilities `predict_proba` gives back.
- Matrix embeddings are cached with the leading and trailing special tokens, so a stored per-residue embedding has `len(seq) + 2` rows. Nothing may reshape one against `len(seq)`.

## Python Coding Standard

Write direct, readable Python for human maintainers. Prefer domain language and visible data flow over cleverness, speculative generality, or abstraction for its own sake.

Apply this priority order when guidance conflicts:

1. Follow the user's explicit scope and requested behavior.
2. Preserve correctness, security, public interfaces, data schemas, tested behavior, and scientific conventions.
3. Follow repository instructions and enforced formatter, linter, and type-checker configuration.
4. Apply the preferences below.
5. Minimize unrelated diff churn.

Treat cleanup and refactoring as behavior-preserving unless behavior changes are explicitly requested. Observable behavior includes exceptions, import side effects, CLI output, serialization, random-number use, dtype, device, tensor shape, and documented performance guarantees. Do not silently fix unrelated bugs or expand an ambiguous cleanup into a repository-wide rewrite. Exclude generated, vendored, migration, build, and external code unless it is explicitly in scope.

### Validation and Workflow

- Read the relevant repository guidance, configuration, implementation, and focused tests before editing.
- Establish a behavioral baseline before structural changes and run the same focused tests afterward. Record pre-existing failures and compare like with like; never weaken tests to make a refactor pass.
- Keep validation proportional. Use formatter, linter, type checker, compile checks, or focused CPU-only tests for mechanical changes; require meaningful behavioral tests for structural changes.
- Do not trigger downloads, CUDA, large datasets, remote machines, or slow integration suites unless requested.
- Make safe mechanical improvements first. Make structural changes only when they materially improve ownership, cohesion, or visible data flow. Review the final diff for behavior drift, stale imports, unnecessary movement, and formatter conflicts.

### Imports, Types, and Control Flow

- Keep the module docstring first and `from __future__` imports immediately after it. Then group direct `import` statements before ordinary `from ... import ...` statements. Within those blocks, place standard-library names before third-party names and keep repository-local imports in a distinct final section. Leave two blank lines after the complete import block.
- Preserve guarded, optional, registration, and initialization-sensitive import ordering. Do not fight an enforced formatter.
- Give each function one narrow, typed responsibility. Prefer precise domain types and parameterized containers; use `Any` only at genuinely dynamic boundaries.
- Respect the minimum Python version and runtime annotation behavior. Do not alter framework-discovered fields, import cycles, or serialization schemas merely to add types.
- Use domain-specific names instead of vague names such as `data`, `result`, `info`, or `manager` when a clearer term exists.
- Prefer explicit sequential control flow. Separate distinct guards, loops, exception handling, and logical phases with blank lines when that improves scanning.
- Raise ordinary exceptions for invalid external input and use assertions only for internal invariants or programmer assumptions.
- Avoid broad `except Exception`, silent fallback cascades, unsupported compatibility shims, wrappers that merely rename one call, and one-use helpers that do not name a real concept or isolate a testable phase.
- Keep argument parser declarations on one line when they remain comfortable to read.

### Comments and Numerical Code

- Comment intent, assumptions, units, biological conventions, tensor shapes, non-obvious mechanics, and design rationale. Remove comments and docstrings that merely narrate syntax or repeat a precise name and type.
- For NumPy, PyTorch, JAX, and similar code, maintain a complete, accurate shape trace through tensor-valued inputs, assignments, transformations, layer calls, reductions, mutations, and returns.
- Use lowercase shape symbols such as `b` (batch), `l` (sequence length), `d` (hidden width), `h` (heads), `c` (classes), and `n` (generic count). Define nonstandard or dynamic symbols near first use, use `()` for scalars, and use an ellipsis when rank cannot be proven. Never invent a fixed shape.
- Put shape comments on the same line as the operation when practical. Treat stale shape comments as correctness defects.
- Use uppercase mathematical tensor names such as `X`, `Q`, `K`, `V`, and `W_q`; keep descriptive tensor names such as `hidden_states`, `attention_mask`, and `logits` lowercase. Preserve established domain notation when clearer.

### Modules and Interfaces

- Give each module one primary responsibility. Split by ownership and dependency direction, not by line count, and keep reusable helpers near the feature that owns them. Do not create catch-all `utils.py` modules.
- Keep extracted modules' public surfaces small and intentional. Avoid circular imports and implicit shared state.
- Use a class when state and behavior form a meaningful domain object or workflow. Prefer a function, typed value, or dataclass for simple transformations and records.
- Keep entry points thin: parse arguments, construct configuration and collaborators, then invoke a short sequence whose names expose the workflow.
- Extract abstractions only when they remove real complexity, reduce meaningful duplication, name a cohesive concept, or isolate a separately testable phase.

## Canonical Commands

Build and run the broad CPU suite from the repository root:

```powershell
docker build -t protify-env:latest .
docker run --rm --ipc=host -v ${PWD}:/workspace -w /workspace protify-env:latest \
  python -m pytest src/protify/testing_suite -v -m "not gpu and not slow"
```

Focused parallel-probe tests run with `src/protify/` as the working directory:

```powershell
docker run --rm --gpus all --ipc=host -v ${PWD}:/workspace -e PYTHONPATH=/workspace \
  -w /workspace/src/protify protify-env:latest python -m pytest \
  testing_suite/test_parallel_probe_plan.py testing_suite/test_parallel_linear_probe.py -v
```
