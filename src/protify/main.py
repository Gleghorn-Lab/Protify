import argparse
import json
import os
import sys

import yaml
from types import SimpleNamespace

# Ensure src is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Docker/Mount fallback: If the 'src/protify' folder was mounted directly
# as the root (e.g. /workspace) and lost its 'protify' wrapper name.
try:
    import protify
except ImportError:
    if os.path.exists(os.path.join(_current_dir, "__init__.py")):
        import importlib.util
        _spec = importlib.util.spec_from_file_location("protify", os.path.join(_current_dir, "__init__.py"))
        if _spec is not None:
            _protify_mod = importlib.util.module_from_spec(_spec)
            sys.modules["protify"] = _protify_mod
            _spec.loader.exec_module(_protify_mod)

from protify.cloud_cli import _run_on_cloud, _should_auto_run_cloud
from protify.probes.estimator_probe import ESTIMATOR_PROBE_TYPES, check_task_type, is_estimator_probe


def parse_arguments():  
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Script with arguments mirroring the provided YAML settings.")
    # ----------------- ID ----------------- #
    parser.add_argument("--hf_username", default="Synthyra", help="Hugging Face username.")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token.")
    parser.add_argument("--synthyra_api_key", default=None, help="Synthyra API key.")
    parser.add_argument("--wandb_api_key", default=None, help="Wandb API key.")
    parser.add_argument("--cloud_api_key", default=None, help="Cloud backend API key. When provided, jobs are dispatched to the remote cloud backend.")
    parser.add_argument("--cloud_url", default=None, help="Cloud backend URL (default: https://api.synthyra.com).")
    parser.add_argument("--cloud_gpu_type", default=None, help="GPU type for cloud execution (e.g. A10, A100, H100).")
    parser.add_argument("--cloud_timeout_seconds", type=int, default=None, help="Timeout in seconds for cloud jobs (default: 86400).")
    parser.add_argument("--cloud_poll_interval", type=int, default=None, help="Poll interval in seconds for cloud job status (default: 5).")
    parser.add_argument("--cloud_artifacts_dir", default=None, help="Local directory to save cloud job artifacts (default: cloud_artifacts).")

    # ----------------- Paths ----------------- #
    parser.add_argument("--hf_home", type=str, default=None, help="Customize the HF cache directory.")
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to the YAML file.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to the log directory.")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to the results directory.")
    parser.add_argument("--model_save_dir", default="weights", help="Directory to save models.")
    parser.add_argument("--embedding_save_dir", default="embeddings", help="Directory to save embeddings.")
    parser.add_argument("--download_dir", default="Synthyra/vector_embeddings", help="Directory to download embeddings to.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to save plots.")
    parser.add_argument("--replay_path", type=str, default=None, help="Path to the replay file.")
    parser.add_argument("--pretrained_probe_path", type=str, default=None, help=argparse.SUPPRESS)
    
    # ----------------- DataArguments ----------------- #
    parser.add_argument("--delimiter", default=",", help="Delimiter for data.")
    parser.add_argument("--col_names", nargs="+", default=["seqs", "labels"], help="Column names.") # DEPRECATED, found automatically now
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument(
        "--padding",
        choices=["max_length", "longest"],
        default="max_length",
        help="Padding strategy. 'max_length' pads all sequences to --max_length (recommended for torch.compile + flex attention). 'longest' pads to the longest sequence in each batch.",
    )
    parser.add_argument("--trim", action="store_true",
                        help="Truncate sequences longer than --max_length instead of dropping them from the dataset.")
    parser.add_argument("--data_names", nargs="+", default=[], help="List of HF dataset names.")
    parser.add_argument("--data_dirs", nargs="+", default=[], help="List of local data directories.")
    parser.add_argument("--aa_to_dna", action="store_true", help="Translate amino-acid sequences to DNA codon sequences using common human synonymous codons.")
    parser.add_argument("--aa_to_rna", action="store_true", help="Translate amino-acid sequences to RNA codon sequences using common human synonymous codons.")
    parser.add_argument("--dna_to_aa", action="store_true", help="Translate DNA codon sequences to amino-acid sequences and drop stop codons.")
    parser.add_argument("--rna_to_aa", action="store_true", help="Translate RNA codon sequences to amino-acid sequences and drop stop codons.")
    parser.add_argument("--codon_to_aa", action="store_true", help="Translate codon-token sequences to amino-acid sequences and drop stop codons.")
    parser.add_argument("--aa_to_codon", action="store_true", help="Translate amino-acid sequences to codon-token sequences.")
    parser.add_argument("--random_pair_flipping", action="store_true", help="Randomly swap paired inputs during training.")

    # ----------------- BaseModelArguments ----------------- #
    parser.add_argument("--model_names", nargs="+", default=None, help="List of preset model names to use (e.g. ESM2-8). Mutually exclusive with --model_paths/--model_types.")
    parser.add_argument("--model_paths", nargs="+", default=None, help="List of model paths (HuggingFace or local). Must be paired with --model_types. Mutually exclusive with --model_names.")
    parser.add_argument("--model_types", nargs="+", default=None, help="List of model type keywords paired with --model_paths (e.g. esm2, esmc, protbert, prott5, ankh, glm, dplm, dplm2, protclm, onehot, amplify, e1, calm, custom, random).")
    parser.add_argument("--model_dtype", type=str, choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"], default="bf16", help="Data type for loading base models.")
    parser.add_argument("--use_xformers", action="store_true", help="Use xformers memory-efficient attention for AMPLIFY models.")
    parser.add_argument("--sae_layer", type=int, default=None, help="ESMC hidden-state layer the sparse autoencoder reads. Defaults to the 75%% depth layer each backbone publishes (23, 27, and 60). Only layers other than that one are restricted to --sae_k 64 with --sae_codebook_dim 16384.")
    parser.add_argument("--sae_k", type=int, choices=[16, 32, 64, 128, 256, 512], default=None, help="Active sparse autoencoder features per residue. Defaults to 64.")
    parser.add_argument("--sae_codebook_dim", type=int, choices=[8192, 16384, 32768, 65536, 131072], default=None, help="Sparse autoencoder codebook width, which is the embedding dimension per pooling type. Defaults to 8192.")

    # ----------------- ProbeArguments ----------------- #
    parser.add_argument("--probe_type", choices=["mlp", "linear", "transformer", "lyra", "xgboost", "lightgbm", "random_forest"], default="mlp", help="Type of probe. 'mlp' is the pooled feed-forward probe ('linear' is its former name and still works). 'xgboost', 'lightgbm', and 'random_forest' fit an estimator on pooled embeddings instead of a network.")
    parser.add_argument("--probe_n_jobs", type=int, default=-1, help="Worker processes for estimator probes (xgboost, lightgbm, random_forest). -1 uses every core.")
    parser.add_argument("--tokenwise", action="store_true", help="Use a tokenwise probe (per-token outputs) instead of a sequence-level probe.")
    parser.add_argument("--hidden_size", type=int, default=8192, help="Hidden dimension size for probe.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers.")
    parser.add_argument("--pre_ln", action="store_false",
                        help="Disable pre-layernorm in the transformer probe (pre-LN enabled by default).")
    parser.add_argument("--classifier_size", type=int, default=4096, help="Feed-forward dimension.")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout rate for the transformer layers.")
    parser.add_argument("--classifier_dropout", type=float, default=0.2, help="Dropout rate for the classifier.")
    parser.add_argument("--head_size", type=int, default=128, help="Attention head dimension. n_heads is derived as hidden_size // head_size.")
    parser.add_argument("--n_heads", type=int, default=None, help="[DEPRECATED] Use --head_size. If provided, head_size is derived as hidden_size // n_heads.")
    parser.add_argument("--rotary", action="store_false",
                        help="Disable rotary embeddings in the transformer probe (rotary enabled by default).")
    parser.add_argument("--attention_backend", choices=["kernels", "flex", "sdpa"], default="flex", help="Attention backend for transformer-style probes.")
    parser.add_argument("--output_s_max", action="store_true", help="Return s_max bounds from transformer-style probe attention layers.")
    parser.add_argument("--probe_pooling_types", nargs="+", default=["mean", "var"], help="Pooling types to use.")
    parser.add_argument("--bom_k", type=int, default=60, help="K-mer window size for 'bom' pooling in the transformer probe. Default 60 is a cross-task compromise from Hoang & Singh 2025: peak for DPI (Section 4.3), mid-range of the {20,40,60,80,100} sweep on FLUO/BLAC, and close to the reported k=100 optimum for remote homology. Only used when 'bom' is in --probe_pooling_types.")
    parser.add_argument("--use_bias", action="store_true", help="Use bias terms in Linear layers.")
    parser.add_argument("--expansion_ratio", type=float, default=8/3, help="FFN expansion ratio for transformer probes.")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model/probe to disk.")
    parser.add_argument("--push_raw_probe", action="store_true", help="With --save_model, push the raw probe class to the Hub instead of the packaged AutoModel.")
    parser.add_argument("--push_raw_probe_repo", type=str, default=None, help="Custom HF repo id for --push_raw_probe. If omitted, auto-generated.")
    parser.add_argument("--production_model", action="store_true", help="Train a production-grade scikit model (used with --use_scikit).")
    parser.add_argument("--lora", action="store_true", help="Wrap the base model in LoRA adapters during full-finetuning / hybrid training.")
    parser.add_argument("--lora_r", type=int, default=8, help="Number of trainable parameters in the LoRA model.")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="Alpha for the LoRA model.")
    parser.add_argument("--lora_dropout", type=float, default=0.01, help="Dropout rate for the LoRA model.")
    parser.add_argument("--sim_type", choices=["dot", "euclidean", "cosine"], default="dot", help="Cross-attention mechanism for token-parameter-attention")
    parser.add_argument("--add_token_ids", action="store_true", help="Add learned token-type embeddings to distinguish protein A vs B in PPI tasks.")

    # ----------------- ScikitArguments ----------------- #
    parser.add_argument("--scikit_n_iter", type=int, default=10, help="Number of iterations for scikit model.")
    parser.add_argument("--scikit_cv", type=int, default=3, help="Number of cross-validation folds for scikit model.")
    parser.add_argument("--scikit_random_state", type=int, default=None, help="Random state for scikit model (if None, uses global seed).")
    parser.add_argument("--scikit_model_name", type=str, default=None, help="Name of the scikit model to use.")
    parser.add_argument("--scikit_model_args", type=str, default=None, help="JSON string of hyperparameters to use (skips tuning). E.g. '{\"n_estimators\": 500, \"max_depth\": 7}'")
    parser.add_argument("--use_scikit", action="store_true", help="Use a scikit-learn model instead of a neural probe.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of processes to use in scikit.")

    # ----------------- EmbeddingArguments ----------------- #
    parser.add_argument("--embedding_batch_size", type=int, default=16, help="Batch size for embedding generation.")
    parser.add_argument("--embedding_num_workers", type=int, default=0, help="Number of worker processes for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--download_embeddings", action="store_true", help="Download pre-computed embeddings from the Hub instead of computing them locally.")
    parser.add_argument("--matrix_embed", action="store_true", help="Store per-token (matrix) embeddings instead of pooled vector embeddings.")
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean", "var"], help="Pooling types for embeddings.")
    parser.add_argument("--embedding_hidden_state_index", type=int, default=-1, help="Hidden-state tuple index to embed from. -1 uses the final hidden state.")
    parser.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings to disk.")
    parser.add_argument("--embed_dtype", type=str, choices=["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"], default=None, help="Data type for embeddings. If omitted, uses --model_dtype.")
    parser.add_argument("--no_embedding_scaler", dest="embedding_scaler", action="store_false", default=True,
                        help="Disable StandardScaler for pooled vector embeddings during probe/scikit training.")
    parser.add_argument("--sql", action="store_true", help="Store embeddings in a SQLite backend (streamed at train time) instead of in-RAM .pth.")
    parser.add_argument("--read_scaler", type=int, default=100, help="Read scaler for SQL storage.")

    # ----------------- Multi-Column Sequences ----------------- #
    parser.add_argument("--multi_column", nargs="+", default=None, help="If set, list of sequence column names to combine per sample.")

    # ----------------- TrainerArguments ----------------- #
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--probe_batch_size", type=int, default=64, help="Batch size for probe training.")
    parser.add_argument("--base_batch_size", type=int, default=4, help="Batch size for base model training.")
    parser.add_argument("--probe_grad_accum", type=int, default=1, help='Gradient accumulation steps for probe training.')
    parser.add_argument("--base_grad_accum", type=int, default=8, help='Gradient accumulation steps for base model training.')
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--probe_lr", type=float, default=None,
                        help="Learning rate for probe training. If omitted, falls back to --lr.")
    parser.add_argument("--lr_scheduler", type=str, default='cosine',
                        help="Learning rate scheduler passed to transformers.TrainingArguments.")
    parser.add_argument("--optimizer", type=str, default='adamw_torch',
                        help="Optimizer passed to transformers.TrainingArguments.")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping (probe phase, and base phase unless --base_patience is set).")
    parser.add_argument("--base_num_epochs", type=int, default=None,
                        help="Epoch count for the base-model phase of hybrid / full-finetuning training. If omitted, falls back to --num_epochs.")
    parser.add_argument("--base_patience", type=int, default=None,
                        help="Early-stopping patience for the base-model phase of hybrid / full-finetuning training. If omitted, falls back to --patience.")
    parser.add_argument("--base_lr", type=float, default=None,
                        help="Learning rate for the base-model phase of hybrid / full-finetuning training (useful when LoRA/full-FT wants a different LR than the probe). If omitted, falls back to --lr.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility (if omitted, current time is used).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic behavior for reproducibility (slightly slower training).")
    parser.add_argument("--full_finetuning", action="store_true", help="Fully fine-tune the base model end-to-end.")
    parser.add_argument("--hybrid_probe", action="store_true", help="Train probe first, then fine-tune base model + probe jointly.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of training runs with different seeds. Results will show mean±std across runs.")
    parser.add_argument("--parallel_probe_runs", action="store_true",
                        help="Train eligible linear-probe num_runs in one vectorized HF Trainer pass.")
    parser.add_argument("--parallel_probe_batch_mode", type=str, default='shared',
                        choices=['shared', 'run_specific'],
                        help="Batch mode for --parallel_probe_runs: shared reuses each minibatch across runs; run_specific uses deterministic per-run training permutations.")
    parser.add_argument("--parallel_probe_index_strategy", type=str, default='permutation',
                        choices=['permutation', 'affine'],
                        help="Index strategy for --parallel_probe_batch_mode run_specific: permutation materializes per-run shuffled indices; affine uses memory-free deterministic bijections.")
    parser.add_argument("--parallel_probe_max_group_size", type=int, default=None,
                        help="Optional maximum number of seeded linear probes to place in one vectorized parallel-probe bank.")
    parser.add_argument("--parallel_probe_training_state_budget_gb", type=float, default=None,
                        help="Optional static training-state budget in GiB for automatic --parallel_probe_runs group-size chunking.")
    parser.add_argument("--parallel_probe_estimated_peak_budget_gb", type=float, default=None,
                        help="Optional static estimated peak-memory budget in GiB for automatic --parallel_probe_runs group-size chunking.")
    parser.add_argument("--parallel_probe_max_grad_norm", type=float, default=0.0,
                        help="Gradient clipping max norm for --parallel_probe_runs. Default 0 disables global clipping to avoid coupling seed banks.")
    parser.add_argument("--parallel_probe_grad_clip_mode", type=str, default='global',
                        choices=['none', 'global', 'per_run'],
                        help="Gradient clipping mode for --parallel_probe_runs: none disables clipping, global uses HuggingFace global clipping, per_run clips each seed bank independently.")
    parser.add_argument("--parallel_probe_ensemble_average_mode", type=str, default='logits',
                        choices=['logits', 'probabilities'],
                        help="How to average parallel-probe seed-bank predictions for reported ensemble metrics.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on probes during training (compiled by default).")

    # ----------------- Balanced Regression Metrics (EpHod-style) ----------------- #
    parser.add_argument("--balanced_regression_metrics", action="store_true",
                        help="Enable EpHod-style balanced regression metrics on valid/test (enabled by default).")
    parser.add_argument("--no_balanced_regression_metrics", dest="balanced_regression_metrics", action="store_false",
                        help="Disable EpHod-style balanced regression metrics.")
    parser.set_defaults(balanced_regression_metrics=True)
    parser.add_argument("--balanced_weight_method", type=str, default='bin_inv',
                        choices=['none', 'bin_inv', 'bin_inv_sqrt', 'LDS_inv', 'LDS_inv_sqrt', 'LDS_extreme'],
                        help="Weighting scheme for balanced regression metrics.")
    parser.add_argument("--balanced_bin_borders", type=float, nargs='+', default=None,
                        help="Explicit bin borders for balanced metrics (e.g., 5 9 for pH). Default: 1/3 and 2/3 quantiles of training labels.")
    parser.add_argument("--balanced_n_resamples", type=int, default=100,
                        help="Number of resamples for balanced Pearson/Spearman (default: 100).")
    parser.add_argument("--balanced_lds_bins", type=int, default=100,
                        help="Number of bins for LDS density estimation.")
    parser.add_argument("--balanced_lds_ks", type=int, default=5,
                        help="Kernel size for LDS Gaussian smoothing.")
    parser.add_argument("--balanced_lds_sigma", type=float, default=2.0,
                        help="Sigma for LDS Gaussian smoothing.")

    # ----------------- ProteinGym Arguments ----------------- #
    parser.add_argument("--dms_ids", nargs="+", default=["all"],
                        help="ProteinGym DMS assay IDs to evaluate (space-separated), or 'all' to run all assays.")
    parser.add_argument("--proteingym", action="store_true", help="Run a ProteinGym zero-shot experiment.")
    parser.add_argument("--mode", type=str, default='benchmark',
                        help="ProteinGym zero-shot mode: 'benchmark', 'indels', 'multiples', 'singles'")
    parser.add_argument("--scoring_method", choices=["masked_marginal", "mutant_marginal", "wildtype_marginal", "pll", "global_log_prob"], default="masked_marginal",
                        help="Select a scoring method for ProteinGym zero-shot.")
    parser.add_argument("--scoring_window", choices=["optimal", "sliding"], default="optimal",
                        help="Select how to slice the sequence for ProteinGym zero-shot.")
    parser.add_argument("--pg_batch_size", type=int, default=32,
                        help="Batch size for ProteinGym zero-shot scoring (default: 32).")
    parser.add_argument("--compare_scoring_methods", action="store_true",
                        help="Compare different scoring methods across models and DMS assays.")
    parser.add_argument("--score_only", action="store_true",
                        help="Run only the ProteinGym benchmarking script on existing CSV files; skip zero-shot scoring.")

    # ----------------- W&B Arguments ----------------- #
    parser.add_argument("--use_wandb_hyperopt", action="store_true", help="Run a Weights & Biases hyperparameter sweep instead of a single training run.")
    parser.add_argument("--wandb_project", type=str, default="Protify", help="W&B project name for sweeps.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user) for sweeps.")
    parser.add_argument("--sweep_config_path", type=str, default="yamls/sweep.yaml", help="Path to W&B sweep config YAML.")
    parser.add_argument("--sweep_count", type=int, default=10, help="Number of hyperparameter trials to run in the sweep.")
    parser.add_argument("--sweep_method", type=str, default="bayes", choices=["bayes", "grid", "random"], help="Sweep method for hyperparameter optimization.")
    parser.add_argument("--sweep_metric_cls",type=str,default="eval_loss", help="Classification metric to optimize during sweep (e.g., eval_f1, eval_accuracy, eval_mcc)")
    parser.add_argument("--sweep_metric_reg",type=str,default="eval_loss", help="Regression metric to optimize during sweep (e.g., eval_r_squared, eval_spearman_rho, eval_pearson_rho)")
    parser.add_argument("--sweep_goal", type=str, default='minimize', choices=['maximize', 'minimize'], help="Goal for the sweep metric (maximize/minimize)")
    parser.add_argument("--sweep_name", type=str, default=None, help="Display name for the W&B sweep. Overrides 'name' in the sweep YAML.")
    args = parser.parse_args()

    if args.pretrained_probe_path is not None:
        raise AssertionError("--pretrained_probe_path is no longer supported. Use --save_model outputs or packaged probe exports instead.")

    if args.n_heads is not None:
        import warnings
        warnings.warn(
            "--n_heads is deprecated and will be removed in a future release. "
            "Use --head_size instead (n_heads = hidden_size // head_size).",
            DeprecationWarning,
            stacklevel=2,
        )
        assert args.hidden_size % args.n_heads == 0, (
            f"--hidden_size {args.hidden_size} not divisible by deprecated --n_heads {args.n_heads}"
        )
        derived_head_size = args.hidden_size // args.n_heads
        explicit_head_size = "--head_size" in raw_argv
        if explicit_head_size:
            assert derived_head_size == args.head_size, (
                f"--head_size {args.head_size} conflicts with deprecated --n_heads {args.n_heads} "
                f"(derived head_size={derived_head_size})"
            )
        args.head_size = derived_head_size
        args.n_heads = None

    # Validate model_names vs model_paths/model_types mutual exclusivity
    if args.model_paths is not None:
        assert args.model_types is not None, "--model_types is required when --model_paths is provided."
        assert len(args.model_paths) == len(args.model_types), f"--model_paths ({len(args.model_paths)}) and --model_types ({len(args.model_types)}) must have the same length."
        assert args.model_names is None, "--model_names cannot be used together with --model_paths/--model_types."
    elif args.model_types is not None:
        assert args.model_paths is not None, "--model_paths is required when --model_types is provided."
    if args.model_names is None and args.model_paths is None:
        args.model_names = ["ESM2-8"]

    # Only probes that consume a pooled vector can run without per-residue embeddings.
    pooled_vector_probes = ("mlp", "linear") + ESTIMATOR_PROBE_TYPES
    assert args.probe_type in pooled_vector_probes or args.matrix_embed, (
        f"--probe_type {args.probe_type} reads per-residue embeddings, so --matrix_embed must be set. "
        f"Probes that take a pooled vector are {pooled_vector_probes}."
    )

    if args.hf_token is not None:
        from huggingface_hub import login
        # Override environment variable to ensure this token is used
        os.environ["HF_TOKEN"] = args.hf_token
        login(args.hf_token)
        print(f"Logged in to HuggingFace Hub with token from arguments")
    else:
        # Check if token exists in environment
        hf_token_env = os.environ.get("HF_TOKEN")
        if hf_token_env:
            print(f"Note: HF_TOKEN found in environment")
            print(f"Note: This token will be used for read operations only unless overridden")
    if args.wandb_api_key is not None:
        try:
            import wandb
            wandb.login(key=args.wandb_api_key)
            print('Logged into Weights & Biases')
        except Exception as e:
            print(f'W&B login failed: {e}')
    if args.synthyra_api_key is not None:
        print('Synthyra API key provided')

    if args.yaml_path is not None:
        with open(args.yaml_path, 'r') as file: 
            settings = yaml.safe_load(file)
        yaml_args = SimpleNamespace(**settings)

        def _merge_store_true(cli_value: bool, key: str) -> bool:
            if cli_value:
                return True
            if key in yaml_args.__dict__:
                return bool(yaml_args.__dict__[key])
            return False

        def _merge_if_cli_explicit(key: str) -> None:
            explicit = any(token == f"--{key}" or token.startswith(f"--{key}=") for token in raw_argv)
            if explicit or key not in yaml_args.__dict__:
                yaml_args.__dict__[key] = args.__dict__[key]

        if "pretrained_probe_path" in yaml_args.__dict__ and yaml_args.pretrained_probe_path is not None:
            raise AssertionError("pretrained_probe_path is no longer supported. Use save_model outputs or packaged probe exports instead.")

        if args.hf_token is not None:
            yaml_args.hf_token = args.hf_token
        elif "hf_token" not in yaml_args.__dict__:
            yaml_args.hf_token = None

        if args.hf_home is not None:
            yaml_args.hf_home = args.hf_home
        elif "hf_home" not in yaml_args.__dict__:
            yaml_args.hf_home = None

        if args.synthyra_api_key is not None:
            yaml_args.synthyra_api_key = args.synthyra_api_key
        elif "synthyra_api_key" not in yaml_args.__dict__:
            yaml_args.synthyra_api_key = None

        if args.wandb_api_key is not None:
            yaml_args.wandb_api_key = args.wandb_api_key
        elif "wandb_api_key" not in yaml_args.__dict__:
            yaml_args.wandb_api_key = None

        if args.cloud_api_key is not None:
            yaml_args.cloud_api_key = args.cloud_api_key
        elif "cloud_api_key" not in yaml_args.__dict__:
            yaml_args.cloud_api_key = None

        if args.cloud_url is not None:
            yaml_args.cloud_url = args.cloud_url
        elif "cloud_url" not in yaml_args.__dict__:
            yaml_args.cloud_url = None

        if args.cloud_gpu_type is not None:
            yaml_args.cloud_gpu_type = args.cloud_gpu_type
        elif "cloud_gpu_type" not in yaml_args.__dict__:
            yaml_args.cloud_gpu_type = None

        if args.cloud_timeout_seconds is not None:
            yaml_args.cloud_timeout_seconds = args.cloud_timeout_seconds
        elif "cloud_timeout_seconds" not in yaml_args.__dict__:
            yaml_args.cloud_timeout_seconds = None

        if args.cloud_poll_interval is not None:
            yaml_args.cloud_poll_interval = args.cloud_poll_interval
        elif "cloud_poll_interval" not in yaml_args.__dict__:
            yaml_args.cloud_poll_interval = None

        if args.cloud_artifacts_dir is not None:
            yaml_args.cloud_artifacts_dir = args.cloud_artifacts_dir
        elif "cloud_artifacts_dir" not in yaml_args.__dict__:
            yaml_args.cloud_artifacts_dir = None

        yaml_args.use_wandb_hyperopt = _merge_store_true(args.use_wandb_hyperopt, "use_wandb_hyperopt")

        if (args.wandb_project != "Protify") or ("wandb_project" not in yaml_args.__dict__):
            yaml_args.wandb_project = args.wandb_project
        if (args.wandb_entity is not None) or ("wandb_entity" not in yaml_args.__dict__):
            yaml_args.wandb_entity = args.wandb_entity
        if (args.sweep_config_path != "yamls/sweep.yaml") or ("sweep_config_path" not in yaml_args.__dict__):
            yaml_args.sweep_config_path = args.sweep_config_path
        if (args.sweep_count != 10) or ("sweep_count" not in yaml_args.__dict__):
            yaml_args.sweep_count = args.sweep_count
        if (args.sweep_method != "bayes") or ("sweep_method" not in yaml_args.__dict__):
            yaml_args.sweep_method = args.sweep_method
        if (args.sweep_metric_cls != "eval_loss") or ("sweep_metric_cls" not in yaml_args.__dict__):
            yaml_args.sweep_metric_cls = args.sweep_metric_cls
        if (args.sweep_metric_reg != "eval_loss") or ("sweep_metric_reg" not in yaml_args.__dict__):
            yaml_args.sweep_metric_reg = args.sweep_metric_reg
        if (args.sweep_goal != "minimize") or ("sweep_goal" not in yaml_args.__dict__):
            yaml_args.sweep_goal = args.sweep_goal
        yaml_args.yaml_path = args.yaml_path
        yaml_args.aa_to_dna = _merge_store_true(args.aa_to_dna, "aa_to_dna")
        yaml_args.aa_to_rna = _merge_store_true(args.aa_to_rna, "aa_to_rna")
        yaml_args.dna_to_aa = _merge_store_true(args.dna_to_aa, "dna_to_aa")
        yaml_args.rna_to_aa = _merge_store_true(args.rna_to_aa, "rna_to_aa")
        yaml_args.codon_to_aa = _merge_store_true(args.codon_to_aa, "codon_to_aa")
        yaml_args.aa_to_codon = _merge_store_true(args.aa_to_codon, "aa_to_codon")
        yaml_args.random_pair_flipping = _merge_store_true(args.random_pair_flipping, "random_pair_flipping")
        yaml_args.push_raw_probe = _merge_store_true(args.push_raw_probe, "push_raw_probe")
        yaml_args.parallel_probe_runs = _merge_store_true(args.parallel_probe_runs, "parallel_probe_runs")
        if (args.parallel_probe_batch_mode != "shared") or ("parallel_probe_batch_mode" not in yaml_args.__dict__):
            yaml_args.parallel_probe_batch_mode = args.parallel_probe_batch_mode
        if (args.parallel_probe_index_strategy != "permutation") or ("parallel_probe_index_strategy" not in yaml_args.__dict__):
            yaml_args.parallel_probe_index_strategy = args.parallel_probe_index_strategy
        if (args.parallel_probe_max_group_size is not None) or ("parallel_probe_max_group_size" not in yaml_args.__dict__):
            yaml_args.parallel_probe_max_group_size = args.parallel_probe_max_group_size
        if (
                (args.parallel_probe_training_state_budget_gb is not None)
                or ("parallel_probe_training_state_budget_gb" not in yaml_args.__dict__)
            ):
            yaml_args.parallel_probe_training_state_budget_gb = args.parallel_probe_training_state_budget_gb
        if (
                (args.parallel_probe_estimated_peak_budget_gb is not None)
                or ("parallel_probe_estimated_peak_budget_gb" not in yaml_args.__dict__)
            ):
            yaml_args.parallel_probe_estimated_peak_budget_gb = args.parallel_probe_estimated_peak_budget_gb
        if (
                (args.parallel_probe_max_grad_norm != 0.0)
                or ("parallel_probe_max_grad_norm" not in yaml_args.__dict__)
            ):
            yaml_args.parallel_probe_max_grad_norm = args.parallel_probe_max_grad_norm
        if (
                (args.parallel_probe_grad_clip_mode != "global")
                or ("parallel_probe_grad_clip_mode" not in yaml_args.__dict__)
            ):
            yaml_args.parallel_probe_grad_clip_mode = args.parallel_probe_grad_clip_mode
        if (
                (args.parallel_probe_ensemble_average_mode != "logits")
                or ("parallel_probe_ensemble_average_mode" not in yaml_args.__dict__)
            ):
            yaml_args.parallel_probe_ensemble_average_mode = args.parallel_probe_ensemble_average_mode
        # Ensure ProteinGym defaults exist when using YAML configs
        if not hasattr(yaml_args, 'proteingym'):
            yaml_args.proteingym = False
        if not hasattr(yaml_args, 'dms_ids'):
            yaml_args.dms_ids = ["all"]
        if not hasattr(yaml_args, 'mode'):
            yaml_args.mode = None
        if not hasattr(yaml_args, 'scoring_method'):
            yaml_args.scoring_method = "masked_marginal"
        # Ensure num_runs default exists
        if not hasattr(yaml_args, 'num_runs'):
            yaml_args.num_runs = 1
        if "model_dtype" not in yaml_args.__dict__ or yaml_args.model_dtype is None:
            yaml_args.model_dtype = args.model_dtype
        if "embed_dtype" not in yaml_args.__dict__:
            yaml_args.embed_dtype = args.embed_dtype
        explicit_hidden_state_index = any(
            token == "--embedding_hidden_state_index"
            or token.startswith("--embedding_hidden_state_index=")
            for token in raw_argv
        )
        if explicit_hidden_state_index or "embedding_hidden_state_index" not in yaml_args.__dict__:
            yaml_args.embedding_hidden_state_index = args.embedding_hidden_state_index
        if "--no_embedding_scaler" in raw_argv:
            yaml_args.embedding_scaler = False
        elif "embedding_scaler" not in yaml_args.__dict__:
            yaml_args.embedding_scaler = args.embedding_scaler
        for key in [
            "probe_lr",
            "base_lr",
            "lr_scheduler",
            "optimizer",
            "base_num_epochs",
            "base_patience",
        ]:
            _merge_if_cli_explicit(key)
        if "model_paths" not in yaml_args.__dict__:
            yaml_args.model_paths = args.model_paths
        if "model_types" not in yaml_args.__dict__:
            yaml_args.model_types = args.model_types
        if "model_names" not in yaml_args.__dict__:
            yaml_args.model_names = args.model_names
        # A YAML file names the settings a run cares about, not every argument. Backfill
        # the rest from the parser so adding an argument does not break existing configs.
        for key, value in args.__dict__.items():
            if key not in yaml_args.__dict__:
                setattr(yaml_args, key, value)
        return yaml_args
    else:
        return args


if __name__ == "__main__":
    # Settings that need to happen pre-imports
    args = parse_arguments()

    # Require that either datasets are specified or a ProteinGym experiment is chosen
    has_datasets = bool(args.data_names or args.data_dirs)
    has_proteingym = bool(args.proteingym)
    if not has_datasets and not has_proteingym and args.yaml_path is None:
        raise AssertionError("No datasets specified. Provide --data_names or --data_dirs, or run a ProteinGym experiment.")

    if args.use_xformers:
        os.environ["_USE_XFORMERS"] = "1"
        print("xformers memory efficient attention enabled for AMPLIFY models")

    if args.hf_home is not None:
        # Needs to happen before any HF imports
        import pathlib
        base_path = args.hf_home
        cache_root = f"{base_path}/hf_cache"
        tmp_root   = f"{base_path}/tmp"
        pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(tmp_root).mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"]            = cache_root
        os.environ["HF_DATASETS_CACHE"]  = f"{cache_root}/datasets"
        os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/transformers" # this is deprecated, but does not hurt anything
        os.environ["HF_HUB_CACHE"]       = f"{cache_root}/hub"
        print(f"HF_HOME: {os.environ['HF_HOME']}")
        print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
        print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
        print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")

    # Set global seed before doing anything else
    # If seed is None, set_global_seed will derive it from current time
    if args.deterministic:
        from protify.seed_utils import set_determinism
        set_determinism()
    
    import protify.entrypoint_setup # needs to happen after set_determinism()


import torch
from torchinfo import summary

from protify.base_models.get_base_models import BaseModelArguments, get_base_model_for_training, get_tokenizer
from protify.base_models.utils import wrap_lora
from protify.benchmarks.proteingym.compare_scoring_methods import compare_scoring_methods
from protify.benchmarks.proteingym.scorer import ProteinGymRunner
from protify.data.data_mixin import DataArguments, DataMixin
from protify.embedder import Embedder, EmbeddingArguments, get_embedding_filename
from protify.hyperopt_utils import HyperoptModule
from protify.logger import MetricsLogger, log_method_calls
from protify.probes.estimator_probe import EstimatorProbeArguments, train_estimator_probe
from protify.probes.get_probe import ProbeArguments, get_probe
from protify.probes.scikit_classes import ScikitArguments, ScikitProbe
from protify.probes.trainers import TrainerArguments, TrainerMixin
from protify.seed_utils import set_global_seed
from protify.utils import expand_dms_ids_all, print_message, torch_load
from protify.visualization.plot_result import create_plots


class MainProcess(MetricsLogger, DataMixin, TrainerMixin):
    def __init__(self, full_args, GUI=False):
        super(MainProcess, self).__init__(full_args)
        super(DataMixin, self).__init__()
        super(TrainerMixin, self).__init__()
        self.full_args = full_args
        if not GUI:
            self.start_log_main()

        self.dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            #"int8": torch.int8,
        }

    def _build_scikit_args(self):
        if "scikit_n_iter" in self.full_args.__dict__:
            n_iter = self.full_args.scikit_n_iter
        else:
            n_iter = 10

        if "scikit_cv" in self.full_args.__dict__:
            cv = self.full_args.scikit_cv
        else:
            cv = 3

        if "scikit_random_state" in self.full_args.__dict__:
            random_state = self.full_args.scikit_random_state
        else:
            random_state = None

        if "scikit_model_name" in self.full_args.__dict__:
            model_name = self.full_args.scikit_model_name
        else:
            model_name = None

        if "production_model" in self.full_args.__dict__:
            production_model = self.full_args.production_model
        else:
            production_model = False

        if "n_jobs" in self.full_args.__dict__:
            n_jobs = self.full_args.n_jobs
        else:
            n_jobs = 1

        return ScikitArguments(
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            model_name=model_name,
            production_model=production_model,
            n_jobs=n_jobs,
        )

    @log_method_calls
    def apply_current_settings(self):
        if "model_dtype" not in self.full_args.__dict__:
            self.full_args.model_dtype = "bf16"
        if "embed_dtype" not in self.full_args.__dict__:
            self.full_args.embed_dtype = None
        if isinstance(self.full_args.model_dtype, str):
            self.full_args.model_dtype = self.dtype_map[self.full_args.model_dtype]
        if self.full_args.embed_dtype is None:
            self.full_args.embed_dtype = self.full_args.model_dtype
        elif isinstance(self.full_args.embed_dtype, str):
            self.full_args.embed_dtype = self.dtype_map[self.full_args.embed_dtype]
        else:
            self.full_args.embed_dtype = self.full_args.embed_dtype
        if "torch_compile" not in self.full_args.__dict__:
            no_compile = getattr(self.full_args, "no_compile", False)
            self.full_args.torch_compile = not no_compile
        self.data_args = DataArguments(**self.full_args.__dict__)
        self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)
        self.model_args = BaseModelArguments(**self.full_args.__dict__)
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        self.trainer_args = TrainerArguments(**self.full_args.__dict__)
        self.logger_args = SimpleNamespace(**self.full_args.__dict__)
        self.scikit_args = self._build_scikit_args()
        self._sql = self.full_args.sql
        self._full = self.full_args.matrix_embed
        self._max_length = self.full_args.max_length
        self._trim = self.full_args.trim
        self._delimiter = self.full_args.delimiter
        self._col_names = self.full_args.col_names
        self._aa_to_dna = self.full_args.aa_to_dna
        self._aa_to_rna = self.full_args.aa_to_rna
        self._dna_to_aa = self.full_args.dna_to_aa
        self._rna_to_aa = self.full_args.rna_to_aa
        self._codon_to_aa = self.full_args.codon_to_aa
        self._aa_to_codon = self.full_args.aa_to_codon
        self._multi_column = getattr(self.full_args, 'multi_column', None)

    @log_method_calls
    def get_datasets(self):
        self.datasets, self.all_seqs = self.get_data()

    @log_method_calls
    def save_embeddings_to_disk(self):
        self.embedding_args.save_embeddings = True
        embedder = Embedder(self.embedding_args, self.all_seqs)
        for display_name, dispatch_type, model_path in self.model_args.model_entries():
            _ = embedder(display_name, model_type=dispatch_type, model_path=model_path)

    def _create_model_factory(self, model_name, tokenwise, num_labels, hybrid, model_path=None):
        """Function for creating fresh models in multi-run mode."""
        def factory():
            model, _ = get_base_model_for_training(
                model_name,
                tokenwise=tokenwise,
                num_labels=num_labels,
                hybrid=hybrid,
                dtype=self.model_args.model_dtype,
                model_path=model_path,
            )
            if self.probe_args.lora:
                model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
            return model
        return factory
    
    def _create_probe_factory(self):
        """Function for creating fresh probes in multi-run mode."""
        def factory():
            return get_probe(self.probe_args)
        return factory

    def _run_nn_probe(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            tokenizer,
            emb_dict=None,
            ppi=False,
            source_model_name=None,
            sweep_mode: bool = False,
        ):
        if source_model_name is None:
            source_model_name = model_name
        # Create initial probe (for single run or as template for multi-run)
        probe = get_probe(self.probe_args)
        summary(probe)
        
        # trainer_probe handles multi-run internally if num_runs > 1
        probe, valid_metrics, test_metrics, _, _ = self.trainer_probe(
            model=probe,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            emb_dict=emb_dict,
            ppi=ppi,
            log_id=self.random_id,
            source_model_name=source_model_name,
        )
        if not sweep_mode:
            self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
            self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return probe, valid_metrics, test_metrics

    def _train_nn_probe_fold(self, model_name, dms_id, subtrain_seqs, subtrain_labels,
                            valid_seqs, valid_labels, test_seqs, test_labels, 
                            emb_dict, fold_info):
        """Trains a neural network probe on a ProteinGym DMS assay CV fold."""

        train_set = {'seqs': subtrain_seqs, 'labels': subtrain_labels}
        valid_set = None if (valid_seqs is None or valid_labels is None) else {'seqs': valid_seqs, 'labels': valid_labels}
        test_set = {'seqs': test_seqs, 'labels': test_labels}
        
        # Get tokenizer and determine input dimensions
        tokenizer = get_tokenizer(model_name)
        pooling_types = self.embedding_args.pooling_types
        hidden_state_index = self.embedding_args.hidden_state_index
        
        if self._sql:
            filename = get_embedding_filename(model_name, self._full, pooling_types, 'db', hidden_state_index)
            save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
            input_dim = self.get_embedding_dim_sql(save_path, subtrain_seqs[0], tokenizer)
            emb_for_training = None
        else:
            filename = get_embedding_filename(model_name, self._full, pooling_types, 'pth', hidden_state_index)
            save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
            emb_for_training = torch_load(save_path) if os.path.exists(save_path) else emb_dict
            input_dim = self.get_embedding_dim_pth(emb_for_training, subtrain_seqs[0], tokenizer)
        
        # Configure probe for regression
        self.probe_args.input_size = input_dim
        self.probe_args.task_type = 'regression'
        self.probe_args.num_labels = 1
        self.trainer_args.task_type = 'regression'
        
        probe = get_probe(self.probe_args)
        _, _, test_metrics = self.trainer_probe(
            model=probe,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=f"{dms_id}_{fold_info}",
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            emb_dict=emb_for_training,
            ppi=False,
            log_id=f"{self.random_id}_{fold_info}",
            source_model_name=model_name,
        )
        
        # Handle both plain and test-prefixed metric keys returned by HF Trainer
        rho = test_metrics.get('spearman_rho', test_metrics.get('test_spearman_rho', None))
        mse = test_metrics.get('mse', test_metrics.get('test_mse', None))
        return rho, mse
    
    def _run_full_finetuning(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            ppi=False,
            source_model_name=None,
            sweep_mode: bool = False,
            model_path: str = None,
        ):
        if source_model_name is None:
            source_model_name = model_name
        tokenwise = self.probe_args.tokenwise
        num_labels = self.probe_args.num_labels
        num_runs = getattr(self.trainer_args, 'num_runs', 1)
        
        model_factory = self._create_model_factory(model_name, tokenwise, num_labels, hybrid=False, model_path=model_path) if num_runs > 1 else None
        model, tokenizer = get_base_model_for_training(
            model_name,
            tokenwise=tokenwise,
            num_labels=num_labels,
            hybrid=False,
            dtype=self.model_args.model_dtype,
            model_path=model_path,
        )
        if self.probe_args.lora:
            model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
        summary(model)
        model, valid_metrics, test_metrics, _, _ = self.trainer_base_model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            ppi=ppi,
            log_id=self.random_id,
            source_model_name=source_model_name,
            model_factory=model_factory,
        )
        if not sweep_mode:
            self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
            self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return model, valid_metrics, test_metrics

    def _run_hybrid_probe(
            self,
            model_name,
            data_name,
            train_set,
            valid_set,
            test_set,
            tokenizer,
            emb_dict=None,
            ppi=False,
            source_model_name=None,
            sweep_mode: bool = False,
            model_path: str = None,
        ):
        if source_model_name is None:
            source_model_name = model_name
        # Random models don't have a trainable base model, so fall back to regular probe
        if "random" in model_name.lower():
            print_message(f"Model {model_name} does not support hybrid training. Training a linear probe instead.")
            probe = get_probe(self.probe_args)
            summary(probe)
            probe, valid_metrics, test_metrics = self.trainer_probe(
                model=probe,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_set,
                valid_dataset=valid_set,
                test_dataset=test_set,
                emb_dict=emb_dict,
                ppi=ppi,
                log_id=self.random_id,
                source_model_name=source_model_name,
            )
            if not sweep_mode:
                self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
                self.log_metrics(data_name, model_name, test_metrics, split_name='test')
            return probe, valid_metrics, test_metrics
        
        tokenwise = self.probe_args.tokenwise
        num_labels = self.probe_args.num_labels
        num_runs = getattr(self.trainer_args, 'num_runs', 1)
        
        model_factory = self._create_model_factory(model_name, tokenwise, num_labels, hybrid=True, model_path=model_path) if num_runs > 1 else None
        probe_factory = self._create_probe_factory() if num_runs > 1 else None
        model, tokenizer = get_base_model_for_training(
            model_name,
            tokenwise=tokenwise,
            num_labels=num_labels,
            hybrid=True,
            dtype=self.model_args.model_dtype,
            model_path=model_path,
        )
        if self.probe_args.lora:
            model = wrap_lora(model, self.probe_args.lora_r, self.probe_args.lora_alpha, self.probe_args.lora_dropout)
        probe = get_probe(self.probe_args)
        summary(model)
        summary(probe)
        model, valid_metrics, test_metrics, _, _ = self.trainer_hybrid_model(
            model=model,
            tokenizer=tokenizer,
            probe=probe,
            model_name=model_name,
            data_name=data_name,
            train_dataset=train_set,
            valid_dataset=valid_set,
            test_dataset=test_set,
            emb_dict=emb_dict,
            ppi=ppi,
            log_id=self.random_id,
            source_model_name=source_model_name,
            model_factory=model_factory,
            probe_factory=probe_factory,
        )
        if not sweep_mode:
            self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
            self.log_metrics(data_name, model_name, test_metrics, split_name='test')
        return model, valid_metrics, test_metrics

    @log_method_calls
    def run_full_finetuning(self):
        total_combinations = len(self.model_args.model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        for display_name, dispatch_type, model_path in self.model_args.model_entries():
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {display_name}')
                _ = self._run_full_finetuning(dispatch_type, data_name, train_set, valid_set, test_set, ppi, model_path=model_path)
                torch.cuda.empty_cache()

    @log_method_calls
    def run_hybrid_probes(self):
        probe_args = self.probe_args
        test_seq = self.all_seqs[0]

        # Log the combinations we're going to process
        total_combinations = len(self.model_args.model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        
        # for each model, gather the settings and embeddings
        # assumes save_embeddings_to_disk has already been called
        for display_name, dispatch_type, model_path in self.model_args.model_entries():
            self.logger.info(f"Processing model: {display_name}")
    
            # get tokenizer
            tokenizer = get_tokenizer(dispatch_type, model_path=model_path)

            # get embedding size
            pooling_types = self.embedding_args.pooling_types
            hidden_state_index = self.embedding_args.hidden_state_index
            if self._sql:
                # for sql, the embeddings will be gathered in real time during training
                filename = get_embedding_filename(display_name, self._full, pooling_types, 'db', hidden_state_index)
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                input_size = self.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                emb_dict = None
            else:
                # for pth, the embeddings are loaded entirely into RAM and accessed during training
                filename = get_embedding_filename(display_name, self._full, pooling_types, 'pth', hidden_state_index)
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                emb_dict = torch_load(save_path)
                input_size = self.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)

            # Adjust input dim for multi-column vector embeddings
            if (not self._full) and getattr(self.full_args, 'multi_column', None):
                input_size = input_size * len(self.full_args.multi_column)

            # for each dataset, gather the settings and train the probe
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                if ppi and not self._full:
                    probe_args.input_size = input_size * 2
                else:
                    probe_args.input_size = input_size
                # PPI concatenates two proteins along the sequence dim, so matrix-mode
                # PPI sequences can reach up to 2 * max_length.
                probe_args.max_seq_len = self._max_length * 2 if (ppi and self._full) else self._max_length

                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {display_name}')
                _ = self._run_hybrid_probe(
                    model_name=dispatch_type,
                    data_name=data_name,
                    train_set=train_set,
                    valid_set=valid_set,
                    test_set=test_set,
                    tokenizer=tokenizer,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    source_model_name=display_name,
                    model_path=model_path,
                )
                torch.cuda.empty_cache()

    @log_method_calls
    def run_nn_probes(self):
        probe_args = self.probe_args
        test_seq = self.all_seqs[0]

        # Log the combinations we're going to process
        total_combinations = len(self.model_args.model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        
        # for each model, gather the settings and embeddings
        # assumes save_embeddings_to_disk has already been called
        for display_name, dispatch_type, model_path in self.model_args.model_entries():
            self.logger.info(f"Processing model: {display_name}")
    
            # get tokenizer
            tokenizer = get_tokenizer(dispatch_type, model_path=model_path)

            # get embedding size
            pooling_types = self.embedding_args.pooling_types
            hidden_state_index = self.embedding_args.hidden_state_index
            if self._sql:
                # for sql, the embeddings will be gathered in real time during training
                filename = get_embedding_filename(display_name, self._full, pooling_types, 'db', hidden_state_index)
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                input_size = self.get_embedding_dim_sql(save_path, test_seq, tokenizer)
                emb_dict = None
            else:
                # for pth, the embeddings are loaded entirely into RAM and accessed during training
                filename = get_embedding_filename(display_name, self._full, pooling_types, 'pth', hidden_state_index)
                save_path = os.path.join(self.embedding_args.embedding_save_dir, filename)
                emb_dict = torch_load(save_path)
                input_size = self.get_embedding_dim_pth(emb_dict, test_seq, tokenizer)

            # Adjust input dim for multi-column vector embeddings
            if (not self._full) and getattr(self.full_args, 'multi_column', None):
                input_size = input_size * len(self.full_args.multi_column)

            print(f'Input dim: {input_size}')

            # for each dataset, gather the settings and train the probe
            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                if ppi and not self._full:
                    probe_args.input_size = input_size * 2
                else:
                    probe_args.input_size = input_size
                # PPI concatenates two proteins along the sequence dim, so matrix-mode
                # PPI sequences can reach up to 2 * max_length.
                probe_args.max_seq_len = self._max_length * 2 if (ppi and self._full) else self._max_length

                self.probe_args.num_labels = num_labels
                self.probe_args.task_type = label_type
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {display_name}')
                _ = self._run_nn_probe(
                    model_name=display_name,
                    data_name=data_name,
                    train_set=train_set,
                    valid_set=valid_set,
                    test_set=test_set,
                    tokenizer=tokenizer,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    source_model_name=display_name,
                )
                torch.cuda.empty_cache()

    @log_method_calls
    def run_estimator_probes(self):
        """Fit gradient-boosted or forest probes on pooled embeddings.

        Shares the embedding cache, results table, and plots with the neural probes, so
        --probe_type xgboost differs from --probe_type mlp only in the learner.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for display_name, dispatch_type, model_path in self.model_args.model_entries():
            for data_name, dataset in self.datasets.items():
                num_labels = dataset[3]
                X_train, y_train, X_valid, y_valid, X_test, y_test, label_type = self.prepare_scikit_dataset(
                    display_name,
                    dataset,
                    standardize=False,
                    flip_train_pairs=self.full_args.random_pair_flipping,
                )
                check_task_type(self.probe_args.probe_type, label_type)
                if label_type == 'multilabel':
                    # prepare_scikit_dataset flattens list-valued labels; estimators need
                    # one row of targets per sample.
                    y_train = y_train.reshape(-1, num_labels)
                    y_valid = y_valid.reshape(-1, num_labels)
                    y_test = y_test.reshape(-1, num_labels)
                self.logger.info(f'Training {self.probe_args.probe_type} probe for {data_name} with {display_name}')

                valid_runs, test_runs = [], []
                for run_index in range(self.trainer_args.num_runs):
                    estimator_args = EstimatorProbeArguments(
                        probe_type=self.probe_args.probe_type,
                        task_type=label_type,
                        num_labels=num_labels,
                        seed=self.full_args.seed + run_index,
                        n_jobs=self.full_args.probe_n_jobs,
                        device=device,
                        overrides=self._estimator_overrides(),
                    )
                    _, valid_metrics, test_metrics = train_estimator_probe(
                        estimator_args, X_train, y_train, X_valid, y_valid, X_test, y_test,
                    )
                    valid_runs.append(valid_metrics)
                    test_runs.append(test_metrics)

                # A single run reports raw metrics; several report mean and standard
                # deviation, matching how the neural probes report multi-run results.
                if len(valid_runs) == 1:
                    valid_metrics, test_metrics = valid_runs[0], test_runs[0]
                else:
                    valid_metrics = self._aggregate_metrics(valid_runs)
                    test_metrics = self._aggregate_metrics(test_runs)
                self.log_metrics(data_name, display_name, valid_metrics, split_name='valid')
                self.log_metrics(data_name, display_name, test_metrics, split_name='test')

    def _estimator_overrides(self):
        """Estimator hyperparameters supplied through --scikit_model_args JSON."""
        raw = getattr(self.full_args, 'scikit_model_args', None)
        if not raw:
            return {}

        overrides = json.loads(raw)
        print_message(f'Overriding {self.probe_args.probe_type} hyperparameters: {overrides}')
        return overrides

    @log_method_calls
    def run_scikit_scheme(self):
        self.scikit_args = self._build_scikit_args()
        scikit_probe = ScikitProbe(self.scikit_args)
        for display_name, dispatch_type, model_path in self.model_args.model_entries():
            for data_name, dataset in self.datasets.items():
                ### find best scikit model and parameters via cross validation and lazy predict
                X_train, y_train, X_valid, y_valid, X_test, y_test, label_type = self.prepare_scikit_dataset(
                    display_name, dataset, flip_train_pairs=self.full_args.random_pair_flipping
                )

                # If a specific model is specified, skip LazyPredict and go straight to that model
                if self.scikit_args.model_name is not None:
                    print_message(f"Skipping LazyPredict, using specified model: {self.scikit_args.model_name}")
                    results = scikit_probe.run_specific_model(X_train, y_train, X_valid, y_valid, X_test, y_test, model_results=None)
                else:
                    # Find best model via LazyPredict
                    if label_type == 'singlelabel':
                        results = scikit_probe.find_best_classifier(X_train, y_train, X_valid, y_valid)
                    elif label_type == 'regression':
                        results = scikit_probe.find_best_regressor(X_train, y_train, X_valid, y_valid)
                    else:
                        raise ValueError(f'Label type {label_type} not supported')
                    # Train and evaluate best model with optimal hyperparameters
                    results = scikit_probe.run_specific_model(X_train, y_train, X_valid, y_valid, X_test, y_test, results)
                
                # Log the results for plotting
                metrics_dict = {'test_mcc': results.final_scores} if isinstance(results.final_scores, (int, float)) else results.final_scores
                self.log_metrics(data_name, display_name, metrics_dict, split_name='test')
    
    @log_method_calls
    def generate_plots(self):
        print_message("Generating visualization plots...")
        # Determine which results file to use
        results_file = os.path.join(self.full_args.results_dir, f"{self.random_id}.tsv")
        
        # Check if the results file exists
        if not os.path.exists(results_file):
            print_message(f"Results file not found: {results_file}")
            return
        
        # Get output directory
        output_dir = self.full_args.plots_dir

        print_message(f"Generating plots in {output_dir}...")
        create_plots(results_file, output_dir)
        print_message("Plots generated successfully!")
        
    def run_proteingym_zero_shot(self):
        """Run ProteinGym zero-shot for all specified models and DMS ids."""
        dms_ids = getattr(self.full_args, 'dms_ids', []) or []
        mode = getattr(self.full_args, 'mode', 'benchmark')
        dms_ids = expand_dms_ids_all(dms_ids, mode=mode)
        if len(dms_ids) == 0:
            raise ValueError("--dms_ids is required when --proteingym is specified")
        model_names = self.model_args.model_names
        if len(model_names) == 0:
            raise ValueError("--model_names must specify at least one model")
        assert self.model_args._model_paths is None, "ProteinGym zero-shot requires --model_names (preset names), not --model_paths/--model_types."
        # Where to write results
        results_root = getattr(self.full_args, 'results_dir', 'results')
        results_dir = os.path.join(results_root, 'proteingym')
        scoring_method = getattr(self.full_args, 'scoring_method', 'masked_marginal')
        scoring_window = getattr(self.full_args, 'scoring_window', 'optimal')
        if isinstance(mode, str) and mode.lower() == 'indels':
            print_message("Only pll is currently supported for indels scoring.")
            scoring_method = 'pll'
        
        # Log the run
        self.logger.info(f"Running ProteinGym zero-shot with [{scoring_method}] scoring on {len(dms_ids)} DMS ids with models: {model_names}")
        
        runner = ProteinGymRunner(
            results_dir=results_dir,
            repo_id="GleghornLab/ProteinGym_DMS",
        )
        self._proteingym_timing = runner.run(
            dms_ids=dms_ids,
            model_names=model_names,
            mode=mode,
            scoring_method=scoring_method,
            scoring_window=scoring_window,
            batch_size=getattr(self.full_args, 'pg_batch_size', 32),
        )
        print_message(f"ProteinGym zero-shot complete. Results in {results_dir}")

        # After all models are scored, run standardized performance benchmarking
        runner.run_benchmark(model_names, dms_ids, mode, scoring_method)

def main(args: SimpleNamespace):
    chosen_seed = set_global_seed(args.seed)
    args.seed = chosen_seed

    if not hasattr(args, 'padding'):
        args.padding = 'max_length'

    if args.padding == "longest":
        print_message(
            "WARNING: padding='longest' causes dynamic tensor shapes. "
            "This disables torch.compile graph caching and reduces flex attention efficiency. "
            "Use padding='max_length' (default) for best performance."
        )

    if _should_auto_run_cloud(args):
        return _run_on_cloud(args)

    if args.replay_path is not None:
        from logger import LogReplayer
        replayer = LogReplayer(args.replay_path)
        replay_args = replayer.parse_log()
        replay_args.replay_path = args.replay_path
        # Re-apply seed using the replayed settings to ensure exact reproducibility
        try:
            # If no seed is present in replay, fall back to time-based seed
            if not hasattr(replay_args, 'seed') or replay_args.seed is None:
                replay_args.seed = None
            if not hasattr(replay_args, 'deterministic') or replay_args.deterministic is None:
                replay_args.deterministic = getattr(args, 'deterministic', False)
            chosen_seed = set_global_seed(replay_args.seed, deterministic=replay_args.deterministic)
            replay_args.seed = chosen_seed
        except Exception:
            pass
        main = MainProcess(replay_args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")
        replayer.run_replay(main)
    
    else:
        main = MainProcess(args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")

        if getattr(args, 'compare_scoring_methods', False) and getattr(args, 'proteingym', False):
            # Run scoring method comparison
            print_message("Running scoring method comparison...")
            dms_ids = getattr(args, 'dms_ids', []) or []
            mode = getattr(args, 'mode', 'benchmark')
            dms_ids = expand_dms_ids_all(dms_ids, mode=mode)
            model_names = getattr(args, 'model_names', []) or []
            if len(model_names) == 0:
                raise ValueError("--model_names must specify at least one model")
            
            # Set up output path
            results_root = getattr(args, 'results_dir', 'results')
            output_csv = os.path.join(results_root, 'scoring_methods_comparison.csv')
            
            summary_df = compare_scoring_methods(
                model_names=model_names,
                device=None,
                methods=None,
                dms_ids=dms_ids,
                progress=True,
                output_csv=output_csv
            )
            print_message(f"Scoring method comparison complete. Results saved to {output_csv}")
            return

        # Determine if current experiment passed datasets
        has_datasets = bool(getattr(args, 'data_names', []) or getattr(args, 'data_dirs', []))

        # Run through datasets first (if any)
        if has_datasets:
          main.apply_current_settings()
          main.get_datasets()
          print_message(f"Number of sequences: {len(main.all_seqs)}")
          if main.full_args.use_wandb_hyperopt:
              if not main.full_args.full_finetuning:
                  main.save_embeddings_to_disk()
              HyperoptModule.run_wandb_hyperopt(main)

          elif main.full_args.full_finetuning:
              main.run_full_finetuning()

          elif main.full_args.hybrid_probe:
              main.save_embeddings_to_disk()
              main.run_hybrid_probes()

          elif main.full_args.use_scikit:
              main.save_embeddings_to_disk()
              main.run_scikit_scheme()
          else:
              main.save_embeddings_to_disk()
              if os.environ.get('_PROTIFY_EMBED_PHASE') == '1':
                  print_message("Embed phase complete, proceeding to merge and train.")
                  return
              if is_estimator_probe(main.full_args.probe_type):
                  main.run_estimator_probes()
              else:
                  main.run_nn_probes()
        else:
            # Determine if current experiment passed datasets
            has_datasets = bool(getattr(args, 'data_names', []) or getattr(args, 'data_dirs', []))

            # Run through datasets first (if any)
            if has_datasets:
                main.apply_current_settings()
                main.get_datasets()
                num_seqs = len(main.all_seqs) if hasattr(main, 'all_seqs') else 0
                print_message(f"Number of sequences: {num_seqs}")

                if main.full_args.full_finetuning:
                    main.run_full_finetuning()

                elif main.full_args.hybrid_probe:
                    main.save_embeddings_to_disk()
                    main.run_hybrid_probes()

                elif main.full_args.use_scikit:
                    main.save_embeddings_to_disk()
                    main.run_scikit_scheme()
                
                else:
                    main.save_embeddings_to_disk()
                    if os.environ.get('_PROTIFY_EMBED_PHASE') == '1':
                        print_message("Embed phase complete, proceeding to merge and train.")
                        return
                    if is_estimator_probe(main.full_args.probe_type):
                        main.run_estimator_probes()
                    else:
                        main.run_nn_probes()
            else:
                print_message("No datasets specified; proceeding with ProteinGym.")

            if getattr(args, 'proteingym', False):
                main.run_proteingym_zero_shot()
                try:
                    results_root = getattr(args, 'results_dir', 'results')
                    results_dir = os.path.join(results_root, 'proteingym')
                    pg_scores = ProteinGymRunner.collect_spearman(results_dir, getattr(args, 'model_names', []))
                    for model_name, score in pg_scores.items():
                        if isinstance(score, (int, float)):
                            training_time = getattr(main, '_proteingym_timing', {}).get(model_name, None)
                            metrics_dict = {'spearman': float(score)}
                            metrics_dict['training_time_seconds'] = float(training_time)
                            main.log_metrics('proteingym', model_name, metrics_dict)
                except Exception as e:
                    print_message(f"Failed to log ProteinGym metrics: {e}")

        # Write results and generate plots
        main.write_results()
        main.generate_plots()
        main.end_log()
    return 0

if __name__ == "__main__":
    sys.exit(main(args))
