import torch
import torch.nn.functional as F
import os
import sqlite3
import numpy as np
from copy import deepcopy
from typing import Optional, Dict, List, Any, Union
from huggingface_hub import HfApi
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction, TrainerCallback
from dataclasses import dataclass
try:
    from probes.hybrid_probe import HybridProbe, HybridProbeConfig
    from probes.parallel_linear_probe import ParallelLinearProbe, ParallelLinearProbeConfig
    from probes.parallel_probe_batches import ParallelRunDataset
    from probes.parallel_probe_plan import (
        build_seed_run_specs,
        estimate_parallel_probe_plan,
        max_linear_probe_runs_for_estimated_peak_budget,
        max_linear_probe_runs_for_training_state_budget,
        plan_parallel_probe_runs,
    )
    from probes.export_packaged_model import export_packaged_model_to_hub
    from data.dataset_classes import (
        EmbedsLabelsDatasetFromDisk,
        PairEmbedsLabelsDatasetFromDisk,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
        StringLabelDataset,
        PairStringLabelDataset,
        MultiEmbedsLabelsDatasetFromDisk,
        MultiEmbedsLabelsDataset,
        EmbeddingStandardizer,
    )
except ImportError:
    from .hybrid_probe import HybridProbe, HybridProbeConfig
    from .parallel_linear_probe import ParallelLinearProbe, ParallelLinearProbeConfig
    from .parallel_probe_batches import ParallelRunDataset
    from .parallel_probe_plan import (
        build_seed_run_specs,
        estimate_parallel_probe_plan,
        max_linear_probe_runs_for_estimated_peak_budget,
        max_linear_probe_runs_for_training_state_budget,
        plan_parallel_probe_runs,
    )
    from .export_packaged_model import export_packaged_model_to_hub
    from ..data.dataset_classes import (
        EmbedsLabelsDatasetFromDisk,
        PairEmbedsLabelsDatasetFromDisk,
        EmbedsLabelsDataset,
        PairEmbedsLabelsDataset,
        StringLabelDataset,
        PairStringLabelDataset,
        MultiEmbedsLabelsDatasetFromDisk,
        MultiEmbedsLabelsDataset,
        EmbeddingStandardizer,
    )
try:
    from data.data_collators import (
        EmbedsLabelsCollator,
        PairEmbedsLabelsCollator,
        PairCollator_input_ids,
        StringLabelsCollator,
    )
    from visualization.ci_plots import regression_ci_plot, classification_ci_plot
    from utils import print_message
    from metrics import get_compute_metrics, get_compute_metrics_with_balanced
    from metrics_balanced import compute_balanced_regression_metrics
    from seed_utils import set_global_seed
    from probes.get_probe import get_probe
except ImportError:
    from ..data.data_collators import (
        EmbedsLabelsCollator,
        PairEmbedsLabelsCollator,
        PairCollator_input_ids,
        StringLabelsCollator,
    )
    from ..visualization.ci_plots import regression_ci_plot, classification_ci_plot
    from ..utils import print_message
    from ..metrics import get_compute_metrics, get_compute_metrics_with_balanced
    from ..metrics_balanced import compute_balanced_regression_metrics
    from ..seed_utils import set_global_seed
    from .get_probe import get_probe


def _unwrap_parallel_probe_for_gradient_clipping(model) -> ParallelLinearProbe:
    if isinstance(model, ParallelLinearProbe):
        return model
    model_dict = model.__dict__
    assert '_orig_mod' in model_dict, (
        "Expected a ParallelLinearProbe or a torch.compile wrapper with _orig_mod for per-run clipping."
    )
    original_model = model_dict['_orig_mod']
    assert isinstance(original_model, ParallelLinearProbe), (
        f"Expected _orig_mod to be ParallelLinearProbe, got {type(original_model).__name__}."
    )
    return original_model


def _clip_parallel_probe_gradients_per_run(model, max_norm: float) -> List[float]:
    assert max_norm > 0.0, "max_norm must be positive for per-run clipping."
    parallel_model = _unwrap_parallel_probe_for_gradient_clipping(model)
    first_param = next(parallel_model.parameters())
    pre_clip_norms = []
    for run_idx in range(parallel_model.num_runs):
        norm_sq = torch.zeros((), dtype=torch.float32, device=first_param.device)
        grad_slices = []
        for layer in parallel_model._parameter_layers():
            weight_grad = layer.weight.grad
            if weight_grad is not None:
                run_weight_grad = weight_grad[run_idx]
                grad_slices.append(run_weight_grad)
                norm_sq = norm_sq + run_weight_grad.detach().float().pow(2).sum()
            bias_param = layer.bias
            if bias_param is not None:
                bias_grad = bias_param.grad
                if bias_grad is not None:
                    run_bias_grad = bias_grad[run_idx]
                    grad_slices.append(run_bias_grad)
                    norm_sq = norm_sq + run_bias_grad.detach().float().pow(2).sum()

        total_norm = norm_sq.sqrt()
        pre_clip_norms.append(float(total_norm.detach().cpu().item()))
        clip_coef = torch.clamp(
            torch.as_tensor(max_norm, dtype=torch.float32, device=first_param.device) / (total_norm + 1e-6),
            max=1.0,
        )
        for grad_slice in grad_slices:
            grad_slice.mul_(clip_coef.to(dtype=grad_slice.dtype))
    return pre_clip_norms


class ParallelProbePerRunGradientClipCallback(TrainerCallback):
    def __init__(self, max_norm: float):
        assert max_norm > 0.0, "parallel per-run gradient clipping requires a positive max norm."
        self.max_norm = max_norm
        self.last_pre_clip_norms = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        del args, state
        self.last_pre_clip_norms = _clip_parallel_probe_gradients_per_run(kwargs['model'], self.max_norm)
        return control


def _compute_eval_accumulation_steps(
        eval_dataset_size: int,
        batch_size: int,
    num_labels: int,
    task_type: str,
    output_multiplier: int = 1,
) -> Optional[int]:
    """Compute eval_accumulation_steps based on prediction size vs GPU memory.

    Returns None when all predictions fit comfortably on GPU (fastest path).
    Otherwise returns a step count that keeps accumulated predictions under
    5% of total GPU memory.
    """
    if task_type == 'regression':
        output_dim = 1
    elif task_type == 'binary':
        output_dim = 2
    else:
        output_dim = num_labels
    output_dim *= output_multiplier

    total_pred_bytes = eval_dataset_size * output_dim * 4  # float32

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem_bytes = props.total_memory
    else:
        gpu_mem_bytes = 8 * (1024 ** 3)  # 8 GB fallback

    budget_bytes = gpu_mem_bytes * 0.05

    if total_pred_bytes <= budget_bytes:
        return None

    bytes_per_step = batch_size * output_dim * 4
    return max(1, int(budget_bytes / bytes_per_step))


@dataclass
class TrainerArguments:
    def __init__(
            self,
            model_save_dir: str,
            num_epochs: int = 200,
            probe_batch_size: int = 64,
            base_batch_size: int = 4,
            probe_grad_accum: int = 1,
            base_grad_accum: int = 1,
            lr: float = 1e-4,
            probe_lr: Optional[float] = None,
            lr_scheduler: str = 'cosine',
            optimizer: str = 'adamw_torch',
            weight_decay: float = 0.00,
            task_type: str = 'regression',
            patience: int = 3,
            base_num_epochs: Optional[int] = None,
            base_patience: Optional[int] = None,
            base_lr: Optional[float] = None,
            read_scaler: int = 100,
            save_model: bool = False,
            push_raw_probe: bool = False,
            push_raw_probe_repo: str = None,
            seed: int = 42,
            train_data_size: int = 100,
            plots_dir: str = None,
            full_finetuning: bool = False,
            hybrid_probe: bool = False,
            num_workers: int = 0,
            make_plots: bool = True,
            num_runs: int = 1,
            torch_compile: bool = True,
            eval_accumulation_steps: Union[int, str] = "auto",
            balanced_regression_metrics: bool = True,
            balanced_weight_method: str = 'bin_inv',
            balanced_bin_borders: Optional[List[float]] = None,
            balanced_n_resamples: int = 100,
            balanced_lds_bins: int = 100,
            balanced_lds_ks: int = 5,
            balanced_lds_sigma: float = 2.0,
            parallel_probe_runs: bool = False,
            parallel_probe_batch_mode: str = 'shared',
            parallel_probe_index_strategy: str = 'permutation',
            parallel_probe_max_group_size: Optional[int] = None,
            parallel_probe_training_state_budget_gb: Optional[float] = None,
            parallel_probe_estimated_peak_budget_gb: Optional[float] = None,
            parallel_probe_max_grad_norm: float = 0.0,
            parallel_probe_grad_clip_mode: str = 'global',
            parallel_probe_ensemble_average_mode: str = 'logits',
            **kwargs
    ):
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        self.probe_batch_size = probe_batch_size
        self.base_batch_size = base_batch_size
        self.probe_grad_accum = probe_grad_accum
        self.base_grad_accum = base_grad_accum
        self.lr = lr
        self.probe_lr = probe_lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.task_type = task_type
        self.patience = patience
        self.base_num_epochs = base_num_epochs
        self.base_patience = base_patience
        self.base_lr = base_lr
        self.save = save_model
        self.push_raw_probe = push_raw_probe
        self.push_raw_probe_repo = push_raw_probe_repo
        self.read_scaler = read_scaler
        self.seed = seed
        self.train_data_size = train_data_size
        self.plots_dir = plots_dir
        self.full_finetuning = full_finetuning
        self.hybrid_probe = hybrid_probe
        self.num_workers = num_workers
        self.make_plots = make_plots
        self.num_runs = num_runs
        self.torch_compile = torch_compile
        self.eval_accumulation_steps = eval_accumulation_steps
        self.balanced_regression_metrics = balanced_regression_metrics
        self.balanced_weight_method = balanced_weight_method
        self.balanced_bin_borders = balanced_bin_borders
        self.balanced_n_resamples = balanced_n_resamples
        self.balanced_lds_bins = balanced_lds_bins
        self.balanced_lds_ks = balanced_lds_ks
        self.balanced_lds_sigma = balanced_lds_sigma
        self.parallel_probe_runs = parallel_probe_runs
        assert parallel_probe_max_grad_norm >= 0.0, "parallel_probe_max_grad_norm must be non-negative."
        self.parallel_probe_max_grad_norm = parallel_probe_max_grad_norm
        assert parallel_probe_grad_clip_mode in ('none', 'global', 'per_run'), (
            "parallel_probe_grad_clip_mode must be 'none', 'global', or 'per_run'."
        )
        self.parallel_probe_grad_clip_mode = parallel_probe_grad_clip_mode
        assert parallel_probe_ensemble_average_mode in ('logits', 'probabilities'), (
            "parallel_probe_ensemble_average_mode must be 'logits' or 'probabilities'."
        )
        self.parallel_probe_ensemble_average_mode = parallel_probe_ensemble_average_mode
        assert parallel_probe_batch_mode in ('shared', 'run_specific'), (
            "parallel_probe_batch_mode must be 'shared' or 'run_specific'."
        )
        self.parallel_probe_batch_mode = parallel_probe_batch_mode
        assert parallel_probe_index_strategy in ('permutation', 'affine'), (
            "parallel_probe_index_strategy must be 'permutation' or 'affine'."
        )
        self.parallel_probe_index_strategy = parallel_probe_index_strategy
        if parallel_probe_max_group_size is not None:
            assert parallel_probe_max_group_size > 0, "parallel_probe_max_group_size must be positive when provided."
        self.parallel_probe_max_group_size = parallel_probe_max_group_size
        if parallel_probe_training_state_budget_gb is not None:
            assert parallel_probe_training_state_budget_gb > 0.0, (
                "parallel_probe_training_state_budget_gb must be positive when provided."
            )
        self.parallel_probe_training_state_budget_gb = parallel_probe_training_state_budget_gb
        if parallel_probe_estimated_peak_budget_gb is not None:
            assert parallel_probe_estimated_peak_budget_gb > 0.0, (
                "parallel_probe_estimated_peak_budget_gb must be positive when provided."
            )
        self.parallel_probe_estimated_peak_budget_gb = parallel_probe_estimated_peak_budget_gb
        self.eval_dataset_size = 10000
        self.num_labels = 2
        self.eval_output_multiplier = 1

    def __call__(self, probe: Optional[bool] = True):
        batch_size = self.probe_batch_size if probe else self.base_batch_size
        grad_accum = self.probe_grad_accum if probe else self.base_grad_accum
        num_epochs = self.num_epochs if (probe or self.base_num_epochs is None) else self.base_num_epochs
        if probe:
            lr = self.lr if self.probe_lr is None else self.probe_lr
        else:
            lr = self.lr if self.base_lr is None else self.base_lr

        if self.train_data_size > 250000:
            eval_steps = max(1, int(100000 / (batch_size * grad_accum)))
            eval_strats = {
                'eval_strategy': 'steps',
                'eval_steps': eval_steps,
                'save_strategy': 'steps',
                'save_steps': eval_steps,
            }
        else:
            eval_strats = {
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
            }

        if '/' in self.model_save_dir:
            save_dir = self.model_save_dir.split('/')[-1]
        else:
            save_dir = self.model_save_dir

        warmup_steps = 100 if probe else 1000

        if self.eval_accumulation_steps == "auto":
            eval_accum = _compute_eval_accumulation_steps(
                eval_dataset_size=self.eval_dataset_size,
                batch_size=batch_size,
                num_labels=self.num_labels,
                task_type=self.task_type,
                output_multiplier=self.eval_output_multiplier,
            )
        else:
            eval_accum = self.eval_accumulation_steps

        return TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=float(lr),
            lr_scheduler_type=self.lr_scheduler,
            optim=self.optimizer,
            weight_decay=float(self.weight_decay),
            warmup_steps=warmup_steps,
            save_total_limit=3,
            logging_steps=1000,
            report_to='none',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            seed=self.seed,
            label_names=['labels'],
            dataloader_num_workers=self.num_workers,
            dataloader_prefetch_factor=2 if self.num_workers > 0 else None,
            # Explicitly disable mixed precision training to prevent automatic fp16 conversion
            fp16=False,
            bf16=False,
            torch_compile=self.torch_compile,
            eval_accumulation_steps=eval_accum,
            **eval_strats
        )


class TrainerMixin:
    def __init__(self, trainer_args: Optional[TrainerArguments] = None):
        self.trainer_args = trainer_args

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def _format_metrics_markdown(self, metrics: Dict[str, Any]) -> str:
        if metrics is None or len(metrics) == 0:
            return "- No metrics recorded."
        lines = []
        for key in sorted(metrics.keys()):
            lines.append(f"- `{key}`: {self._format_metric_value(metrics[key])}")
        return "\n".join(lines)

    def _build_model_card(
            self,
            repo_id: str,
            data_name: str,
            model_name: str,
            log_id: str,
            train_dataset,
            valid_dataset,
            test_dataset,
            valid_metrics: Dict[str, Any],
            test_metrics: Dict[str, Any],
        ) -> str:
        train_size = len(train_dataset)
        valid_size = "N/A" if valid_dataset is None else str(len(valid_dataset))
        test_size = len(test_dataset)
        task_type = self.trainer_args.task_type
        num_runs = self.trainer_args.num_runs
        validation_metrics_text = self._format_metrics_markdown(valid_metrics)
        test_metrics_text = self._format_metrics_markdown(test_metrics)
        return f"""---
library_name: transformers
tags: []
---

# {repo_id}

Fine-tuned with Protify.

## About Protify

Protify is an open source platform designed to simplify and democratize workflows for chemical language models. With Protify, deep learning models can be trained to predict chemical properties without requiring extensive coding knowledge or computational resources.

### Why Protify?

- Benchmark multiple models efficiently.
- Flexible for all skill levels.
- Accessible computing with support for precomputed embeddings.
- Cost-effective workflows for training and evaluation.

## Training Run

- `dataset`: {data_name}
- `model`: {model_name}
- `run_id`: {log_id}
- `task_type`: {task_type}
- `num_runs`: {num_runs}

## Dataset Statistics

- `train_size`: {train_size}
- `valid_size`: {valid_size}
- `test_size`: {test_size}

## Validation Metrics

{validation_metrics_text}

## Test Metrics

{test_metrics_text}
"""

    def _extract_prediction_array(self, predictions):
        if isinstance(predictions, tuple):
            return predictions[0]
        return predictions

    def _extract_label_array(self, label_ids):
        if isinstance(label_ids, tuple):
            return label_ids[1]
        return label_ids

    def _parallel_probe_labels_for_run(
            self,
            labels,
            run_idx: int,
            num_runs: int,
            num_labels: Optional[int] = None,
        ):
        task_type = self.trainer_args.task_type
        if num_labels is None:
            num_labels = self.probe_args.num_labels
        if task_type == 'singlelabel':
            labels_have_run_dimension = (
                labels.ndim >= 2
                and labels.shape[1] == num_runs
            )
            if labels_have_run_dimension:
                if labels.ndim == 3:
                    assert labels.shape[-1] == 1, (
                        f"Singlelabel run-specific labels must end in dim 1, got shape {tuple(labels.shape)}."
                    )
                    return labels[:, run_idx, 0]
                return labels[:, run_idx]
            return labels

        labels_have_explicit_run_dimension = (
            labels.ndim == 3
            and labels.shape[1] == num_runs
        )
        if labels_have_explicit_run_dimension:
            return labels[:, run_idx, :]

        labels_have_scalar_run_dimension = (
            labels.ndim == 2
            and labels.shape[1] == num_runs
            and num_labels == 1
        )
        if labels_have_scalar_run_dimension:
            return labels[:, run_idx]

        return labels

    def _parallel_probe_losses_by_run(
            self,
            logits: np.ndarray,
            labels: np.ndarray,
            task_type: str,
        ) -> List[float]:
        logits_t = torch.as_tensor(logits)
        labels_t = torch.as_tensor(labels)
        num_runs = logits_t.shape[1]
        losses = []
        for run_idx in range(num_runs):
            run_logits = logits_t[:, run_idx, :]
            run_labels = self._parallel_probe_labels_for_run(labels_t, run_idx, num_runs, run_logits.shape[-1])
            if task_type == 'singlelabel':
                target = run_labels.long().view(-1)
                loss = F.cross_entropy(run_logits, target)
            elif task_type == 'multilabel':
                target = run_labels.float()
                if target.ndim == 1:
                    target = target.unsqueeze(-1)
                loss = F.binary_cross_entropy_with_logits(run_logits, target)
            elif task_type == 'regression':
                target = run_labels.float()
                if target.ndim == 1:
                    target = target.unsqueeze(-1)
                loss = F.mse_loss(run_logits, target)
            elif task_type == 'sigmoid_regression':
                target = run_labels.float()
                if target.ndim == 1:
                    target = target.unsqueeze(-1)
                valid_mask = target != -100.0
                safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
                element_losses = F.binary_cross_entropy(run_logits, safe_target, reduction='none')
                denominator = valid_mask.float().sum()
                if denominator.item() == 0:
                    loss = torch.zeros((), dtype=run_logits.dtype, device=run_logits.device)
                else:
                    loss = (element_losses * valid_mask).sum() / denominator
            else:
                raise ValueError(f"Task type {task_type} not supported by parallel probe metrics.")
            losses.append(float(loss.item()))
        return losses

    def _parallel_probe_probability_averaged_predictions(
            self,
            logits: np.ndarray,
            task_type: str,
        ) -> np.ndarray:
        if task_type == 'singlelabel':
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp_logits = np.exp(shifted)
            probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            clipped = np.clip(probabilities.mean(axis=1), 1e-12, 1.0)
            return np.log(clipped).astype(np.float32)
        if task_type == 'multilabel':
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            clipped = np.clip(probabilities.mean(axis=1), 1e-12, 1.0 - 1e-12)
            return np.log(clipped / (1.0 - clipped)).astype(np.float32)
        return logits.mean(axis=1).astype(np.float32)

    def _parallel_probe_ensemble_predictions(
            self,
            logits: np.ndarray,
            task_type: str,
        ) -> np.ndarray:
        if self.trainer_args.parallel_probe_ensemble_average_mode == 'probabilities':
            return self._parallel_probe_probability_averaged_predictions(logits, task_type)
        return logits.mean(axis=1).astype(np.float32)

    def _parallel_probe_prediction_loss(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            task_type: str,
        ) -> float:
        predictions_t = torch.as_tensor(predictions)
        labels_t = torch.as_tensor(labels)
        if task_type == 'singlelabel':
            target = labels_t.long().view(-1)
            loss = F.cross_entropy(predictions_t, target)
        elif task_type == 'multilabel':
            target = labels_t.float()
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(predictions_t, target)
        elif task_type == 'regression':
            target = labels_t.float()
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            loss = F.mse_loss(predictions_t, target)
        elif task_type == 'sigmoid_regression':
            target = labels_t.float()
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            valid_mask = target != -100.0
            safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
            element_losses = F.binary_cross_entropy(predictions_t, safe_target, reduction='none')
            denominator = valid_mask.float().sum()
            if denominator.item() == 0:
                loss = torch.zeros((), dtype=predictions_t.dtype, device=predictions_t.device)
            else:
                loss = (element_losses * valid_mask).sum() / denominator
        else:
            raise ValueError(f"Task type {task_type} not supported by parallel probe prediction metrics.")
        return float(loss.item())

    def _parallel_probe_ensemble_metrics(
            self,
            logits: np.ndarray,
            labels: np.ndarray,
            split_name: str,
        ) -> Dict[str, Any]:
        task_type = self.trainer_args.task_type
        ensemble_predictions = self._parallel_probe_ensemble_predictions(logits, task_type)
        compute_metrics = get_compute_metrics(task_type, tokenwise=False)
        raw_metrics = compute_metrics(EvalPrediction(predictions=ensemble_predictions, label_ids=labels))
        prefixed_metrics = {
            f"{split_name}_{key}": value
            for key, value in raw_metrics.items()
        }
        prefixed_metrics[f"{split_name}_loss"] = self._parallel_probe_prediction_loss(
            ensemble_predictions,
            labels,
            task_type,
        )
        return prefixed_metrics

    def _seed_ensemble_metrics_from_run_results(
            self,
            run_results,
            split_name: str,
        ) -> Dict[str, Any]:
        predictions_by_run = []
        reference_labels = None
        for _run_idx, _test_loss, y_pred, y_true, _seed, _model in run_results:
            predictions = np.asarray(y_pred, dtype=np.float32)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            labels = np.asarray(y_true)
            if reference_labels is None:
                reference_labels = labels
            else:
                assert np.array_equal(reference_labels, labels), (
                    "Sequential seed ensemble labels changed between runs."
                )
            predictions_by_run.append(predictions)

        assert len(predictions_by_run) > 0, "Seed ensemble metrics require at least one run result."
        assert reference_labels is not None, "Seed ensemble metrics require labels."
        prediction_bank = np.stack(predictions_by_run, axis=1)
        return self._parallel_probe_ensemble_metrics(
            prediction_bank,
            reference_labels,
            split_name,
        )

    def _parallel_probe_metrics_by_run(
            self,
            logits: np.ndarray,
            labels: np.ndarray,
            data_name: str,
            split_name: str,
        ) -> List[Dict[str, Any]]:
        task_type = self.trainer_args.task_type
        compute_metrics = get_compute_metrics(task_type, tokenwise=False)
        losses = self._parallel_probe_losses_by_run(logits, labels, task_type)
        num_runs = logits.shape[1]
        metrics_by_run = []
        for run_idx in range(num_runs):
            run_logits = logits[:, run_idx, :]
            run_labels = self._parallel_probe_labels_for_run(labels, run_idx, num_runs, run_logits.shape[-1])
            raw_metrics = compute_metrics(EvalPrediction(predictions=run_logits, label_ids=run_labels))
            prefixed_metrics = {
                f"{split_name}_{key}": value
                for key, value in raw_metrics.items()
            }
            prefixed_metrics[f"{split_name}_loss"] = losses[run_idx]

            bw_store = self.balanced_weights if 'balanced_weights' in self.__dict__ else None
            balanced_active = (
                task_type in ('regression', 'sigmoid_regression')
                and self.trainer_args.balanced_regression_metrics
                and bw_store is not None
                and data_name in bw_store
            )
            if balanced_active:
                bw = bw_store[data_name]
                y_pred = np.asarray(run_logits, dtype=np.float64)
                y_true = np.asarray(run_labels, dtype=np.float64)
                if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[-1] == 1:
                    y_pred = np.squeeze(y_pred, axis=-1)
                y_pred = y_pred.flatten()
                y_true = y_true.flatten()
                valid_mask = y_true != -100.0
                if valid_mask.sum() != y_true.size:
                    y_true = y_true[valid_mask]
                    y_pred = y_pred[valid_mask]
                if split_name == 'eval':
                    weights = bw['valid']
                else:
                    weights = bw['test']
                balanced_metrics = compute_balanced_regression_metrics(
                    y_true,
                    y_pred,
                    weights,
                    bin_borders=bw['bin_borders'],
                    n_resamples=self.trainer_args.balanced_n_resamples,
                    seed=self.trainer_args.seed + run_idx,
                )
                for key, value in balanced_metrics.items():
                    prefixed_metrics[f"{split_name}_balanced_{key}"] = value

            metrics_by_run.append(prefixed_metrics)
        return metrics_by_run

    def _parallel_probe_flat_metrics_by_run(
            self,
            logits: np.ndarray,
            labels: np.ndarray,
            data_name: str,
            split_name: str,
        ) -> Dict[str, Any]:
        metrics_by_run = self._parallel_probe_metrics_by_run(logits, labels, data_name, split_name)
        flat_metrics = {}
        split_prefix = f"{split_name}_"
        for run_idx, run_metrics in enumerate(metrics_by_run):
            for key, value in run_metrics.items():
                metric_name = key
                if key.startswith(split_prefix):
                    metric_name = key[len(split_prefix):]
                flat_metrics[f"run_{run_idx}_{metric_name}"] = value
        return self._parallel_probe_json_safe(flat_metrics)

    def _parallel_probe_compute_metrics(self, data_name: str, split_name: str):
        def compute_metrics(eval_prediction: EvalPrediction) -> Dict[str, Any]:
            logits = np.asarray(self._extract_prediction_array(eval_prediction.predictions), dtype=np.float32)
            labels = np.asarray(self._extract_label_array(eval_prediction.label_ids))
            return self._parallel_probe_flat_metrics_by_run(logits, labels, data_name, split_name)

        return compute_metrics

    def _can_parallelize_probe_runs(self, full: bool, ppi: bool = False) -> bool:
        return (
            self.trainer_args.parallel_probe_runs
            and self.trainer_args.num_runs > 1
            and self.probe_args.probe_type == 'mlp'
            and not self.probe_args.tokenwise
            and not self.embedding_args.matrix_embed
            and not ppi
            and not full
        )

    def _parallel_probe_trainer_key(self) -> str:
        return (
            f"epochs={self.trainer_args.num_epochs}|"
            f"batch={self.trainer_args.probe_batch_size}|"
            f"grad_accum={self.trainer_args.probe_grad_accum}|"
            f"lr={self.trainer_args.lr}|"
            f"probe_lr={self.trainer_args.probe_lr}|"
            f"scheduler={self.trainer_args.lr_scheduler}|"
            f"optimizer={self.trainer_args.optimizer}|"
            f"weight_decay={self.trainer_args.weight_decay}|"
            f"patience={self.trainer_args.patience}|"
            f"parallel_max_grad_norm={self.trainer_args.parallel_probe_max_grad_norm}|"
            f"parallel_grad_clip_mode={self.trainer_args.parallel_probe_grad_clip_mode}|"
            f"batch_mode={self.trainer_args.parallel_probe_batch_mode}|"
            f"index_strategy={self.trainer_args.parallel_probe_index_strategy}"
        )

    def _parallel_probe_train_dataset(self, train_dataset, run_seeds: List[int], ppi: bool):
        if self.trainer_args.parallel_probe_batch_mode == 'shared':
            return train_dataset
        assert not ppi, "run_specific parallel probe batches currently support pooled non-PPI datasets only."
        return ParallelRunDataset(
            train_dataset,
            run_seeds=run_seeds,
            independent_shuffles=True,
            index_strategy=self.trainer_args.parallel_probe_index_strategy,
        )

    def _parallel_probe_hf_max_grad_norm(self) -> float:
        if self.trainer_args.parallel_probe_grad_clip_mode == 'global':
            return self.trainer_args.parallel_probe_max_grad_norm
        return 0.0

    def _parallel_probe_callbacks(self):
        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.trainer_args.patience)]
        if (
                self.trainer_args.parallel_probe_grad_clip_mode == 'per_run'
                and self.trainer_args.parallel_probe_max_grad_norm > 0.0
            ):
            callbacks.append(
                ParallelProbePerRunGradientClipCallback(self.trainer_args.parallel_probe_max_grad_norm)
            )
        return callbacks

    def _unwrap_parallel_probe_model(self, model) -> ParallelLinearProbe:
        if isinstance(model, ParallelLinearProbe):
            return model
        model_dict = model.__dict__
        assert '_orig_mod' in model_dict, (
            "Expected a ParallelLinearProbe or a torch.compile wrapper with _orig_mod."
        )
        original_model = model_dict['_orig_mod']
        assert isinstance(original_model, ParallelLinearProbe), (
            f"Expected compiled wrapper to contain ParallelLinearProbe, got {type(original_model)}."
        )
        return original_model

    def _parallel_probe_json_safe(self, value):
        if isinstance(value, dict):
            return {
                str(key): self._parallel_probe_json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._parallel_probe_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._parallel_probe_json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    def _parallel_probe_effective_max_group_size(
            self,
            run_specs,
            train_dataset_size: int,
        ) -> Optional[int]:
        cap_metadata = self._parallel_probe_group_size_cap_metadata(
            run_specs,
            train_dataset_size=train_dataset_size,
        )
        if len(cap_metadata['candidate_group_sizes']) == 0:
            return None
        return cap_metadata['effective_max_group_size']

    def _parallel_probe_group_size_cap_metadata(
            self,
            run_specs,
            train_dataset_size: int,
        ) -> Dict[str, Any]:
        explicit_max_group_size = self.trainer_args.parallel_probe_max_group_size
        candidate_group_sizes = []
        explicit_group_size = None
        if explicit_max_group_size is not None:
            explicit_group_size = explicit_max_group_size
            candidate_group_sizes.append(explicit_group_size)

        training_state_budget_gb = self.trainer_args.parallel_probe_training_state_budget_gb
        training_state_group_size = None
        if training_state_budget_gb is not None:
            training_state_budget_bytes = int(training_state_budget_gb * (1024 ** 3))
            training_state_group_size = max_linear_probe_runs_for_training_state_budget(
                run_specs[0],
                memory_budget_bytes=training_state_budget_bytes,
                dtype_bytes=4,
                optimizer_state_multiplier=2,
            )
            candidate_group_sizes.append(training_state_group_size)

        estimated_peak_budget_gb = self.trainer_args.parallel_probe_estimated_peak_budget_gb
        estimated_peak_group_size = None
        include_run_specific_index = (
            self.trainer_args.parallel_probe_batch_mode == 'run_specific'
            and self.trainer_args.parallel_probe_index_strategy == 'permutation'
        )
        if estimated_peak_budget_gb is not None:
            estimated_peak_budget_bytes = int(estimated_peak_budget_gb * (1024 ** 3))
            estimated_peak_group_size = max_linear_probe_runs_for_estimated_peak_budget(
                run_specs[0],
                memory_budget_bytes=estimated_peak_budget_bytes,
                batch_size=self.trainer_args.probe_batch_size,
                dataset_size=train_dataset_size,
                include_run_specific_index=include_run_specific_index,
                dtype_bytes=4,
                optimizer_state_multiplier=2,
                index_dtype_bytes=8,
            )
            candidate_group_sizes.append(estimated_peak_group_size)

        if len(candidate_group_sizes) == 0:
            effective_max_group_size = None
        else:
            effective_max_group_size = min(self.trainer_args.num_runs, min(candidate_group_sizes))

        return {
            'explicit_max_group_size': explicit_group_size,
            'training_state_budget_group_size': training_state_group_size,
            'estimated_peak_budget_group_size': estimated_peak_group_size,
            'candidate_group_sizes': candidate_group_sizes,
            'effective_max_group_size': effective_max_group_size,
            'include_run_specific_index_in_peak_budget': include_run_specific_index,
        }

    def _save_parallel_best_probe_to_hub(
            self,
            best_model,
            train_dataset,
            valid_dataset,
            test_dataset,
            valid_metrics: Dict[str, Any],
            test_metrics: Dict[str, Any],
            tokenizer,
            log_id: str,
            model_name: str,
            data_name: str,
            source_model_name: Optional[str],
            ppi: bool,
        ) -> None:
        try:
            if source_model_name is None:
                source_model_name = model_name
            hf_username = self.full_args.hf_username
            if hf_username is None or hf_username == "":
                print_message("Warning: hf_username is not set. Cannot save model to HuggingFace Hub.")
                return

            repo_id = self.trainer_args.push_raw_probe_repo or f"{hf_username}/{data_name}_{model_name}_{log_id}"
            if self.full_args.hf_token is None:
                hf_token = os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else None
            else:
                hf_token = self.full_args.hf_token

            model_card = self._build_model_card(
                repo_id=repo_id,
                data_name=data_name,
                model_name=model_name,
                log_id=log_id,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                valid_metrics=valid_metrics,
                test_metrics=test_metrics,
            )

            if self.trainer_args.push_raw_probe:
                if hf_token is not None:
                    best_model.push_to_hub(repo_id, private=True, token=hf_token)
                    api = HfApi(token=hf_token)
                else:
                    best_model.push_to_hub(repo_id, private=True)
                    api = HfApi()
                api.upload_file(
                    path_or_fileobj=model_card.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                )
                print_message(
                    f"Raw best parallel probe uploaded to Hugging Face Hub: {repo_id} "
                    f"(load with e.g. MLPProbe.from_pretrained('{repo_id}'))"
                )
            else:
                packaged_export_succeeded = False
                try:
                    packaged_export_succeeded, export_message = export_packaged_model_to_hub(
                        trained_model=best_model,
                        source_model_name=source_model_name,
                        probe_args=self.probe_args,
                        embedding_args=self.embedding_args,
                        tokenizer=tokenizer,
                        repo_id=repo_id,
                        model_card=model_card,
                        ppi=ppi,
                        private=True,
                        hf_token=hf_token,
                    )
                    print_message(export_message)
                except Exception as packaged_error:
                    print_message(f"Warning: packaged export failed for {repo_id}: {packaged_error}")

                if not packaged_export_succeeded:
                    print_message(f"Falling back to direct best parallel probe push_to_hub for {repo_id}")
                    if hf_token is not None:
                        best_model.push_to_hub(repo_id, private=True, token=hf_token)
                        api = HfApi(token=hf_token)
                    else:
                        best_model.push_to_hub(repo_id, private=True)
                        api = HfApi()
                    api.upload_file(
                        path_or_fileobj=model_card.encode("utf-8"),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type="model",
                    )

            print_message(f"Successfully saved best parallel probe to HuggingFace Hub: {repo_id}")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print_message(f"Error saving best parallel probe to HuggingFace Hub: {e}")
            print_message(f"Error traceback: {error_trace}")
            print_message(f"save_model flag: {self.trainer_args.save}")

    def _train_parallel_linear_probe_runs(
            self,
            train_dataset,
            valid_dataset,
            test_dataset,
            data_collator,
            tokenizer,
            log_id,
            model_name,
            data_name,
            source_model_name: Optional[str] = None,
            ppi: bool = False,
            skip_plot: bool = False,
        ):
        assert valid_dataset is not None, "Parallel probe runs require a validation dataset for early stopping."
        task_type = self.trainer_args.task_type
        base_seed = self.trainer_args.seed
        embedding_key = source_model_name if source_model_name is not None else model_name
        dataset_key = f"{data_name}|ppi={ppi}"
        run_specs = build_seed_run_specs(
            run_id_prefix=f"{data_name}/{model_name}",
            base_seed=base_seed,
            num_runs=self.trainer_args.num_runs,
            model_name=model_name,
            data_name=data_name,
            embedding_key=embedding_key,
            dataset_key=dataset_key,
            trainer_key=self._parallel_probe_trainer_key(),
            probe_type=self.probe_args.probe_type,
            input_size=self.probe_args.input_size,
            hidden_size=self.probe_args.hidden_size,
            dropout=self.probe_args.dropout,
            num_labels=self.probe_args.num_labels,
            n_layers=self.probe_args.n_layers,
            task_type=self.probe_args.task_type,
            pre_ln=self.probe_args.pre_ln,
            use_bias=self.probe_args.use_bias,
            batch_mode=self.trainer_args.parallel_probe_batch_mode,
            index_strategy=self.trainer_args.parallel_probe_index_strategy,
            tokenwise=self.probe_args.tokenwise,
            matrix_embed=self.embedding_args.matrix_embed,
            full_finetuning=False,
            save_model=self.trainer_args.save,
        )
        train_dataset_size = len(train_dataset)
        cap_metadata = self._parallel_probe_group_size_cap_metadata(
            run_specs,
            train_dataset_size=train_dataset_size,
        )
        effective_max_group_size = cap_metadata['effective_max_group_size']
        parallel_plan = plan_parallel_probe_runs(
            run_specs,
            max_parallel_group_size=effective_max_group_size,
        )
        include_run_specific_index = (
            self.trainer_args.parallel_probe_batch_mode == 'run_specific'
            and self.trainer_args.parallel_probe_index_strategy == 'permutation'
        )
        parallel_plan_estimate = estimate_parallel_probe_plan(
            parallel_plan,
            dtype_bytes=4,
            optimizer_state_multiplier=2,
            batch_size=self.trainer_args.probe_batch_size,
            dataset_size=train_dataset_size,
            include_run_specific_index=include_run_specific_index,
            index_dtype_bytes=8,
        )
        assert all(group.eligible for group in parallel_plan.groups), (
            "Parallel probe trainer received an ineligible execution group."
        )
        all_run_seeds = [spec.seed for spec in run_specs]
        self.trainer_args.train_data_size = train_dataset_size
        self.trainer_args.num_labels = self.probe_args.num_labels
        self.trainer_args.eval_dataset_size = len(valid_dataset)
        previous_eval_output_multiplier = self.trainer_args.eval_output_multiplier

        try:
            all_valid_metrics = []
            all_test_metrics = []
            total_train_runtime = 0.0
            peak_index_memory_bytes = 0
            total_index_memory_bytes = 0
            best_model = None
            best_y_pred = None
            best_y_true = None
            best_run_idx = -1
            best_run_id = ""
            best_seed = base_seed
            best_loss = float('inf')
            group_output_dirs = []
            group_runtime_records = []
            parallel_run_records = []
            valid_logits_by_run = [None] * self.trainer_args.num_runs
            test_logits_by_run = [None] * self.trainer_args.num_runs
            valid_labels_for_ensemble = None
            test_labels_for_ensemble = None

            for group_idx, parallel_group in enumerate(
                    parallel_plan.execution_groups(prefer_largest_parallel=False)
                ):
                group_num_runs = parallel_group.num_runs
                run_seeds = list(parallel_group.run_seeds)
                parallel_train_dataset = self._parallel_probe_train_dataset(train_dataset, run_seeds, ppi)
                if isinstance(parallel_train_dataset, ParallelRunDataset):
                    group_index_memory_bytes = parallel_train_dataset.index_memory_bytes
                else:
                    group_index_memory_bytes = 0
                peak_index_memory_bytes = max(peak_index_memory_bytes, group_index_memory_bytes)
                total_index_memory_bytes += group_index_memory_bytes
                self.trainer_args.eval_output_multiplier = group_num_runs

                config = ParallelLinearProbeConfig(
                    **self.probe_args.__dict__,
                    num_runs=group_num_runs,
                    run_seeds=run_seeds,
                )
                parallel_model = ParallelLinearProbe(config)
                hf_trainer_args = self.trainer_args(probe=True)
                hf_trainer_args.max_grad_norm = self._parallel_probe_hf_max_grad_norm()
                if parallel_plan.trainer_invocations > 1:
                    hf_trainer_args.output_dir = os.path.join(
                        hf_trainer_args.output_dir,
                        f"parallel_group_{group_idx + 1}",
                    )
                group_output_dirs.append(hf_trainer_args.output_dir)
                trainer = Trainer(
                    model=parallel_model,
                    args=hf_trainer_args,
                    train_dataset=parallel_train_dataset,
                    eval_dataset=valid_dataset,
                    data_collator=data_collator,
                    compute_metrics=self._parallel_probe_compute_metrics(data_name, 'eval'),
                    callbacks=self._parallel_probe_callbacks()
                )
                trainer.can_return_loss = True
                initial_metrics = trainer.evaluate(valid_dataset)
                print_message(f'Initial parallel probe group {group_idx + 1} metrics: {initial_metrics}')

                train_output = trainer.train()
                train_metrics = train_output.metrics
                train_runtime = train_metrics['train_runtime'] if 'train_runtime' in train_metrics else 0.0
                total_train_runtime += train_runtime
                group_estimate = parallel_plan_estimate.group_estimates[group_idx]
                group_runtime_records.append(
                    {
                        'group_index': group_idx,
                        'group_number': group_idx + 1,
                        'execution_kind': parallel_group.execution_kind,
                        'num_runs': group_num_runs,
                        'run_seeds': run_seeds,
                        'train_runtime_seconds': train_runtime,
                        'seconds_per_run': train_runtime / float(group_num_runs),
                        'output_dir': hf_trainer_args.output_dir,
                        'index_memory_bytes': group_index_memory_bytes,
                        'estimated_training_state_bytes': group_estimate.training_state_bytes,
                        'estimated_peak_bytes': group_estimate.estimated_peak_bytes,
                        'estimated_training_flops_per_batch': group_estimate.group_training_flops_per_batch,
                    }
                )

                trainer.compute_metrics = None
                valid_output = trainer.predict(valid_dataset)
                test_output = trainer.predict(test_dataset)
                valid_logits = self._extract_prediction_array(valid_output.predictions).astype(np.float32)
                test_logits = self._extract_prediction_array(test_output.predictions).astype(np.float32)
                valid_labels = self._extract_label_array(valid_output.label_ids).astype(np.float32)
                test_labels = self._extract_label_array(test_output.label_ids).astype(np.float32)
                if valid_labels_for_ensemble is None:
                    valid_labels_for_ensemble = valid_labels
                else:
                    assert np.array_equal(valid_labels_for_ensemble, valid_labels), (
                        "Parallel probe validation labels changed between seed groups."
                    )
                if test_labels_for_ensemble is None:
                    test_labels_for_ensemble = test_labels
                else:
                    assert np.array_equal(test_labels_for_ensemble, test_labels), (
                        "Parallel probe test labels changed between seed groups."
                    )

                group_valid_metrics = self._parallel_probe_metrics_by_run(
                    valid_logits,
                    valid_labels,
                    data_name,
                    'eval',
                )
                group_test_metrics = self._parallel_probe_metrics_by_run(
                    test_logits,
                    test_labels,
                    data_name,
                    'test',
                )
                all_valid_metrics.extend(group_valid_metrics)
                all_test_metrics.extend(group_test_metrics)

                parallel_model = self._unwrap_parallel_probe_model(trainer.model).cpu()
                for local_run_idx, test_metrics in enumerate(group_test_metrics):
                    valid_metrics = group_valid_metrics[local_run_idx]
                    test_loss = test_metrics['test_loss']
                    if 'eval_loss' in valid_metrics:
                        valid_loss = valid_metrics['eval_loss']
                    else:
                        valid_loss = None
                    run_spec = parallel_group.runs[local_run_idx]
                    global_run_idx = run_spec.seed - base_seed
                    assert 0 <= global_run_idx < self.trainer_args.num_runs, (
                        f"Unexpected global run index {global_run_idx} for seed {run_spec.seed}."
                    )
                    valid_logits_by_run[global_run_idx] = valid_logits[:, local_run_idx, :].astype(np.float32)
                    test_logits_by_run[global_run_idx] = test_logits[:, local_run_idx, :].astype(np.float32)
                    parallel_run_records.append(
                        {
                            'run_index': global_run_idx,
                            'run_number': global_run_idx + 1,
                            'seed': run_spec.seed,
                            'run_id': run_spec.run_id,
                            'group_index': group_idx,
                            'group_number': group_idx + 1,
                            'local_run_index': local_run_idx,
                            'local_run_number': local_run_idx + 1,
                            'valid_loss': valid_loss,
                            'test_loss': test_loss,
                        }
                    )
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_run_idx = global_run_idx
                        best_run_id = run_spec.run_id
                        best_seed = run_spec.seed
                        best_model = parallel_model.to_mlp_probe(local_run_idx)
                        best_y_pred = test_logits[:, local_run_idx, :].astype(np.float32)
                        best_y_true = test_labels.astype(np.float32)

                trainer.accelerator.free_memory()
                torch.cuda.empty_cache()

            assert best_model is not None, "Parallel probe training did not produce a best model."
            assert best_y_pred is not None, "Parallel probe training did not produce predictions."
            assert best_y_true is not None, "Parallel probe training did not produce labels."
            assert valid_labels_for_ensemble is not None, "Parallel probe training did not produce validation labels."
            assert test_labels_for_ensemble is not None, "Parallel probe training did not produce test labels."
            assert all(logits is not None for logits in valid_logits_by_run), (
                "Parallel probe training did not produce validation logits for every run."
            )
            assert all(logits is not None for logits in test_logits_by_run), (
                "Parallel probe training did not produce test logits for every run."
            )
            valid_bank_logits = np.stack(valid_logits_by_run, axis=1)
            test_bank_logits = np.stack(test_logits_by_run, axis=1)
            ensemble_valid_metrics = self._parallel_probe_ensemble_metrics(
                valid_bank_logits,
                valid_labels_for_ensemble,
                'eval',
            )
            ensemble_test_metrics = self._parallel_probe_ensemble_metrics(
                test_bank_logits,
                test_labels_for_ensemble,
                'test',
            )
            aggregated_valid = self._aggregate_metrics(all_valid_metrics)
            aggregated_test = self._aggregate_metrics(all_test_metrics)
            for key, value in ensemble_valid_metrics.items():
                aggregated_valid[f"parallel_probe_ensemble_{key}"] = value
            for key, value in ensemble_test_metrics.items():
                aggregated_test[f"parallel_probe_ensemble_{key}"] = value
            aggregated_test['training_time_seconds'] = total_train_runtime
            aggregated_test['parallel_probe_num_runs'] = parallel_plan.total_runs
            aggregated_test['parallel_probe_seconds_per_run'] = total_train_runtime / float(parallel_plan.total_runs)
            aggregated_test['parallel_probe_batch_mode'] = self.trainer_args.parallel_probe_batch_mode
            aggregated_test['parallel_probe_index_strategy'] = self.trainer_args.parallel_probe_index_strategy
            aggregated_test['parallel_probe_ensemble_average_mode'] = (
                self.trainer_args.parallel_probe_ensemble_average_mode
            )
            aggregated_test['parallel_probe_max_group_size'] = self.trainer_args.parallel_probe_max_group_size
            aggregated_test['parallel_probe_training_state_budget_gb'] = (
                self.trainer_args.parallel_probe_training_state_budget_gb
            )
            aggregated_test['parallel_probe_estimated_peak_budget_gb'] = (
                self.trainer_args.parallel_probe_estimated_peak_budget_gb
            )
            aggregated_test['parallel_probe_max_grad_norm'] = self.trainer_args.parallel_probe_max_grad_norm
            aggregated_test['parallel_probe_grad_clip_mode'] = self.trainer_args.parallel_probe_grad_clip_mode
            aggregated_test['parallel_probe_effective_max_group_size'] = effective_max_group_size
            aggregated_test['parallel_probe_explicit_max_group_size'] = (
                cap_metadata['explicit_max_group_size']
            )
            aggregated_test['parallel_probe_training_state_budget_group_size'] = (
                cap_metadata['training_state_budget_group_size']
            )
            aggregated_test['parallel_probe_estimated_peak_budget_group_size'] = (
                cap_metadata['estimated_peak_budget_group_size']
            )
            aggregated_test['parallel_probe_group_size_candidates'] = (
                cap_metadata['candidate_group_sizes']
            )
            aggregated_test['parallel_probe_peak_budget_includes_index'] = (
                cap_metadata['include_run_specific_index_in_peak_budget']
            )
            aggregated_test['parallel_probe_group_sizes'] = [group.num_runs for group in parallel_plan.groups]
            aggregated_test['parallel_probe_group_run_seeds'] = [
                list(group.run_seeds) for group in parallel_plan.groups
            ]
            aggregated_test['parallel_probe_group_output_dirs'] = group_output_dirs
            aggregated_test['parallel_probe_group_runtime_records'] = (
                self._parallel_probe_json_safe(group_runtime_records)
            )
            aggregated_test['parallel_probe_estimated_parameter_count'] = (
                parallel_plan_estimate.total_parameter_count
            )
            aggregated_test['parallel_probe_estimated_training_state_bytes'] = (
                parallel_plan_estimate.total_training_state_bytes
            )
            aggregated_test['parallel_probe_estimated_peak_group_training_state_bytes'] = (
                parallel_plan_estimate.peak_group_training_state_bytes
            )
            aggregated_test['parallel_probe_estimated_batch_activation_bytes'] = (
                parallel_plan_estimate.total_batch_activation_bytes
            )
            aggregated_test['parallel_probe_estimated_peak_group_batch_activation_bytes'] = (
                parallel_plan_estimate.peak_group_batch_activation_bytes
            )
            aggregated_test['parallel_probe_estimated_run_specific_index_bytes'] = (
                parallel_plan_estimate.total_run_specific_index_bytes
            )
            aggregated_test['parallel_probe_estimated_peak_group_bytes'] = (
                parallel_plan_estimate.peak_group_estimated_peak_bytes
            )
            aggregated_test['parallel_probe_estimated_forward_flops_per_batch'] = (
                parallel_plan_estimate.total_forward_flops_per_batch
            )
            aggregated_test['parallel_probe_estimated_peak_group_forward_flops_per_batch'] = (
                parallel_plan_estimate.peak_group_forward_flops_per_batch
            )
            aggregated_test['parallel_probe_estimated_training_flops_per_batch'] = (
                parallel_plan_estimate.total_training_flops_per_batch
            )
            aggregated_test['parallel_probe_estimated_peak_group_training_flops_per_batch'] = (
                parallel_plan_estimate.peak_group_training_flops_per_batch
            )
            aggregated_test['parallel_probe_index_memory_bytes'] = peak_index_memory_bytes
            aggregated_test['parallel_probe_index_memory_total_bytes'] = total_index_memory_bytes
            aggregated_test['parallel_probe_total_runs'] = parallel_plan.total_runs
            aggregated_test['parallel_probe_vectorized_runs'] = parallel_plan.vectorized_runs
            aggregated_test['parallel_probe_sequential_runs'] = parallel_plan.sequential_runs
            aggregated_test['parallel_probe_trainer_invocations'] = parallel_plan.trainer_invocations
            aggregated_test['parallel_probe_invocation_reduction'] = parallel_plan.invocation_reduction
            aggregated_test['parallel_probe_compression_ratio'] = parallel_plan.compression_ratio
            aggregated_test['parallel_probe_run_seeds'] = all_run_seeds
            aggregated_test['parallel_probe_run_records'] = self._parallel_probe_json_safe(parallel_run_records)
            aggregated_test['parallel_probe_valid_run_metrics'] = self._parallel_probe_json_safe(all_valid_metrics)
            aggregated_test['parallel_probe_test_run_metrics'] = self._parallel_probe_json_safe(all_test_metrics)
            aggregated_test['parallel_probe_best_selection_metric'] = 'test_loss'
            aggregated_test['parallel_probe_best_run_index'] = best_run_idx
            aggregated_test['parallel_probe_best_run_number'] = best_run_idx + 1
            aggregated_test['parallel_probe_best_run_id'] = best_run_id
            aggregated_test['parallel_probe_best_seed'] = best_seed
            aggregated_test['parallel_probe_best_test_loss'] = best_loss
            print_message(f"Best parallel run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")

            if self.trainer_args.make_plots and self.trainer_args.plots_dir is not None and not skip_plot:
                output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_parallel_best.png")
                title = f"{data_name} {model_name} (best of {parallel_plan.total_runs} parallel runs, seed={best_seed})"
                if task_type == 'regression':
                    regression_ci_plot(best_y_true, best_y_pred, save_path, title)
                else:
                    classification_ci_plot(best_y_true, best_y_pred, save_path, title)

            if self.trainer_args.save:
                self._save_parallel_best_probe_to_hub(
                    best_model=best_model,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    test_dataset=test_dataset,
                    valid_metrics=aggregated_valid,
                    test_metrics=aggregated_test,
                    tokenizer=tokenizer,
                    log_id=log_id,
                    model_name=model_name,
                    data_name=data_name,
                    source_model_name=source_model_name,
                    ppi=ppi,
                )
            return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true
        finally:
            self.trainer_args.eval_output_multiplier = previous_eval_output_multiplier

    def _train(
            self,
            model,
            train_dataset,
            valid_dataset,
            test_dataset,
            data_collator,
            tokenizer,
            log_id,
            model_name,
            data_name,
            source_model_name: Optional[str] = None,
            ppi: bool = False,
            probe: Optional[bool] = True,
            skip_plot: bool = False,
        ):
        task_type = self.trainer_args.task_type
        tokenwise = self.probe_args.tokenwise
        compute_metrics = get_compute_metrics(task_type, tokenwise=tokenwise)
        self.trainer_args.train_data_size = len(train_dataset)
        self.trainer_args.num_labels = self.probe_args.num_labels
        self.trainer_args.eval_dataset_size = len(valid_dataset) if valid_dataset is not None else len(test_dataset)
        self.trainer_args.eval_output_multiplier = 1
        hf_trainer_args = self.trainer_args(probe=probe)
        if probe or self.trainer_args.base_patience is None:
            early_stopping_patience = self.trainer_args.patience
        else:
            early_stopping_patience = self.trainer_args.base_patience
        trainer = Trainer(
            model=model,
            args=hf_trainer_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        trainer.can_return_loss = True
        metrics = trainer.evaluate(test_dataset)
        print_message(f'Initial metrics: {metrics}')

        bw_store = self.balanced_weights if 'balanced_weights' in self.__dict__ else None
        bw = bw_store[data_name] if (bw_store is not None and data_name in bw_store) else None
        balanced_active = (
            task_type in ('regression', 'sigmoid_regression')
            and self.trainer_args.balanced_regression_metrics
            and bw is not None
        )
        if balanced_active:
            trainer.compute_metrics = get_compute_metrics_with_balanced(
                compute_metrics,
                weights=bw['valid'],
                bin_borders=bw['bin_borders'],
                n_resamples=self.trainer_args.balanced_n_resamples,
                seed=self.trainer_args.seed,
            )

        train_output = trainer.train()
        train_runtime = train_output.metrics.get('train_runtime', 0.0)

        valid_metrics = trainer.evaluate(valid_dataset)
        print_message(f'Final validation metrics: {valid_metrics}')

        y_pred_valid, y_true_valid, _vm_raw = trainer.predict(valid_dataset)
        if isinstance(y_pred_valid, tuple):
            y_pred_valid = y_pred_valid[0]
        if isinstance(y_true_valid, tuple):
            y_true_valid = y_true_valid[0]
        y_pred_valid = y_pred_valid.astype(np.float32)
        y_true_valid = y_true_valid.astype(np.float32)

        if balanced_active:
            trainer.compute_metrics = compute_metrics
        y_pred, y_true, test_metrics = trainer.predict(test_dataset)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if isinstance(y_true, tuple):
            y_true = y_true[0]

        y_pred, y_true = y_pred.astype(np.float32), y_true.astype(np.float32)

        # Remove singleton dimension if present
        if y_pred.ndim == 3 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        if y_true.ndim == 3 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        if task_type in ('regression', 'sigmoid_regression') and self.trainer_args.balanced_regression_metrics:
            bw_store = self.balanced_weights if 'balanced_weights' in self.__dict__ else None
            bw = bw_store[data_name] if (bw_store is not None and data_name in bw_store) else None
            if bw is not None:
                bin_borders = bw['bin_borders']
                n_res = self.trainer_args.balanced_n_resamples
                valid_bal = compute_balanced_regression_metrics(
                    y_true_valid.flatten(),
                    y_pred_valid.flatten(),
                    bw['valid'],
                    bin_borders=bin_borders,
                    n_resamples=n_res,
                    seed=self.trainer_args.seed,
                )
                test_bal = compute_balanced_regression_metrics(
                    y_true.flatten(),
                    y_pred.flatten(),
                    bw['test'],
                    bin_borders=bin_borders,
                    n_resamples=n_res,
                    seed=self.trainer_args.seed,
                )
                for k, v in valid_bal.items():
                    valid_metrics[f'balanced_{k}'] = v
                for k, v in test_bal.items():
                    test_metrics[f'balanced_{k}'] = v

        test_metrics['training_time_seconds'] = train_runtime
        print_message(f'y_pred: {y_pred.shape}\ny_true: {y_true.shape}\nFinal test metrics: \n{test_metrics}\n')

        if self.trainer_args.make_plots and self.trainer_args.plots_dir is not None and not skip_plot:
            output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}.png")
            title = f"{data_name} {model_name} {log_id}"

            if task_type == 'regression':
                regression_ci_plot(y_true, y_pred, save_path, title)
            else:
                classification_ci_plot(y_true, y_pred, save_path, title)

        if source_model_name is None:
            source_model_name = model_name

        if self.trainer_args.save:
            try:
                hf_username = self.full_args.hf_username
                if hf_username is None or hf_username == "":
                    print_message("Warning: hf_username is not set. Cannot save model to HuggingFace Hub.")
                else:
                    repo_id = self.trainer_args.push_raw_probe_repo or f"{hf_username}/{data_name}_{model_name}_{log_id}"
                    hf_token = self.full_args.hf_token
                    if hf_token is None:
                        hf_token = os.environ.get("HF_TOKEN")

                    model_card = self._build_model_card(
                        repo_id=repo_id,
                        data_name=data_name,
                        model_name=model_name,
                        log_id=log_id,
                        train_dataset=train_dataset,
                        valid_dataset=valid_dataset,
                        test_dataset=test_dataset,
                        valid_metrics=valid_metrics,
                        test_metrics=test_metrics,
                    )

                    if self.trainer_args.push_raw_probe and (probe or isinstance(trainer.model, HybridProbe)):
                        probe_to_push = trainer.model.probe if isinstance(trainer.model, HybridProbe) else trainer.model
                        if hf_token is not None:
                            probe_to_push.push_to_hub(repo_id, private=True, token=hf_token)
                            api = HfApi(token=hf_token)
                        else:
                            probe_to_push.push_to_hub(repo_id, private=True)
                            api = HfApi()
                        api.upload_file(
                            path_or_fileobj=model_card.encode("utf-8"),
                            path_in_repo="README.md",
                            repo_id=repo_id,
                            repo_type="model",
                        )
                        print_message(f"Raw probe uploaded to Hugging Face Hub: {repo_id} (load with e.g. Class.from_pretrained('{repo_id}'))")
                    else:
                        packaged_export_succeeded = False
                        if probe or isinstance(trainer.model, HybridProbe):
                            try:
                                packaged_export_succeeded, export_message = export_packaged_model_to_hub(
                                    trained_model=trainer.model,
                                    source_model_name=source_model_name,
                                    probe_args=self.probe_args,
                                    embedding_args=self.embedding_args,
                                    tokenizer=tokenizer,
                                    repo_id=repo_id,
                                    model_card=model_card,
                                    ppi=ppi,
                                    private=True,
                                    hf_token=hf_token,
                                )
                                print_message(export_message)
                            except Exception as packaged_error:
                                print_message(f"Warning: packaged export failed for {repo_id}: {packaged_error}")

                        if not packaged_export_succeeded:
                            print_message(f"Falling back to direct model push_to_hub for {repo_id}")
                            if hf_token is not None:
                                trainer.model.push_to_hub(repo_id, private=True, token=hf_token)
                                api = HfApi(token=hf_token)
                            else:
                                trainer.model.push_to_hub(repo_id, private=True)
                                api = HfApi()
                            api.upload_file(
                                path_or_fileobj=model_card.encode("utf-8"),
                                path_in_repo="README.md",
                                repo_id=repo_id,
                                repo_type="model",
                            )
                    print_message(f"Successfully saved model to HuggingFace Hub: {repo_id}")
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print_message(f"Error saving model to HuggingFace Hub: {e}")
                print_message(f"Error traceback: {error_trace}")
                print_message(f"save_model flag: {self.trainer_args.save}")

        model = trainer.model.cpu()
        trainer.accelerator.free_memory()
        torch.cuda.empty_cache()
        return model, valid_metrics, test_metrics, y_pred, y_true

    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple runs, computing mean ± std for each metric."""
        if not metrics_list:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        aggregated = {}
        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m and m[key] is not None]
            if not values:
                continue
            
            # Check if all values are numeric
            if all(isinstance(v, (int, float)) for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                # Store as formatted string with mean±std
                aggregated[key] = f"{mean_val:.4f}±{std_val:.4f}"
                # Also store raw mean for sorting/comparison purposes
                aggregated[f"{key}_mean"] = float(mean_val)
                aggregated[f"{key}_std"] = float(std_val)
            else:
                # For non-numeric values, just take the first one
                aggregated[key] = values[0]
        
        return aggregated

    def _iter_vector_embeddings_for_standardizer(
            self,
            train_dataset,
            emb_dict,
            db_path: str,
            ppi: bool,
            use_multi,
        ):
        if self.embedding_args.sql:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                if ppi:
                    for seq_a, seq_b in zip(list(train_dataset['SeqA']), list(train_dataset['SeqB'])):
                        emb_a = self._select_from_sql(cursor, seq_a, cast_to_torch=True)
                        emb_b = self._select_from_sql(cursor, seq_b, cast_to_torch=True)
                        yield torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)
                        if self.full_args.random_pair_flipping:
                            yield torch.cat([emb_b.reshape(1, -1), emb_a.reshape(1, -1)], dim=-1)
                elif use_multi:
                    seq_columns = [list(train_dataset[col]) for col in use_multi]
                    for seqs in zip(*seq_columns):
                        parts = [
                            self._select_from_sql(cursor, seq, cast_to_torch=True).reshape(1, -1)
                            for seq in seqs
                        ]
                        yield torch.cat(parts, dim=-1)
                else:
                    for seq in list(train_dataset['seqs']):
                        yield self._select_from_sql(cursor, seq, cast_to_torch=True)
        else:
            assert emb_dict is not None, "emb_dict is required for in-memory embedding standardization."
            if ppi:
                for seq_a, seq_b in zip(list(train_dataset['SeqA']), list(train_dataset['SeqB'])):
                    emb_a = emb_dict[seq_a]
                    emb_b = emb_dict[seq_b]
                    yield torch.cat([emb_a.reshape(1, -1), emb_b.reshape(1, -1)], dim=-1)
                    if self.full_args.random_pair_flipping:
                        yield torch.cat([emb_b.reshape(1, -1), emb_a.reshape(1, -1)], dim=-1)
            elif use_multi:
                seq_columns = [list(train_dataset[col]) for col in use_multi]
                for seqs in zip(*seq_columns):
                    parts = [emb_dict[seq].reshape(1, -1) for seq in seqs]
                    yield torch.cat(parts, dim=-1)
            else:
                for seq in list(train_dataset['seqs']):
                    yield emb_dict[seq]

    def _fit_embedding_standardizer(
            self,
            train_dataset,
            emb_dict,
            db_path: str,
            ppi: bool,
            use_multi,
            full: bool,
        ) -> Optional[EmbeddingStandardizer]:
        embedding_scaler = self.embedding_args.embedding_scaler
        assert isinstance(embedding_scaler, bool), f"Invalid embedding_scaler: {embedding_scaler}"
        if full or not embedding_scaler:
            return None
        print_message("Fitting StandardScaler on training embeddings")
        embeddings = self._iter_vector_embeddings_for_standardizer(
            train_dataset,
            emb_dict,
            db_path,
            ppi,
            use_multi,
        )
        return EmbeddingStandardizer.fit_tensors(embeddings)

    def trainer_probe(
            self,
            model,
            tokenizer,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            emb_dict=None,
            ppi=False,
            log_id=None,
            skip_plot=False,
            source_model_name: Optional[str] = None,
        ):
        batch_size = self.trainer_args.probe_batch_size
        read_scaler = self.trainer_args.read_scaler
        input_size = self.probe_args.input_size
        task_type = self.probe_args.task_type
        tokenwise = self.probe_args.tokenwise
        num_runs = getattr(self.trainer_args, 'num_runs', 1)
        base_seed = self.trainer_args.seed
        
        print(f'task_type: {task_type}')
        # The cache belongs to the model that produced the embeddings, which is the source
        # model when this run reuses another model's cache.
        cache_model_name = source_model_name or model_name
        db_filename = self._embedding_cache_filename(cache_model_name, 'db')
        db_path = os.path.join(self.embedding_args.embedding_save_dir, db_filename)

        use_multi = getattr(self.full_args, 'multi_column', None)
        if self.embedding_args.sql:
            print('SQL enabled')
            if ppi:
                DatasetClass = PairEmbedsLabelsDatasetFromDisk
                CollatorClass = PairEmbedsLabelsCollator
            elif use_multi:
                DatasetClass = MultiEmbedsLabelsDatasetFromDisk
                CollatorClass = EmbedsLabelsCollator
            else:
                DatasetClass = EmbedsLabelsDatasetFromDisk
                CollatorClass = EmbedsLabelsCollator
        else:
            print('SQL disabled')
            if ppi:
                DatasetClass = PairEmbedsLabelsDataset
                CollatorClass = PairEmbedsLabelsCollator
            elif use_multi:
                DatasetClass = MultiEmbedsLabelsDataset
                CollatorClass = EmbedsLabelsCollator
            else:
                DatasetClass = EmbedsLabelsDataset
                CollatorClass = EmbedsLabelsCollator

        """
        For collator need to pass tokenizer, full, task_type
        For dataset need to pass
        hf_dataset, col_a, col_b, label_col, input_size, task_type, db_path, emb_dict, batch_size, read_scaler, full, train
        """

        add_token_ids = getattr(self.probe_args, 'add_token_ids', False)
        padding = getattr(self.full_args, 'padding', 'max_length')
        max_length = getattr(self.full_args, 'max_length', 2048)
        data_collator = CollatorClass(tokenizer=tokenizer, full=full, task_type=task_type, tokenwise=tokenwise, add_token_ids=add_token_ids, padding=padding, max_length=max_length)
        embedding_standardizer = self._fit_embedding_standardizer(
            train_dataset,
            emb_dict,
            db_path,
            ppi,
            use_multi,
            full,
        )
        common_kwargs = dict(
            hf_dataset=train_dataset,
            input_size=input_size,
            task_type=task_type,
            db_path=db_path,
            emb_dict=emb_dict,
            batch_size=batch_size,
            read_scaler=read_scaler,
            full=full,
            train=True,
            random_pair_flipping=self.full_args.random_pair_flipping,
            embedding_standardizer=embedding_standardizer,
        )
        if use_multi:
            train_ds = DatasetClass(seq_cols=use_multi, **deepcopy(common_kwargs))
        else:
            train_ds = DatasetClass(**deepcopy(common_kwargs))
        
        # BUG FIX: Update hf_dataset in common_kwargs before creating validation and test datasets.
        # Previously, common_kwargs['hf_dataset'] was set to train_dataset and never updated,
        # causing valid_dataset and test_dataset to incorrectly use training data. This resulted
        # in valid_metrics and test_metrics being identical since they were computed on the same
        # (training) dataset. The fix ensures each dataset uses the correct HuggingFace dataset.
        # We use deepcopy to ensure each dataset gets an independent copy of the kwargs dictionary
        # to prevent any potential shared state issues.
        common_kwargs['train'] = False
        common_kwargs['hf_dataset'] = valid_dataset
        if use_multi:
            valid_ds = DatasetClass(seq_cols=use_multi, **deepcopy(common_kwargs))
        else:
            valid_ds = DatasetClass(**deepcopy(common_kwargs))
        common_kwargs['hf_dataset'] = test_dataset
        if use_multi:
            test_ds = DatasetClass(seq_cols=use_multi, **deepcopy(common_kwargs))
        else:
            test_ds = DatasetClass(**deepcopy(common_kwargs))
        
        # Single run - original behavior
        if num_runs == 1:
            return self._train(
                model=model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=log_id,
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=True,
                skip_plot=skip_plot,
            )

        if self.trainer_args.parallel_probe_runs:
            if self._can_parallelize_probe_runs(full, ppi=ppi):
                batch_mode = self.trainer_args.parallel_probe_batch_mode
                if batch_mode == 'shared':
                    print_message(
                        f"Running {num_runs} linear probe runs in parallel for {data_name}/{model_name}. "
                        "All runs share minibatch order and data loading while keeping independent parameters."
                    )
                else:
                    print_message(
                        f"Running {num_runs} linear probe runs in parallel for {data_name}/{model_name}. "
                        "Each run uses a deterministic per-seed training permutation."
                    )
                return self._train_parallel_linear_probe_runs(
                    train_dataset=train_ds,
                    valid_dataset=valid_ds,
                    test_dataset=test_ds,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                    log_id=log_id,
                    model_name=model_name,
                    data_name=data_name,
                    source_model_name=source_model_name,
                    ppi=ppi,
                    skip_plot=skip_plot,
                )
            print_message(
                "--parallel_probe_runs is only available for pooled, sequence-level non-PPI linear probes. "
                "Falling back to sequential num_runs."
            )

        # Multi-run mode: train multiple times with different seeds, reusing datasets
        print_message(f"Running {num_runs} training runs with different seeds for {data_name}/{model_name}")

        all_valid_metrics = []
        all_test_metrics = []
        run_results = []  # Store (run_idx, test_loss, y_pred, y_true, seed, model) for plotting best
        
        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            self.trainer_args.seed = run_seed
            set_global_seed(run_seed)
            
            print_message(f"=== Run {run_idx + 1}/{num_runs} with seed {run_seed} ===")
            
            # Create a fresh probe for each run
            probe = get_probe(self.probe_args)
            
            run_model, valid_metrics, test_metrics, y_pred, y_true = self._train(
                model=probe,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=f"{log_id}_run{run_idx}",
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=True,
                skip_plot=True,  # Skip plots during individual runs
            )
            
            all_valid_metrics.append(valid_metrics)
            all_test_metrics.append(test_metrics)
            
            # Track test loss for determining best run
            test_loss = test_metrics.get('test_loss', test_metrics.get('eval_loss', float('inf')))
            run_results.append((run_idx, test_loss, y_pred, y_true, run_seed, run_model))
        
        # Restore original seed
        self.trainer_args.seed = base_seed
        
        # Compute aggregated metrics (mean ± std)
        aggregated_valid = self._aggregate_metrics(all_valid_metrics)
        aggregated_test = self._aggregate_metrics(all_test_metrics)
        ensemble_test_metrics = self._seed_ensemble_metrics_from_run_results(run_results, 'test')
        for key, value in ensemble_test_metrics.items():
            aggregated_test[f"sequential_probe_ensemble_{key}"] = value
        aggregated_test['sequential_probe_ensemble_average_mode'] = (
            self.trainer_args.parallel_probe_ensemble_average_mode
        )
        aggregated_test['sequential_probe_total_runs'] = num_runs
        aggregated_test['sequential_probe_run_seeds'] = [result[4] for result in run_results]
        aggregated_test['sequential_probe_run_records'] = [
            {
                'run_index': result[0],
                'run_number': result[0] + 1,
                'seed': result[4],
                'test_loss': result[1],
            }
            for result in run_results
        ]
        
        # Find the best run (lowest test loss)
        best_run = min(run_results, key=lambda x: x[1])
        best_run_idx, best_loss, best_y_pred, best_y_true, best_seed, best_model = best_run
        print_message(f"Best run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")
        
        # Generate plot for best run (unless skip_plot is True)
        if not skip_plot:
            output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_best.png")
            title = f"{data_name} {model_name} (best of {num_runs} runs, seed={best_seed})"
            
            if task_type == 'regression':
                regression_ci_plot(best_y_true, best_y_pred, save_path, title)
            else:
                classification_ci_plot(best_y_true, best_y_pred, save_path, title)
        
        # Return the best model along with aggregated metrics
        return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true

    def trainer_base_model(
            self,
            model,
            tokenizer,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            ppi=False,
            log_id=None,
            skip_plot=False,
            model_factory=None,
            source_model_name: Optional[str] = None,
        ):
        task_type = self.probe_args.task_type
        tokenwise = self.probe_args.tokenwise
        num_runs = getattr(self.trainer_args, 'num_runs', 1)
        base_seed = self.trainer_args.seed

        if ppi:
            DatasetClass = PairStringLabelDataset
            CollatorClass = PairCollator_input_ids
        else:
            DatasetClass = StringLabelDataset
            CollatorClass = StringLabelsCollator

        padding = getattr(self.full_args, 'padding', 'max_length')
        max_length = getattr(self.full_args, 'max_length', 2048)
        data_collator = CollatorClass(tokenizer=tokenizer, task_type=task_type, tokenwise=tokenwise, padding=padding, max_length=max_length)

        train_ds = DatasetClass(hf_dataset=train_dataset, train=True, random_pair_flipping=self.full_args.random_pair_flipping)
        valid_ds = DatasetClass(hf_dataset=valid_dataset, train=False, random_pair_flipping=self.full_args.random_pair_flipping)
        test_ds = DatasetClass(hf_dataset=test_dataset, train=False, random_pair_flipping=self.full_args.random_pair_flipping)

        # Single run - original behavior
        if num_runs == 1:
            return self._train(
                model=model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=log_id,
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=False,
                skip_plot=skip_plot,
            )
        
        # Multi-run mode: train multiple times with different seeds
        print_message(f"Running {num_runs} full finetuning runs with different seeds for {data_name}/{model_name}")
        
        all_valid_metrics = []
        all_test_metrics = []
        run_results = []  # Store (run_idx, test_loss, y_pred, y_true, seed, model) for plotting best
        
        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx
            self.trainer_args.seed = run_seed
            set_global_seed(run_seed)
            
            print_message(f"=== Run {run_idx + 1}/{num_runs} with seed {run_seed} ===")
            
            # Create a fresh model for each run using the factory
            if model_factory is not None:
                run_model = model_factory()
            
            trained_model, valid_metrics, test_metrics, y_pred, y_true = self._train(
                model=run_model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                test_dataset=test_ds,
                data_collator=data_collator,
                tokenizer=tokenizer,
                log_id=f"{log_id}_run{run_idx}",
                model_name=model_name,
                data_name=data_name,
                source_model_name=source_model_name,
                ppi=ppi,
                probe=False,
                skip_plot=True,  # Skip plots during individual runs
            )
            
            all_valid_metrics.append(valid_metrics)
            all_test_metrics.append(test_metrics)
            
            # Track test loss for determining best run
            test_loss = test_metrics.get('test_loss', test_metrics.get('eval_loss', float('inf')))
            run_results.append((run_idx, test_loss, y_pred, y_true, run_seed, trained_model))
        
        # Restore original seed
        self.trainer_args.seed = base_seed
        
        # Compute aggregated metrics (mean ± std)
        aggregated_valid = self._aggregate_metrics(all_valid_metrics)
        aggregated_test = self._aggregate_metrics(all_test_metrics)
        
        # Find the best run (lowest test loss)
        best_run = min(run_results, key=lambda x: x[1])
        best_run_idx, best_loss, best_y_pred, best_y_true, best_seed, best_model = best_run
        print_message(f"Best run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")
        
        # Generate plot for best run (unless skip_plot is True)
        if not skip_plot:
            output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_best.png")
            title = f"{data_name} {model_name} (best of {num_runs} runs, seed={best_seed})"
            
            if task_type == 'regression':
                regression_ci_plot(best_y_true, best_y_pred, save_path, title)
            else:
                classification_ci_plot(best_y_true, best_y_pred, save_path, title)
        
        # Return the best model along with aggregated metrics
        return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true

    def trainer_hybrid_model(
            self,
            model,
            tokenizer,
            probe,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            emb_dict=None,
            ppi=False,
            log_id=None,
            skip_plot=False,
            model_factory=None,
            probe_factory=None,
            source_model_name: Optional[str] = None,
        ):
            num_runs = getattr(self.trainer_args, 'num_runs', 1)
            base_seed = self.trainer_args.seed
            
            # Single run - original behavior
            if num_runs == 1:
                return self._train_hybrid_single_run(
                    model=model,
                    tokenizer=tokenizer,
                    probe=probe,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    test_dataset=test_dataset,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    log_id=log_id,
                    skip_plot=skip_plot,
                    source_model_name=source_model_name,
                )
            
            # Multi-run mode for hybrid probe
            # For hybrid probe, we only care about final metrics, not intermediate probe metrics
            # training_time_seconds should sum both probe and model+probe training times
            print_message(f"Running {num_runs} hybrid probe runs with different seeds for {data_name}/{model_name}")
            
            all_valid_metrics = []
            all_test_metrics = []
            run_results = []  # Store (run_idx, test_loss, y_pred, y_true, seed, model) for plotting best
            
            for run_idx in range(num_runs):
                run_seed = base_seed + run_idx
                self.trainer_args.seed = run_seed
                set_global_seed(run_seed)
                
                print_message(f"=== Hybrid Run {run_idx + 1}/{num_runs} with seed {run_seed} ===")
                
                # Create fresh probe and model for each run using factories
                if probe_factory is not None:
                    run_probe = probe_factory()
                if model_factory is not None:
                    run_model = model_factory()
                
                trained_model, valid_metrics, test_metrics, y_pred, y_true = self._train_hybrid_single_run(
                    model=run_model,
                    tokenizer=tokenizer,
                    probe=run_probe,
                    model_name=model_name,
                    data_name=data_name,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    test_dataset=test_dataset,
                    emb_dict=emb_dict,
                    ppi=ppi,
                    log_id=f"{log_id}_run{run_idx}",
                    skip_plot=True,  # Skip plots during individual runs
                    source_model_name=source_model_name,
                )
                
                # Only collect final metrics (not intermediate probe metrics)
                all_valid_metrics.append(valid_metrics)
                all_test_metrics.append(test_metrics)
                
                # Track test loss for determining best run
                test_loss = test_metrics.get('test_loss', test_metrics.get('eval_loss', float('inf')))
                run_results.append((run_idx, test_loss, y_pred, y_true, run_seed, trained_model))
            
            # Restore original seed
            self.trainer_args.seed = base_seed
            
            # Compute aggregated metrics (mean ± std)
            # This will include training_time_seconds which already has probe + base time summed per run
            aggregated_valid = self._aggregate_metrics(all_valid_metrics)
            aggregated_test = self._aggregate_metrics(all_test_metrics)
            
            # Find the best run (lowest test loss)
            best_run = min(run_results, key=lambda x: x[1])
            best_run_idx, best_loss, best_y_pred, best_y_true, best_seed, best_model = best_run
            print_message(f"Best hybrid run: {best_run_idx + 1} (seed={best_seed}, test_loss={best_loss:.4f})")
            
            # Generate plot for best run (unless skip_plot is True)
            task_type = self.probe_args.task_type
            if not skip_plot:
                output_dir = os.path.join(self.trainer_args.plots_dir, log_id)
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_best.png")
                title = f"{data_name} {model_name} hybrid (best of {num_runs} runs, seed={best_seed})"
                
                if task_type == 'regression':
                    regression_ci_plot(best_y_true, best_y_pred, save_path, title)
                else:
                    classification_ci_plot(best_y_true, best_y_pred, save_path, title)
            
            # Return the best model along with aggregated metrics
            return best_model, aggregated_valid, aggregated_test, best_y_pred, best_y_true

    def _train_hybrid_single_run(
            self,
            model,
            tokenizer,
            probe,
            model_name,
            data_name,
            train_dataset,
            valid_dataset,
            test_dataset,
            emb_dict=None,
            ppi=False,
            log_id=None,
            skip_plot=False,
            source_model_name: Optional[str] = None,
        ):
            """Single run of hybrid probe training (probe first, then model+probe)."""
            # Store original num_runs and temporarily set to 1 for the probe phase
            original_num_runs = getattr(self.trainer_args, 'num_runs', 1)
            self.trainer_args.num_runs = 1
            
            probe, _, probe_test_metrics, _, _ = self.trainer_probe(
                model=probe,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                emb_dict=emb_dict,
                ppi=ppi,
                log_id=log_id,
                skip_plot=True,  # Always skip plot for probe phase in hybrid
                source_model_name=source_model_name,
            )
            
            # Restore num_runs
            self.trainer_args.num_runs = original_num_runs
            
            probe_time = probe_test_metrics.get('training_time_seconds')
            if not isinstance(probe_time, (int, float)):
                raise ValueError(f"Probe time is not a number: {probe_time}") # ensure we are capturing the time correctly
            config = HybridProbeConfig(
                tokenwise=self.probe_args.tokenwise,
                matrix_embed=self.embedding_args.matrix_embed,
                pooling_types=self.embedding_args.pooling_types,
            )

            hybrid_model = HybridProbe(config=config, model=model, probe=probe)

            # Temporarily set num_runs to 1 for the base model phase
            self.trainer_args.num_runs = 1
            
            base_model, base_valid_metrics, base_test_metrics, y_pred, y_true = self.trainer_base_model(
                model=hybrid_model,
                tokenizer=tokenizer,
                model_name=model_name,
                data_name=data_name,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                test_dataset=test_dataset,
                ppi=ppi,
                log_id=log_id,
                skip_plot=skip_plot,
                source_model_name=source_model_name,
            )
            
            # Restore num_runs
            self.trainer_args.num_runs = original_num_runs
            
            # Sum probe time and base time for total training time
            if probe_time is not None:
                base_time = base_test_metrics.get('training_time_seconds')
                if isinstance(base_time, (int, float)):
                    base_test_metrics['training_time_seconds'] = base_time + probe_time
                elif base_time is None:
                    base_test_metrics['training_time_seconds'] = probe_time
            return base_model, base_valid_metrics, base_test_metrics, y_pred, y_true
