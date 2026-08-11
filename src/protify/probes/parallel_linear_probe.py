import torch
import torch.nn.functional as F

from typing import Any, Sequence
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from .mlp_probe import MLPProbe, MLPProbeConfig
except ImportError:
    from probes.mlp_probe import MLPProbe, MLPProbeConfig
try:
    from ..model_components.mlp import intermediate_correction_fn
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn
    except ImportError:
        from model_components.mlp import intermediate_correction_fn


class ParallelLinearProbeConfig(PretrainedConfig):
    model_type = "parallel_linear_probe"

    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 8192,
        dropout: float = 0.2,
        num_labels: int = 2,
        n_layers: int = 1,
        task_type: str = 'singlelabel',
        pre_ln: bool = True,
        use_bias: bool = False,
        num_runs: int = 1,
        run_seeds: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.pre_ln = pre_ln
        self.use_bias = use_bias
        self.num_runs = num_runs
        if run_seeds is None:
            run_seeds = list(range(num_runs))
        assert num_runs > 0, "num_runs must be positive."
        assert len(run_seeds) == num_runs, "run_seeds length must match num_runs."
        self.run_seeds = list(run_seeds)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["num_labels"] = self.num_labels
        return output


class ParallelLinear(nn.Module):
    def __init__(
        self,
        num_runs: int,
        in_features: int,
        out_features: int,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.num_runs = num_runs
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(num_runs, out_features, in_features)
        )  # (r, d_out, d_in)
        if use_bias:
            self.bias = nn.Parameter(torch.empty(num_runs, out_features))  # (r, d_out)
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: shared (b, d_in) or run-specific (b, r, d_in).
        weight_t = self.weight.transpose(1, 2)  # (r, d_in, d_out)
        if inputs.ndim == 2:
            output = torch.matmul(inputs, weight_t).transpose(0, 1)  # (b, r, d_out)
        else:
            assert inputs.ndim == 3, f"Expected 2D or 3D input, got shape {tuple(inputs.shape)}"
            assert inputs.shape[1] == self.num_runs, (
                f"Run dimension mismatch: input has {inputs.shape[1]}, module has {self.num_runs}"
            )
            run_inputs = inputs.transpose(0, 1)  # (r, b, d_in)
            output = torch.bmm(run_inputs, weight_t).transpose(0, 1)  # (b, r, d_out)
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)  # (b, r, d_out)
        return output  # (b, r, d_out)


class ParallelLayerNorm(nn.Module):
    def __init__(
        self,
        num_runs: int,
        normalized_shape: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_runs = num_runs
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_runs, normalized_shape))  # (r, d)
        self.bias = nn.Parameter(torch.zeros(num_runs, normalized_shape))  # (r, d)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: shared (b, d) or run-specific (b, r, d).
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1).expand(-1, self.num_runs, -1)  # (b, r, d)
        assert inputs.ndim == 3, f"Expected 2D or 3D input, got shape {tuple(inputs.shape)}"
        assert inputs.shape[1] == self.num_runs, (
            f"Run dimension mismatch: input has {inputs.shape[1]}, module has {self.num_runs}"
        )
        normalized = F.layer_norm(
            inputs,
            (self.normalized_shape,),
            None,
            None,
            self.eps,
        )  # (b, r, d)
        output = normalized * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)  # (b, r, d)
        return output  # (b, r, d)


class ParallelLinearProbe(PreTrainedModel):
    config_class = ParallelLinearProbeConfig
    all_tied_weights_keys = {}

    def __init__(self, config: ParallelLinearProbeConfig) -> None:
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.num_labels = config.num_labels  # c
        self.num_runs = config.num_runs  # r
        layers: list[nn.Module] = []
        if config.pre_ln:
            layers.append(ParallelLayerNorm(config.num_runs, config.input_size))
        layers.append(
            ParallelLinear(
                config.num_runs,
                config.input_size,
                config.hidden_size,
                config.use_bias,
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))

        for _layer_idx in range(config.n_layers):
            layers.append(
                ParallelLinear(
                    config.num_runs,
                    config.hidden_size,
                    config.hidden_size,
                    config.use_bias,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        proj_dim = intermediate_correction_fn(2, config.num_labels)
        layers.append(ParallelLayerNorm(config.num_runs, config.hidden_size))
        layers.append(
            ParallelLinear(
                config.num_runs,
                config.hidden_size,
                proj_dim,
                config.use_bias,
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        layers.append(
            ParallelLinear(
                config.num_runs,
                proj_dim,
                config.num_labels,
                config.use_bias,
            )
        )
        self.layers = nn.ModuleList(layers)
        self._reset_from_linear_probes(config.run_seeds)

    def _parameter_layers(self) -> list[ParallelLinear | ParallelLayerNorm]:
        return [
            layer for layer in self.layers
            if isinstance(layer, (ParallelLinear, ParallelLayerNorm))
        ]

    def _reset_from_linear_probes(self, seeds: list[int]) -> None:
        single_config = MLPProbeConfig(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
            num_labels=self.config.num_labels,
            n_layers=self.config.n_layers,
            task_type=self.config.task_type,
            pre_ln=self.config.pre_ln,
            use_bias=self.config.use_bias,
        )
        parallel_param_layers = self._parameter_layers()
        with torch.no_grad():
            for run_idx, seed in enumerate(seeds):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    probe = MLPProbe(single_config)
                single_param_layers = [
                    layer for layer in probe.layers
                    if isinstance(layer, (nn.Linear, nn.LayerNorm))
                ]
                assert len(parallel_param_layers) == len(single_param_layers), (
                    f"Layer count mismatch: {len(parallel_param_layers)} parallel vs {len(single_param_layers)} single"
                )
                for parallel_layer, single_layer in zip(parallel_param_layers, single_param_layers):
                    if isinstance(parallel_layer, ParallelLinear):
                        assert isinstance(single_layer, nn.Linear), (
                            "Expected nn.Linear while importing seeded probe."
                        )
                        parallel_layer.weight[run_idx].copy_(single_layer.weight)  # (d_out, d_in)
                        if parallel_layer.bias is not None:
                            assert single_layer.bias is not None, "Seeded single probe is missing bias."
                            parallel_layer.bias[run_idx].copy_(single_layer.bias)  # (d_out,)
                    else:
                        assert isinstance(single_layer, nn.LayerNorm), (
                            "Expected nn.LayerNorm while importing seeded probe."
                        )
                        parallel_layer.weight[run_idx].copy_(single_layer.weight)  # (d,)
                        parallel_layer.bias[run_idx].copy_(single_layer.bias)  # (d,)

    def _expanded_continuous_labels(
        self,
        labels: torch.Tensor,
        batch_size: int,
        task_name: str,
    ) -> torch.Tensor:
        # labels: shared (b,), (b, c), or run-specific (b, r), (b, r, c).
        target = labels.float()  # same shape as labels
        if target.ndim == 3:
            assert target.shape[0] == batch_size, (
                f"Expected {task_name} run-specific labels batch size {batch_size}, got shape {tuple(target.shape)}."
            )
            assert target.shape[1] == self.num_runs, (
                f"Expected {task_name} run-specific labels to have {self.num_runs} runs, "
                f"got shape {tuple(target.shape)}."
            )
            assert target.shape[2] == self.num_labels, (
                f"Expected {task_name} run-specific labels to have {self.num_labels} labels, "
                f"got shape {tuple(target.shape)}."
            )
            return target  # (b, r, c)

        if (
            target.ndim == 2
            and target.shape[0] == batch_size
            and target.shape[1] == self.num_runs
        ):
            if self.num_labels == 1:
                return target.unsqueeze(-1)  # (b, r, 1)

        if target.ndim == 1:
            target = target.unsqueeze(-1)  # (b, 1)

        assert target.ndim == 2, (
            f"Expected shared {task_name} labels shaped [batch, labels] or run-specific labels "
            f"shaped [batch, runs, labels], got shape {tuple(target.shape)}."
        )
        assert target.shape[0] == batch_size, (
            f"Expected shared {task_name} labels batch size {batch_size}, got shape {tuple(target.shape)}."
        )
        assert target.shape[1] == self.num_labels, (
            f"Expected shared {task_name} labels to have {self.num_labels} labels, got shape {tuple(target.shape)}."
        )
        target = target.unsqueeze(1).expand(
            batch_size,
            self.num_runs,
            self.num_labels,
        )  # (b, r, c)
        return target  # (b, r, c)

    def _losses_by_run(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: (b, r, c); labels follow the task-specific shared/run-specific contracts.
        batch_size = logits.shape[0]
        if self.task_type == 'singlelabel':
            labels_have_run_dimension = (
                labels.ndim >= 2
                and labels.shape[0] == batch_size
                and labels.shape[1] == self.num_runs
            )
            if labels_have_run_dimension:
                target = labels.long()  # (b, r) or (b, r, 1)
                if target.ndim == 3:
                    assert target.shape[-1] == 1, (
                        f"Singlelabel run-specific labels must end in dim 1, got shape {tuple(target.shape)}."
                    )
                    target = target.squeeze(-1)  # (b, r)
                target = target.reshape(batch_size * self.num_runs)  # (b * r,)
            else:
                target = (
                    labels.long()
                    .view(batch_size, 1)
                    .expand(batch_size, self.num_runs)
                    .reshape(-1)
                )  # (b * r,)
            losses = F.cross_entropy(
                logits.reshape(batch_size * self.num_runs, self.num_labels),  # (b * r, c)
                target,
                reduction='none',
            )  # (b * r,)
            losses = losses.view(batch_size, self.num_runs).mean(dim=0)  # (r,)
            return losses  # (r,)

        if self.task_type == 'multilabel':
            target = self._expanded_continuous_labels(
                labels,
                batch_size,
                'multilabel',
            )  # (b, r, c)
            losses = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction='none',
            )  # (b, r, c)
            losses = losses.view(batch_size, self.num_runs, -1).mean(dim=(0, 2))  # (r,)
            return losses  # (r,)

        if self.task_type == 'regression':
            target = self._expanded_continuous_labels(
                labels,
                batch_size,
                'regression',
            )  # (b, r, c)
            losses = F.mse_loss(logits, target, reduction='none')  # (b, r, c)
            losses = losses.view(batch_size, self.num_runs, -1).mean(dim=(0, 2))  # (r,)
            return losses  # (r,)

        if self.task_type == 'sigmoid_regression':
            target = self._expanded_continuous_labels(
                labels,
                batch_size,
                'sigmoid regression',
            )  # (b, r, c)
            valid_mask = target != -100.0  # (b, r, c)
            safe_target = torch.where(
                valid_mask,
                target,
                torch.zeros_like(target),
            )  # (b, r, c)
            losses = F.binary_cross_entropy(
                logits,
                safe_target,
                reduction='none',
            )  # (b, r, c)
            losses = losses.view(batch_size, self.num_runs, -1)  # (b, r, c)
            valid_mask = valid_mask.view(batch_size, self.num_runs, -1)  # (b, r, c)
            denominators = valid_mask.float().sum(dim=(0, 2))  # (r,)
            masked_sums = (losses * valid_mask).sum(dim=(0, 2))  # (r,)
            losses_by_run = torch.where(
                denominators > 0.0,
                masked_sums / denominators.clamp_min(1.0),
                torch.zeros_like(masked_sums),
            )  # (r,)
            return losses_by_run  # (r,)

        raise ValueError(f"Task type {self.task_type} not supported by ParallelLinearProbe.")

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses_by_run = self._losses_by_run(logits, labels)  # (r,)
        if self.training:
            return losses_by_run.sum()  # ()
        return losses_by_run.mean()  # ()

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        del attention_mask
        first_param = next(self.parameters())  # Parameter tensor; only its dtype is used.
        # embeddings: shared (b, d) or run-specific (b, r, d).
        # labels: task-specific shared or run-specific targets.
        hidden = embeddings.to(first_param.dtype)  # (b, d) or (b, r, d)
        assert hidden.ndim in (2, 3), (
            "ParallelLinearProbe currently supports pooled vector embeddings or run-specific pooled batches only. "
            f"Got shape {tuple(hidden.shape)}."
        )
        if hidden.ndim == 3:
            assert hidden.shape[1] == self.num_runs, (
                f"Run-specific embeddings must have shape [batch, {self.num_runs}, dim], "
                f"got shape {tuple(hidden.shape)}."
            )
        for layer in self.layers:
            hidden = layer(hidden)  # (b, r, d_layer)
        logits = hidden  # (b, r, c)
        if self.task_type == 'sigmoid_regression':
            logits = logits.sigmoid()  # (b, r, c)
        loss = None
        if labels is not None:
            loss = self._loss(logits, labels)  # ()
        # logits: (b, r, c); loss: () or None.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def to_mlp_probe(self, run_idx: int) -> MLPProbe:
        assert 0 <= run_idx < self.num_runs, f"run_idx must be in [0, {self.num_runs})"
        config = MLPProbeConfig(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
            num_labels=self.config.num_labels,
            n_layers=self.config.n_layers,
            task_type=self.config.task_type,
            pre_ln=self.config.pre_ln,
            use_bias=self.config.use_bias,
        )
        probe = MLPProbe(config)
        parallel_param_layers = self._parameter_layers()
        single_param_layers = [
            layer for layer in probe.layers
            if isinstance(layer, (nn.Linear, nn.LayerNorm))
        ]
        assert len(parallel_param_layers) == len(single_param_layers), (
            f"Layer count mismatch: {len(parallel_param_layers)} parallel vs {len(single_param_layers)} single"
        )
        with torch.no_grad():
            for parallel_layer, single_layer in zip(parallel_param_layers, single_param_layers):
                if isinstance(parallel_layer, ParallelLinear):
                    assert isinstance(single_layer, nn.Linear), (
                        "Expected nn.Linear while exporting parallel linear layer."
                    )
                    single_layer.weight.copy_(parallel_layer.weight[run_idx])  # (d_out, d_in)
                    if parallel_layer.bias is not None:
                        assert single_layer.bias is not None, "Single linear layer is missing bias."
                        single_layer.bias.copy_(parallel_layer.bias[run_idx])  # (d_out,)
                else:
                    assert isinstance(single_layer, nn.LayerNorm), (
                        "Expected nn.LayerNorm while exporting parallel norm layer."
                    )
                    single_layer.weight.copy_(parallel_layer.weight[run_idx])  # (d,)
                    single_layer.bias.copy_(parallel_layer.bias[run_idx])  # (d,)
        return probe

    def to_ensemble(
        self,
        run_indices: Sequence[int] | None = None,
        average_mode: str = 'logits',
    ) -> "ParallelLinearProbeEnsemble":
        return ParallelLinearProbeEnsemble(
            parallel_probe=self,
            run_indices=run_indices,
            average_mode=average_mode,
        )


class ParallelLinearProbeEnsemble(nn.Module):
    """Average predictions from selected runs in a ParallelLinearProbe bank."""

    def __init__(
        self,
        parallel_probe: ParallelLinearProbe,
        run_indices: Sequence[int] | None = None,
        average_mode: str = 'logits',
    ) -> None:
        super().__init__()
        assert average_mode in ('logits', 'probabilities'), (
            "average_mode must be 'logits' or 'probabilities'."
        )
        if run_indices is None:
            run_indices = tuple(range(parallel_probe.num_runs))
        run_indices = tuple(int(run_idx) for run_idx in run_indices)
        assert len(run_indices) > 0, "ParallelLinearProbeEnsemble requires at least one run index."
        for run_idx in run_indices:
            assert 0 <= run_idx < parallel_probe.num_runs, (
                f"run index {run_idx} must be in [0, {parallel_probe.num_runs})."
            )

        self.parallel_probe = parallel_probe
        self.run_indices = run_indices
        self.average_mode = average_mode
        self.task_type = parallel_probe.task_type
        self.num_labels = parallel_probe.num_labels

    def _select_run_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (b, r, c); s is the number of selected runs.
        index_tensor = torch.as_tensor(
            self.run_indices,
            dtype=torch.long,
            device=logits.device,
        )  # (s,)
        selected_logits = logits.index_select(1, index_tensor)  # (b, s, c)
        return selected_logits  # (b, s, c)

    def _average_logits(self, run_logits: torch.Tensor) -> torch.Tensor:
        # run_logits: (b, s, c).
        if self.average_mode == 'logits':
            return run_logits.mean(dim=1)  # (b, c)
        if self.task_type == 'singlelabel':
            probabilities = F.softmax(run_logits, dim=-1)  # (b, s, c)
            return probabilities.mean(dim=1)  # (b, c)
        if self.task_type == 'multilabel':
            probabilities = torch.sigmoid(run_logits)  # (b, s, c)
            return probabilities.mean(dim=1)  # (b, c)
        return run_logits.mean(dim=1)  # (b, c)

    def _loss(self, averaged_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # averaged_logits: (b, c); labels: (b,), (b, 1), or (b, c).
        if self.task_type == 'singlelabel':
            assert self.average_mode == 'logits', (
                "Singlelabel ensemble loss is only defined for average_mode='logits'."
            )
            return F.cross_entropy(
                averaged_logits.view(-1, self.num_labels),  # (b, c)
                labels.long().view(-1),  # (b,)
            )  # ()
        if self.task_type == 'multilabel':
            target = labels.float()  # (b,) or (b, c)
            if target.ndim == 1:
                target = target.unsqueeze(-1)  # (b, 1)
            assert self.average_mode == 'logits', (
                "Multilabel ensemble loss is only defined for average_mode='logits'."
            )
            return F.binary_cross_entropy_with_logits(averaged_logits, target)  # ()
        if self.task_type == 'regression':
            target = labels.float()  # (b,) or (b, c)
            if target.ndim == 1:
                target = target.unsqueeze(-1)  # (b, 1)
            return F.mse_loss(averaged_logits, target)  # ()
        if self.task_type == 'sigmoid_regression':
            target = labels.float()  # (b,) or (b, c)
            if target.ndim == 1:
                target = target.unsqueeze(-1)  # (b, 1)
            valid_mask = target != -100.0  # (b, c)
            safe_target = torch.where(
                valid_mask,
                target,
                torch.zeros_like(target),
            )  # (b, c)
            losses = F.binary_cross_entropy(
                averaged_logits,
                safe_target,
                reduction='none',
            )  # (b, c)
            denominator = valid_mask.float().sum()  # ()
            if denominator.item() == 0:
                return torch.zeros(
                    (),
                    dtype=averaged_logits.dtype,
                    device=averaged_logits.device,
                )  # ()
            return (losses * valid_mask).sum() / denominator  # ()
        raise ValueError(f"Task type {self.task_type} not supported by ParallelLinearProbeEnsemble.")

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        # embeddings: shared (b, d) or run-specific (b, r, d).
        output = self.parallel_probe(
            embeddings=embeddings,
            attention_mask=attention_mask,
            labels=None,
        )
        run_logits = self._select_run_logits(output.logits)  # (b, s, c)
        averaged_logits = self._average_logits(run_logits)  # (b, c)
        loss = None
        if labels is not None:
            loss = self._loss(averaged_logits, labels)  # ()
        # averaged_logits: (b, c); loss: () or None.
        return SequenceClassifierOutput(
            loss=loss,
            logits=averaged_logits,
            hidden_states=None,
            attentions=None,
        )
