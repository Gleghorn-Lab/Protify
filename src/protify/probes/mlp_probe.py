import torch

from typing import Any
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from ..model_components.mlp import intermediate_correction_fn
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn
    except ImportError:
        from model_components.mlp import intermediate_correction_fn
from .losses import get_loss_fct


class MLPProbeConfig(PretrainedConfig):
    # Serialized identifier from when this probe was called a linear probe. Published
    # probe repositories carry it in their config.json, so it stays as written.
    model_type = "linear_probe"

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


class MLPProbe(PreTrainedModel):
    """Feed-forward probe over a pooled embedding.

    Named for what it is: a stack of Linear, ReLU, and Dropout layers, not a single
    linear map. `--probe_type linear` still selects it.
    """

    config_class = MLPProbeConfig
    all_tied_weights_keys = {}

    def __init__(self, config: MLPProbeConfig) -> None:
        super().__init__(config)
        self.config = config
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels  # c
        use_bias = config.use_bias
        layers: list[nn.Module] = []
        if config.pre_ln:
            layers.append(nn.LayerNorm(config.input_size))
        layers.append(nn.Linear(config.input_size, config.hidden_size, bias=use_bias))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))

        for _ in range(config.n_layers):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size, bias=use_bias))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        proj_dim = intermediate_correction_fn(2, config.num_labels)
        layers.append(nn.LayerNorm(config.hidden_size))
        layers.append(nn.Linear(config.hidden_size, proj_dim, bias=use_bias))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(proj_dim, config.num_labels, bias=use_bias))
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        # embeddings: (b, d), where d is config.input_size; attention_mask is unused.
        # labels: (b,), (b, 1), or (b, c), depending on the task.
        embeddings = embeddings.to(next(self.layers.parameters()).dtype)  # (b, d)
        logits = self.layers(embeddings)  # (b, c)
        if self.task_type == 'sigmoid_regression':
            logits = logits.sigmoid()  # (b, c)

        loss = None
        if labels is not None:
            batch_size = logits.size(0)
            if self.task_type == 'regression':
                loss = self.loss_fct(
                    logits.view(-1),  # (b * c,)
                    labels.view(-1).float(),  # (b * c,)
                )  # ()
            elif self.task_type == 'sigmoid_regression':
                loss = self.loss_fct(
                    logits.view(-1),  # (b * c,)
                    labels.view(-1).float(),  # (b * c,)
                )  # ()
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(
                    logits.view(batch_size, -1),  # (b, c)
                    labels.view(batch_size, -1).float(),  # (b, c)
                )  # ()
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels),  # (b, c)
                    labels.view(-1).long(),  # (b,)
                )  # ()

        # logits: (b, c); loss: () or None.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
