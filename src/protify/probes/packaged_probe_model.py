import torch

from typing import Any
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

from ..base_models.supported_models import all_presets_with_paths
from ..pooler import Pooler
from .get_probe import rebuild_probe_from_saved_config


class PackagedProbeConfig(PretrainedConfig):
    model_type = "packaged_probe"

    def __init__(
        self,
        base_model_name: str = "",
        probe_type: str = "mlp",
        probe_config: dict[str, Any] | None = None,
        tokenwise: bool = False,
        matrix_embed: bool = False,
        pooling_types: list[str] | None = None,
        task_type: str = "singlelabel",
        num_labels: int = 2,
        ppi: bool = False,
        add_token_ids: bool = False,
        sep_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.probe_type = probe_type
        self.probe_config = {} if probe_config is None else probe_config
        self.tokenwise = tokenwise
        self.matrix_embed = matrix_embed
        self.pooling_types = ["mean"] if pooling_types is None else pooling_types
        self.task_type = task_type
        self.num_labels = num_labels
        self.ppi = ppi
        self.add_token_ids = add_token_ids
        self.sep_token_id = sep_token_id


class PackagedProbeModel(PreTrainedModel):
    config_class = PackagedProbeConfig
    base_model_prefix = "backbone"
    all_tied_weights_keys = {}

    def __init__(
        self,
        config: PackagedProbeConfig,
        base_model: nn.Module | None = None,
        probe: nn.Module | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.backbone = self._load_base_model() if base_model is None else base_model
        self.probe = self._load_probe() if probe is None else probe
        self.pooler = Pooler(self.config.pooling_types)

    def _load_base_model(self) -> nn.Module:
        if self.config.base_model_name in all_presets_with_paths:
            model_path = all_presets_with_paths[self.config.base_model_name]
        else:
            model_path = self.config.base_model_name

        # Transformers 5 initializes custom architectures on the meta device
        # before loading the outer checkpoint. A nested from_pretrained call is
        # rejected in that context, and the packaged checkpoint already contains
        # the backbone weights, so build only the backbone structure on meta.
        if torch.get_default_device().type == "meta":
            backbone_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_config(backbone_config, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return model

    def _load_probe(self) -> nn.Module:
        return rebuild_probe_from_saved_config(
            probe_type=self.config.probe_type,
            tokenwise=self.config.tokenwise,
            probe_config=self.config.probe_config,
        )

    @staticmethod
    def _extract_hidden_states(backbone_output: Any) -> torch.Tensor:
        if isinstance(backbone_output, tuple):
            return backbone_output[0]  # (b, l, d)
        if hasattr(backbone_output, "last_hidden_state"):
            return backbone_output.last_hidden_state  # (b, l, d)
        if isinstance(backbone_output, torch.Tensor):
            return backbone_output  # (b, l, d)
        raise ValueError("Unsupported backbone output format for packaged probe model")

    @staticmethod
    def _extract_attentions(backbone_output: Any) -> Any | None:
        if hasattr(backbone_output, "attentions"):
            # Usually one (b, h, l, l) tensor per backbone layer.
            return backbone_output.attentions
        return None

    def _build_ppi_segment_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input_ids, attention_mask, and token_type_ids (when present): (b, l).
        if token_type_ids is not None and torch.any(token_type_ids == 1):
            mask_a = ((token_type_ids == 0) & (attention_mask == 1)).long()  # (b, l)
            mask_b = ((token_type_ids == 1) & (attention_mask == 1)).long()  # (b, l)
            assert torch.all(mask_a.sum(dim=1) > 0), "PPI token_type_ids produced empty segment A"
            assert torch.all(mask_b.sum(dim=1) > 0), "PPI token_type_ids produced empty segment B"
            return mask_a, mask_b  # each (b, l)

        assert self.config.sep_token_id is not None, "sep_token_id is required for PPI fallback segmentation"
        batch_size, seq_len = input_ids.shape
        mask_a = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)  # (b, l)
        mask_b = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)  # (b, l)

        for batch_idx in range(batch_size):
            valid_positions = torch.where(attention_mask[batch_idx] == 1)[0]  # (n_valid,)
            sep_positions = torch.where(
                (input_ids[batch_idx] == self.config.sep_token_id)
                & (attention_mask[batch_idx] == 1)
            )[0]  # (n_sep,)
            if len(valid_positions) == 0:
                continue

            # n_a and n_b are the row-local token counts assigned to each segment.
            if len(sep_positions) >= 2:
                first_sep = int(sep_positions[0].item())
                second_sep = int(sep_positions[1].item())
                mask_a[batch_idx, :first_sep + 1] = 1  # (n_a,)
                mask_b[batch_idx, first_sep + 1:second_sep + 1] = 1  # (n_b,)
            elif len(sep_positions) == 1:
                first_sep = int(sep_positions[0].item())
                mask_a[batch_idx, :first_sep + 1] = 1  # (n_a,)
                mask_b[
                    batch_idx,
                    first_sep + 1: int(valid_positions[-1].item()) + 1,
                ] = 1  # (n_b,)
            else:
                midpoint = len(valid_positions) // 2
                mask_a[batch_idx, valid_positions[:midpoint]] = 1  # (n_a,)
                mask_b[batch_idx, valid_positions[midpoint:]] = 1  # (n_b,)

        assert torch.all(mask_a.sum(dim=1) > 0), "PPI fallback segmentation produced empty segment A"
        assert torch.all(mask_b.sum(dim=1) > 0), "PPI fallback segmentation produced empty segment B"
        return mask_a, mask_b  # each (b, l)

    def _build_probe_inputs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        attentions: Any | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # hidden_states: (b, l, d); ids/masks: (b, l).
        if self.config.ppi and (not self.config.matrix_embed) and (not self.config.tokenwise):
            mask_a, mask_b = self._build_ppi_segment_masks(
                input_ids,
                attention_mask,
                token_type_ids,
            )  # each (b, l)
            # p is the configured number of pooling operations.
            vec_a = self.pooler(hidden_states, attention_mask=mask_a, attentions=attentions)  # (b, p * d)
            vec_b = self.pooler(hidden_states, attention_mask=mask_b, attentions=attentions)  # (b, p * d)
            probe_embeddings = torch.cat((vec_a, vec_b), dim=-1)  # (b, 2 * p * d)
            return probe_embeddings, None

        if self.config.matrix_embed or self.config.tokenwise:
            return hidden_states, attention_mask  # (b, l, d), (b, l)

        pooled = self.pooler(
            hidden_states,
            attention_mask=attention_mask,
            attentions=attentions,
        )  # (b, p * d)
        return pooled, None  # (b, p * d), None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput | TokenClassifierOutput:
        # input_ids, attention_mask, and token_type_ids (when present): (b, l).
        # Label shape is task- and probe-dependent.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)  # (b, l)

        requires_attentions = (
            "parti" in self.config.pooling_types
            and not self.config.matrix_embed
            and not self.config.tokenwise
        )
        backbone_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if requires_attentions:
            backbone_kwargs["output_attentions"] = True
        backbone_output = self.backbone(**backbone_kwargs)
        hidden_states = self._extract_hidden_states(backbone_output)  # (b, l, d)
        attentions = self._extract_attentions(backbone_output)
        # Backbone-specific payload, typically per-layer (b, h, l, l) tensors, or None.
        if requires_attentions:
            assert attentions is not None, "parti pooling requires base model attentions"
        probe_embeddings, probe_attention_mask = self._build_probe_inputs(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            attentions=attentions,
        )  # (b, d_probe) or (b, l, d); mask is (b, l) or None

        # Exported repositories predating the rename still carry "linear".
        if self.config.probe_type in ("mlp", "linear"):
            probe_output = self.probe(embeddings=probe_embeddings, labels=labels)
            return probe_output  # logits: (b, c)

        if self.config.probe_type == "transformer":
            forward_kwargs: dict[str, Any] = {
                "embeddings": probe_embeddings,
                "labels": labels,
            }
            if probe_attention_mask is not None:
                forward_kwargs["attention_mask"] = probe_attention_mask
            if self.config.add_token_ids and token_type_ids is not None and probe_attention_mask is not None:
                forward_kwargs["token_type_ids"] = token_type_ids
            probe_output = self.probe(**forward_kwargs)
            return probe_output  # logits: (b, c) or (b, l, c)

        probe_output = self.probe(
            embeddings=probe_embeddings,
            attention_mask=probe_attention_mask,
            labels=labels,
        )
        return probe_output  # External boundary; logits rank follows the registered probe.
