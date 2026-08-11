import importlib
from dataclasses import dataclass, field
from typing import Dict, List

from .mlp_probe import MLPProbe, MLPProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig
from .lyra_probe import LyraForSequenceClassification, LyraForTokenClassification, LyraConfig


# The feed-forward probe was called a linear probe before it grew hidden layers. Saved
# probe configs, published repositories, and existing YAML all carry the old value, so it
# resolves to the current name rather than failing.
_RENAMED_PROBE_TYPES = {'linear': 'mlp'}


def normalize_probe_type(probe_type: str) -> str:
    return _RENAMED_PROBE_TYPES.get(probe_type, probe_type)


# External probe types can be registered here by packages on PYTHONPATH.
# Each entry maps probe_type -> {"module": "dotted.module.path", "seq_cls": "ClassName", "tok_cls": "ClassName", "config_cls": "ConfigName"}
_EXTERNAL_PROBE_REGISTRY: Dict[str, Dict[str, str]] = {}


def register_external_probe(probe_type: str, module: str, seq_cls: str, tok_cls: str, config_cls: str):
    _EXTERNAL_PROBE_REGISTRY[probe_type] = {
        "module": module,
        "seq_cls": seq_cls,
        "tok_cls": tok_cls,
        "config_cls": config_cls,
    }


def _load_external_probe(probe_type: str, tokenwise: bool, config_kwargs: dict):
    if probe_type not in _EXTERNAL_PROBE_REGISTRY:
        return None
    entry = _EXTERNAL_PROBE_REGISTRY[probe_type]
    mod = importlib.import_module(entry["module"])
    config_cls = getattr(mod, entry["config_cls"])
    model_cls_name = entry["tok_cls"] if tokenwise else entry["seq_cls"]
    model_cls = getattr(mod, model_cls_name)
    config = config_cls(**config_kwargs)
    return model_cls(config)


@dataclass
class ProbeArguments:
    def __init__(
            self,
            probe_type: str = 'mlp', # valid options: mlp, transformer, lyra, xgboost, lightgbm, random_forest
            tokenwise: bool = False,
            ### MLP Probe
            input_size: int = 960,
            hidden_size: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
            sim_type: str = 'dot',
            use_bias: bool = False,
            add_token_ids: bool = False,
            ### Transformer Probe
            classifier_size: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            head_size: int = 128,
            rotary: bool = True,
            attention_backend: str = "flex",
            output_s_max: bool = False,
            probe_pooling_types: List[str] = field(default_factory=lambda: ['mean', 'cls']),
            bom_k: int = 60,
            expansion_ratio: float = 8 / 3,
            max_length: int = 2048,
            ### LoRA
            lora: bool = False,
            lora_r: int = 8,
            lora_alpha: float = 32.0,
            lora_dropout: float = 0.01,
            **kwargs,

    ):
        self.probe_type = normalize_probe_type(probe_type)
        self.tokenwise = tokenwise
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.add_token_ids = add_token_ids
        self.task_type = task_type
        self.pre_ln = pre_ln
        self.use_bias = use_bias
        self.classifier_size = classifier_size
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.head_size = head_size
        legacy_n_heads = kwargs.pop("n_heads", None)
        if legacy_n_heads is not None and self.hidden_size % legacy_n_heads == 0:
            self.head_size = self.hidden_size // legacy_n_heads
        self.rotary = rotary
        self.attention_backend = attention_backend
        self.output_s_max = output_s_max
        self.pooling_types = probe_pooling_types
        self.bom_k = bom_k
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_length
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def get_probe(args: ProbeArguments):
    if 'bom' in args.pooling_types:
        assert args.probe_type == 'transformer' and not args.tokenwise, (
            f"'bom' pooling is only supported by the transformer sequence probe "
            f"(probe_type='transformer', tokenwise=False). Got probe_type={args.probe_type!r}, "
            f"tokenwise={args.tokenwise}."
        )
    if args.probe_type == 'mlp' and not args.tokenwise:
        config = MLPProbeConfig(**args.__dict__)
        return MLPProbe(config)
    elif args.probe_type == 'transformer' and not args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForSequenceClassification(config)
    elif args.probe_type == 'transformer' and args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForTokenClassification(config)
    elif args.probe_type == 'lyra' and not args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForSequenceClassification(config)
    elif args.probe_type == 'lyra' and args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForTokenClassification(config)
    else:
        external = _load_external_probe(args.probe_type, args.tokenwise, args.__dict__)
        if external is not None:
            return external
        raise ValueError(f"Invalid combination of probe type and tokenwise: {args.probe_type} {args.tokenwise}")


def rebuild_probe_from_saved_config(
        probe_type: str,
        tokenwise: bool,
        probe_config: dict,
    ):
    config_dict = dict(probe_config)
    if "num_labels" not in config_dict and "id2label" in config_dict:
        config_dict["num_labels"] = len(config_dict["id2label"])
    if "pooling_types" in config_dict and "probe_pooling_types" not in config_dict:
        config_dict["probe_pooling_types"] = config_dict["pooling_types"]

    probe_type = normalize_probe_type(probe_type)

    if probe_type == "mlp" and not tokenwise:
        config = MLPProbeConfig(**config_dict)
        return MLPProbe(config)
    if probe_type == "transformer" and not tokenwise:
        config = TransformerProbeConfig(**config_dict)
        return TransformerForSequenceClassification(config)
    if probe_type == "transformer" and tokenwise:
        config = TransformerProbeConfig(**config_dict)
        return TransformerForTokenClassification(config)
    if probe_type == "lyra" and not tokenwise:
        config = LyraConfig(**config_dict)
        return LyraForSequenceClassification(config)
    if probe_type == "lyra" and tokenwise:
        config = LyraConfig(**config_dict)
        return LyraForTokenClassification(config)
    external = _load_external_probe(probe_type, tokenwise, config_dict)
    if external is not None:
        return external
    raise ValueError(f"Unsupported saved probe configuration: {probe_type} tokenwise={tokenwise}")
