import os
import sys
from typing import Any, Optional, Tuple, Type

import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel


def ensure_fastplms_submodule_on_path() -> str:
    fastplms_repository = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'fastplms',
    )
    fastplms_root = os.path.join(fastplms_repository, 'src')
    fastplms_package = os.path.join(fastplms_root, 'fastplms')
    if not os.path.isfile(os.path.join(fastplms_package, '__init__.py')):
        raise ImportError(
            "FastPLMs 1.0 source package is missing. Initialize the submodule with "
            "`git submodule update --init --recursive`."
        )

    normalized_root = os.path.normcase(os.path.abspath(fastplms_root))
    sys.path[:] = [
        path
        for path in sys.path
        if os.path.normcase(os.path.abspath(path)) != normalized_root
    ]
    sys.path.insert(0, fastplms_root)

    if "fastplms" not in sys.modules:
        return fastplms_root

    fastplms_module = sys.modules["fastplms"]
    module_locations = []
    if (
        "__file__" in fastplms_module.__dict__
        and fastplms_module.__dict__["__file__"] is not None
    ):
        module_locations.append(fastplms_module.__dict__["__file__"])
    if "__path__" in fastplms_module.__dict__:
        module_locations.extend(list(fastplms_module.__dict__["__path__"]))

    fastplms_root_abs = os.path.normcase(os.path.abspath(fastplms_package))
    loaded_from_submodule = any(
        os.path.normcase(os.path.abspath(str(location))).startswith(fastplms_root_abs)
        for location in module_locations
    )
    if loaded_from_submodule:
        return fastplms_root

    for module_name in list(sys.modules):
        if module_name == "fastplms" or module_name.startswith("fastplms."):
            del sys.modules[module_name]
    return fastplms_root


def load_fastplms_model(
    model_class: Type[Any],
    model_path: str,
    *,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "flex_attention",
    **kwargs: Any,
) -> nn.Module:
    """Load a FastPLMs 1.0 model through the Transformers attention contract."""
    if dtype not in (None, torch.float32, torch.bfloat16):
        raise ValueError(
            "FastPLMs 1.0 supports model_dtype values float32 and bfloat16; "
            f"received {dtype}."
        )

    load_kwargs = dict(kwargs)
    if dtype is not None:
        module_name = model_class.__module__
        fp32_parameter_families = (
            "fastplms.models.esm2.",
            "fastplms.models.dplm.",
            "fastplms.models.dplm2.",
        )
        load_kwargs["dtype"] = (
            torch.float32
            if dtype is torch.bfloat16
            and module_name.startswith(fp32_parameter_families)
            else dtype
        )
    load_kwargs["attn_implementation"] = attn_implementation
    return model_class.from_pretrained(model_path, **load_kwargs)


def select_hidden_state(
    last_hidden_state: torch.Tensor,
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    hidden_state_index: int,
) -> torch.Tensor:
    assert isinstance(hidden_state_index, int), "hidden_state_index must be an integer."
    if hidden_state_index == -1:
        return last_hidden_state
    assert hidden_states is not None, "hidden_state_index selection requires output_hidden_states=True."
    assert len(hidden_states) > 0, "Model returned no hidden states."
    return hidden_states[hidden_state_index]


# these modules handle ESM++ and ESM2 attention types, as well as any additional transformer blocks from Syndev
LORA_TARGET_MODULES = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]

# Llama-style projection names, which is what CARBON's backbone and head expose. PEFT
# matches target names by suffix, so adding these unconditionally would also attach
# adapters inside any other Llama-derived checkpoint loaded through --model_types custom,
# silently changing that model's trainable parameters. Opt in by model instead.
LLAMA_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# LoRA freezes the backbone, so the task head has to be unfrozen by name. 'classifier' is
# the ESM and BERT head; 'score' is the Llama sequence-classification head.
TRAINABLE_HEAD_NAME_FRAGMENTS = ("classifier", "score")


def wrap_lora(
    module: nn.Module,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    model_name: str = "",
) -> nn.Module:
    target_modules = list(LORA_TARGET_MODULES)
    if 'carbon' in model_name.lower():
        target_modules += LLAMA_LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    module = LoraModel(module, lora_config, 'default')
    for name, param in module.named_parameters():
        if any(fragment in name.lower() for fragment in TRAINABLE_HEAD_NAME_FRAGMENTS):
            param.requires_grad = True
    return module
