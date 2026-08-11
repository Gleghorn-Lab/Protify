from dataclasses import dataclass
from typing import List, Optional

from .supported_models import currently_supported_models, standard_models, experimental_models


def _expand_sae_name(name: str, sae_layer: Optional[int], sae_k: Optional[int], sae_codebook_dim: Optional[int]) -> str:
    """Resolve an ESMC SAE alias to a name that names its checkpoint, leaving others alone.

    The resolved name flows into embedding cache filenames and result tables, so an SAE
    variant never shares a cache with a different layer, sparsity, or codebook width.
    """
    if '-sae' not in name.lower():
        return name

    from .esmc_sae import resolve_sae_model_name
    return resolve_sae_model_name(name, sae_layer, sae_k, sae_codebook_dim)


@dataclass
class BaseModelArguments:
    def __init__(self, model_names: Optional[List[str]] = None, model_paths: Optional[List[str]] = None, model_types: Optional[List[str]] = None, model_dtype: object = None, sae_layer: Optional[int] = None, sae_k: Optional[int] = None, sae_codebook_dim: Optional[int] = None, **kwargs) -> None:
        if model_paths is not None:
            assert model_types is not None, "model_types is required when model_paths is provided."
            assert len(model_paths) == len(model_types), f"model_paths ({len(model_paths)}) and model_types ({len(model_types)}) must have the same length."
            self.model_names = [p.split('/')[-1] for p in model_paths]
            self._model_types = list(model_types)
            self._model_paths = list(model_paths)
        else:
            assert model_names is not None, "Either model_names or model_paths/model_types must be provided."
            if model_names[0] == 'standard':
                self.model_names = standard_models
            elif 'exp' in model_names[0].lower():
                self.model_names = experimental_models
            else:
                self.model_names = model_names
            self._model_types = None
            self._model_paths = None
            self.model_names = [
                _expand_sae_name(name, sae_layer, sae_k, sae_codebook_dim) for name in self.model_names
            ]
        self.model_dtype = model_dtype

    def model_entries(self):
        """Yields (display_name, dispatch_type, model_path) tuples for each model.

        In preset mode: dispatch_type is the preset name, model_path is None.
        In path mode: dispatch_type is the model type keyword, model_path is the explicit path.
        """
        if self._model_paths is not None:
            for name, mtype, mpath in zip(self.model_names, self._model_types, self._model_paths):
                yield name, mtype, mpath
        else:
            for name in self.model_names:
                yield name, name, None


def get_raw_sequence_length_limit(model_name: str, max_length: int) -> int:
    """Map a tokenizer token budget to the raw-sequence budget the dataset must respect.

    Every model except CARBON consumes one residue per token, so its raw budget is the
    token budget. CARBON reads DNA as 6-mers between two boundary tokens.
    """
    if 'carbon' in model_name.lower():
        from .carbon import carbon_dna_length_for_tokens
        return carbon_dna_length_for_tokens(max_length)
    return max_length


def resolve_data_max_length(model_names: List[str], max_length: int, trim: bool) -> int:
    """Raw-sequence limit the dataset must satisfy for every model in this run.

    Under truncation, keep the largest raw input any selected tokenizer can consume and
    let each tokenizer enforce its own token budget. Under --trim, rows are dropped rather
    than shortened, so only rows that fit every selected model may survive. An empty model
    list falls back to the token budget, since there is no tokenizer to widen it.
    """
    raw_sequence_limits = [get_raw_sequence_length_limit(name, max_length) for name in model_names]
    if not raw_sequence_limits:
        return max_length

    return min(raw_sequence_limits) if trim else max(raw_sequence_limits)


def get_base_model(model_name: str, masked_lm: bool = False, dtype=None, model_path: str = None, pooling_types=None):
    if 'random' in model_name.lower():
        from .random import build_random_model
        return build_random_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif '-sae' in model_name.lower():
        from .esmc_sae import build_esmc_sae_model
        return build_esmc_sae_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path, pooling_types=pooling_types)
    elif 'esm2' in model_name.lower() and model_name.lower().count('esm2') == 1:
        from .esm2 import build_esm2_model
        return build_esm2_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'dsm' in model_name.lower():
        from .esm2 import build_esm2_model
        return build_esm2_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'esmc' in model_name.lower():
        from .esmc import build_esmc_model
        return build_esmc_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'protbert' in model_name.lower():
        from .protbert import build_protbert_model
        return build_protbert_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'prott5' in model_name.lower():
        from .prott5 import build_prott5_model
        return build_prott5_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'ankh' in model_name.lower():
        from .ankh import build_ankh_model
        return build_ankh_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'glm' in model_name.lower():
        from .glm import build_glm2_model
        return build_glm2_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'dplm2' in model_name.lower():
        from .dplm2 import build_dplm2_model
        return build_dplm2_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'dplm' in model_name.lower():
        from .dplm import build_dplm_model
        return build_dplm_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'protclm' in model_name.lower():
        from .protCLM import build_protCLM
        return build_protCLM(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'onehot' in model_name.lower():
        from .one_hot import build_one_hot_model
        return build_one_hot_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'amplify' in model_name.lower():
        from .amplify import build_amplify_model
        return build_amplify_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'e1' in model_name.lower():
        from .e1 import build_e1_model
        return build_e1_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'calm' in model_name.lower():
        from .calm import build_calm_model
        return build_calm_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'carbon' in model_name.lower():
        from .carbon import build_carbon_model
        return build_carbon_model(model_name, masked_lm=masked_lm, dtype=dtype, model_path=model_path)
    elif 'custom' in model_name.lower():
        from .custom_model import build_custom_model
        assert model_path is not None, "model_path is required for custom models. Use --model_paths and --model_types custom."
        return build_custom_model(model_path, masked_lm=masked_lm, dtype=dtype)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_base_model_for_training(model_name: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False, dtype=None, model_path: str = None):
    if '-sae' in model_name.lower():
        raise ValueError(
            f"{model_name} reads a frozen sparse autoencoder off ESMC hidden states, so it "
            f"has no trainable path. Drop --full_finetuning and --hybrid_probe, or train the "
            f"plain backbone with --model_names {model_name.split('-SAE')[0]}."
        )
    elif 'esm2' in model_name.lower() or 'dsm' in model_name.lower():
        from .esm2 import get_esm2_for_training
        return get_esm2_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'esmc' in model_name.lower():
        from .esmc import get_esmc_for_training
        return get_esmc_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'protbert' in model_name.lower():
        from .protbert import get_protbert_for_training
        return get_protbert_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'prott5' in model_name.lower():
        from .prott5 import get_prott5_for_training
        return get_prott5_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'ankh' in model_name.lower():
        from .ankh import get_ankh_for_training
        return get_ankh_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'glm' in model_name.lower():
        from .glm import get_glm2_for_training
        return get_glm2_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'dplm2' in model_name.lower():
        from .dplm2 import get_dplm2_for_training
        return get_dplm2_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'dplm' in model_name.lower():
        from .dplm import get_dplm_for_training
        return get_dplm_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'e1' in model_name.lower():
        from .e1 import get_e1_for_training
        return get_e1_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'protclm' in model_name.lower():
        from .protCLM import get_protCLM_for_training
        return get_protCLM_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'amplify' in model_name.lower():
        from .amplify import get_amplify_for_training
        return get_amplify_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'calm' in model_name.lower():
        from .calm import get_calm_for_training
        return get_calm_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    elif 'carbon' in model_name.lower():
        from .carbon import get_carbon_for_training
        return get_carbon_for_training(model_name, tokenwise, num_labels, hybrid, dtype=dtype, model_path=model_path)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_tokenizer(model_name: str, model_path: str = None):
    if 'custom' in model_name.lower():
        from .custom_model import build_custom_tokenizer
        assert model_path is not None, "model_path is required for custom models. Use --model_paths and --model_types custom."
        return build_custom_tokenizer(model_path)
    if 'esm2' in model_name.lower() or 'random' in model_name.lower() or 'dsm' in model_name.lower():
        from .esm2 import get_esm2_tokenizer
        return get_esm2_tokenizer(model_name, model_path=model_path)
    elif 'esmc' in model_name.lower():
        from .esmc import get_esmc_tokenizer
        return get_esmc_tokenizer(model_name, model_path=model_path)
    elif 'protbert' in model_name.lower():
        from .protbert import get_protbert_tokenizer
        return get_protbert_tokenizer(model_name, model_path=model_path)
    elif 'prott5' in model_name.lower():
        from .prott5 import get_prott5_tokenizer
        return get_prott5_tokenizer(model_name, model_path=model_path)
    elif 'ankh' in model_name.lower():
        from .ankh import get_ankh_tokenizer
        return get_ankh_tokenizer(model_name, model_path=model_path)
    elif 'glm' in model_name.lower():
        from .glm import get_glm2_tokenizer
        return get_glm2_tokenizer(model_name, model_path=model_path)
    elif 'dplm2' in model_name.lower():
        from .dplm2 import get_dplm2_tokenizer
        return get_dplm2_tokenizer(model_name, model_path=model_path)
    elif 'dplm' in model_name.lower():
        from .dplm import get_dplm_tokenizer
        return get_dplm_tokenizer(model_name, model_path=model_path)
    elif 'e1' in model_name.lower():
        from .e1 import get_e1_tokenizer
        return get_e1_tokenizer(model_name, model_path=model_path)
    elif 'protclm' in model_name.lower():
        from .protCLM import get_protCLM_tokenizer
        return get_protCLM_tokenizer(model_name, model_path=model_path)
    elif 'onehot' in model_name.lower():
        from .one_hot import get_one_hot_tokenizer
        return get_one_hot_tokenizer(model_name, model_path=model_path)
    elif 'amplify' in model_name.lower():
        from .amplify import get_amplify_tokenizer
        return get_amplify_tokenizer(model_name, model_path=model_path)
    elif 'calm' in model_name.lower():
        from .calm import get_calm_tokenizer
        return get_calm_tokenizer(model_name, model_path=model_path)
    elif 'carbon' in model_name.lower():
        from .carbon import get_carbon_tokenizer
        return get_carbon_tokenizer(model_name, model_path=model_path)
    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == '__main__':
    # py -m src.protify.base_models.get_base_models
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and list supported models')
    parser.add_argument('--download', action='store_true', help='Download all standard models')
    parser.add_argument('--list', action='store_true', help='List all supported models with descriptions')
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    if args.list:
        try:
            from resource_info import model_descriptions
            print("\n=== Currently Supported Models ===\n")
            
            max_name_len = max(len(name) for name in currently_supported_models)
            max_type_len = max(len(model_descriptions.get(name, {}).get('type', 'Unknown')) for name in currently_supported_models if name in model_descriptions)
            max_size_len = max(len(model_descriptions.get(name, {}).get('size', 'Unknown')) for name in currently_supported_models if name in model_descriptions)
            
            # Print header
            print(f"{'Model':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Size':<{max_size_len+2}}Description")
            print("-" * (max_name_len + max_type_len + max_size_len + 50))
            
            for model_name in currently_supported_models:
                if model_name in model_descriptions:
                    model_info = model_descriptions[model_name]
                    print(f"{model_name:<{max_name_len+2}}{model_info.get('type', 'Unknown'):<{max_type_len+2}}{model_info.get('size', 'Unknown'):<{max_size_len+2}}{model_info.get('description', 'No description available')}")
                else:
                    print(f"{model_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_size_len+2}}No description available")
                    
            print("\n=== Standard Models ===\n")
            for model_name in standard_models:
                print(f"- {model_name}")
                
        except ImportError:
            print("Model descriptions file not found. Only listing model names.")
            print("\n=== Currently Supported Models ===\n")
            for model_name in currently_supported_models:
                print(f"- {model_name}")
                
            print("\n=== Standard Models ===\n")
            for model_name in standard_models:
                print(f"- {model_name}")
    
    if args.download:
        ### This will download all standard models
        from torchinfo import summary
        from ..utils import clear_screen
        download_args = BaseModelArguments(model_names=['standard'])
        for model_name in download_args.model_names:
            model, tokenizer = get_base_model(model_name)
            print(f'Downloaded {model_name}')
            tokenized = tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL', return_tensors='pt').input_ids
            summary(model, input_data=tokenized)
            clear_screen()
