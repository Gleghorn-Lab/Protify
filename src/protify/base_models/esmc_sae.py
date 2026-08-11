"""ESMC sparse autoencoder adapters over the FastPLMs hidden-state SAE contract.

FastPLMs owns the SAE runtime: `ESMplusplusModel.load_sae_models` reads a Biohub
repository and `compute_sae=True` returns one sparse `(n, c)` tensor per attached layer,
covering the valid tokens of the whole batch. This module owns what is Protify's instead:
which checkpoints the registry offers, how a model name encodes one, how the per-residue
features pool into one vector per protein, and whether that vector is stored sparsely.

Model names carry the full checkpoint identity, for example
`ESMC-300-SAE-l23-k64-c8192`, so embedding cache filenames, result tables, and plots
separate SAE variants without any further plumbing. A bare `ESMC-300-SAE` resolves to the
depth layer each backbone publishes, at the default sparsity and codebook width.
"""

import re
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any

from .utils import ensure_fastplms_submodule_on_path, load_fastplms_model


ensure_fastplms_submodule_on_path()

from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusModel

from .esmc import ESMTokenizerWrapper, get_esmc_tokenizer


@dataclass(frozen=True)
class EsmcSaeCoverage:
    """Which sparse autoencoders Biohub publishes for one ESMC backbone."""

    biohub_name: str
    model_path: str
    hidden_size: int
    num_hidden_layers: int
    # Biohub trains every sparsity and codebook width at this layer only.
    depth_layer: int
    # Codebook widths published at every other layer, always at ALL_LAYER_K.
    all_layer_codebook_dims: tuple[int, ...]


# Protify display prefix to the published coverage. `biohub_name` and `model_path` are the
# SAE repository stem and the ESM++ backbone checkpoint the SAE reads from.
ESMC_SAE_COVERAGE: dict[str, EsmcSaeCoverage] = {
    'ESMC-300': EsmcSaeCoverage('ESMC-300M', 'Synthyra/ESMplusplus_small', 960, 30, 23, (16384,)),
    'ESMC-600': EsmcSaeCoverage('ESMC-600M', 'Synthyra/ESMplusplus_large', 1152, 36, 27, (16384,)),
    'ESMC-6B': EsmcSaeCoverage('ESMC-6B', 'Synthyra/ESMplusplus_6B', 2560, 80, 60, (16384, 131072)),
}

SAE_K_VALUES = (16, 32, 64, 128, 256, 512)
SAE_CODEBOOK_DIMS = (8192, 16384, 32768, 65536, 131072)
# Sparsity Biohub trains at every layer other than the depth layer.
ALL_LAYER_K = 64

DEFAULT_SAE_K = 64
DEFAULT_SAE_CODEBOOK_DIM = 8192

# Pooling names this adapter reduces the sparse feature tensor with. Anything else would
# need the dense (b, l, c) activation tensor, which is the cost sparse output avoids.
SAE_POOLING_TYPES = ('max', 'mean', 'sum')

# Active feature fraction, k / codebook_dim, below which pooled features stay sparse
# enough across the whole length range to store as coordinates. See
# EsmcSaeForEmbedding.sparse_storage for the measurements this sits between.
SPARSE_ACTIVE_FRACTION = 0.005

_NAME_PATTERN = re.compile(
    r'^(?P<backbone>ESMC-(?:300|600|6B))-SAE'
    r'(?:-l(?P<layer>\d+)-k(?P<k>\d+)-c(?P<codebook>\d+))?$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EsmcSaeSelection:
    """One published Biohub sparse autoencoder checkpoint."""

    backbone: str
    layer: int
    k: int
    codebook_dim: int

    @property
    def repository(self) -> str:
        coverage = ESMC_SAE_COVERAGE[self.backbone]
        return (
            f'biohub/{coverage.biohub_name}-sae'
            f'-layer{self.layer}-k{self.k}-codebook{self.codebook_dim}'
        )


def resolve_sae_selection(backbone: str, layer: int, k: int, codebook_dim: int) -> EsmcSaeSelection:
    """Validate one checkpoint request against what Biohub publishes."""
    coverage = ESMC_SAE_COVERAGE[backbone]
    if not 0 <= layer <= coverage.num_hidden_layers:
        raise ValueError(
            f'{backbone} has hidden states 0 to {coverage.num_hidden_layers}; '
            f'--sae_layer {layer} is outside that range.'
        )
    if k not in SAE_K_VALUES:
        raise ValueError(f'--sae_k {k} is not published; Biohub trains {list(SAE_K_VALUES)}.')
    if codebook_dim not in SAE_CODEBOOK_DIMS:
        raise ValueError(
            f'--sae_codebook_dim {codebook_dim} is not published; '
            f'Biohub trains {list(SAE_CODEBOOK_DIMS)}.'
        )

    if layer != coverage.depth_layer:
        # Away from the depth layer Biohub publishes one sparsity and a short width list.
        if k != ALL_LAYER_K or codebook_dim not in coverage.all_layer_codebook_dims:
            raise ValueError(
                f'{backbone} layer {layer} only publishes k={ALL_LAYER_K} at codebook widths '
                f'{list(coverage.all_layer_codebook_dims)}. Layer {coverage.depth_layer} '
                f'publishes every sparsity and width.'
            )

    return EsmcSaeSelection(backbone, layer, k, codebook_dim)


def is_sae_model_name(name: str) -> bool:
    return _NAME_PATTERN.match(name) is not None


def resolve_sae_model_name(
    name: str,
    sae_layer: int | None = None,
    sae_k: int | None = None,
    sae_codebook_dim: int | None = None,
) -> str:
    """Expand a bare SAE alias into its fully qualified name.

    A fully qualified name is returned unchanged, and conflicting overrides raise rather
    than being dropped, so a name always describes the checkpoint that produced it.
    """
    match = _NAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"{name!r} is not an ESMC SAE model name.")

    backbone = _canonical_backbone(match.group('backbone'))
    requested = {'sae_layer': sae_layer, 'sae_k': sae_k, 'sae_codebook_dim': sae_codebook_dim}

    if match.group('layer') is not None:
        declared = {
            'sae_layer': int(match.group('layer')),
            'sae_k': int(match.group('k')),
            'sae_codebook_dim': int(match.group('codebook')),
        }
        conflicts = {
            key: (value, declared[key])
            for key, value in requested.items()
            if value is not None and value != declared[key]
        }
        if conflicts:
            details = '; '.join(f"--{k}={v[0]} contradicts {v[1]}" for k, v in conflicts.items())
            raise ValueError(f"{name} already fixes its SAE checkpoint, but {details}.")

        selection = resolve_sae_selection(
            backbone, declared['sae_layer'], declared['sae_k'], declared['sae_codebook_dim']
        )
    else:
        selection = resolve_sae_selection(
            backbone,
            sae_layer if sae_layer is not None else ESMC_SAE_COVERAGE[backbone].depth_layer,
            sae_k if sae_k is not None else DEFAULT_SAE_K,
            sae_codebook_dim if sae_codebook_dim is not None else DEFAULT_SAE_CODEBOOK_DIM,
        )

    return f'{backbone}-SAE-l{selection.layer}-k{selection.k}-c{selection.codebook_dim}'


def parse_sae_model_name(name: str) -> EsmcSaeSelection:
    """Read a fully qualified SAE model name back into its checkpoint."""
    match = _NAME_PATTERN.match(name)
    if match is None or match.group('layer') is None:
        raise ValueError(
            f"{name!r} is not a fully qualified ESMC SAE model name. Expected the form "
            f"ESMC-300-SAE-l23-k64-c8192, which resolve_sae_model_name produces."
        )

    return resolve_sae_selection(
        _canonical_backbone(match.group('backbone')),
        int(match.group('layer')),
        int(match.group('k')),
        int(match.group('codebook')),
    )


def _canonical_backbone(prefix: str) -> str:
    for candidate in ESMC_SAE_COVERAGE:
        if candidate.lower() == prefix.lower():
            return candidate

    raise ValueError(f"Unknown ESMC SAE backbone prefix {prefix!r}.")


class EsmcSaeForEmbedding(nn.Module):
    """ESM++ backbone with a Biohub sparse autoencoder read off one hidden state.

    Returns pooled codebook features rather than residue embeddings, because the dense
    per-residue activation tensor is too large to store at these widths.
    """

    def __init__(
        self,
        model_path: str,
        selection: EsmcSaeSelection,
        pooling_types: tuple[str, ...] = ('max',),
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        unsupported = [name for name in pooling_types if name not in SAE_POOLING_TYPES]
        if unsupported:
            raise ValueError(
                f"SAE embeddings support pooling types {SAE_POOLING_TYPES}; received "
                f"{unsupported}. Other pooling would need the dense (b, l, c) activation "
                f"tensor, which is {selection.codebook_dim} wide per residue."
            )

        self.esm = load_fastplms_model(ESMplusplusModel, model_path, dtype=dtype)
        self.esm.load_sae_models(selection.repository, [selection.layer])
        self.selection = selection
        self.pooling_types = tuple(pooling_types)
        self.sae_key = f'layer{selection.layer}'

    @property
    def sparse_storage(self) -> bool:
        """Whether these features belong in the coordinate-list blob format.

        Per residue exactly `k` of `codebook_dim` features are active, whatever the
        sequence length. Pooling unions those sets across residues, so pooled density
        does grow with length, and how fast it grows tracks the active fraction
        `k / codebook_dim` rather than either number alone.

        Measured max-pooled density at codebook 16384, from short proteins to past 1200
        residues: 1.7% to 5.1% at k=16, 9.7% to 34.2% at k=64, and 19.6% to 56.5% at
        k=256. Only the last crosses the roughly 45% break-even. The threshold below sits
        between the k=64 and k=256 active fractions, and independently reproduces the
        codebook 8192 result at k=64, where measured storage split almost evenly between
        the two formats and dense was the better single choice.
        """
        return self.selection.k / self.selection.codebook_dim <= SPARSE_ACTIVE_FRACTION

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        hidden_state_index: int = -1,
        **kwargs: Any,
    ) -> torch.Tensor:
        # input_ids: (b, l); attention_mask: (b, l) or None
        if output_attentions:
            raise ValueError("SAE embeddings do not expose attentions, so 'parti' pooling is out.")
        if hidden_state_index not in (-1, self.selection.layer):
            raise ValueError(
                f"{self.selection.repository} reads layer {self.selection.layer}; "
                f"--embedding_hidden_state_index {hidden_state_index} contradicts it."
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (b, l)

        model_output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            compute_sae=True,
        )
        # FastPLMs returns valid tokens of the whole batch flattened, in row-major order.
        features = model_output.sae_outputs[self.sae_key].coalesce()  # sparse (n, c)
        token_counts = attention_mask.sum(dim=1).to(torch.long)  # (b,)

        pooled = [
            self._pool(features, token_counts, name)  # (b, c)
            for name in self.pooling_types
        ]
        return torch.cat(pooled, dim=-1)  # (b, p * c), p = len(pooling_types)

    def _pool(
        self, features: torch.Tensor, token_counts: torch.Tensor, pooling: str
    ) -> torch.Tensor:
        """Reduce sparse per-residue codebook features to one vector per sequence.

        The leading and trailing special tokens are dropped, matching the Biohub reference
        workflow: those positions carry activations that describe no residue.
        """
        # features: sparse (n, c) over the batch's valid tokens; token_counts: (b,)
        batch_size = int(token_counts.shape[0])  # b
        codebook_dim = int(features.shape[1])  # c
        device = features.device

        owner = torch.repeat_interleave(  # (n,) sequence index of each valid token
            torch.arange(batch_size, device=device), token_counts
        )
        sequence_start = torch.cumsum(token_counts, dim=0) - token_counts  # (b,)
        position = torch.arange(int(owner.shape[0]), device=device) - sequence_start[owner]  # (n,)
        residue_rows = (position > 0) & (position < (token_counts[owner] - 1))  # (n,)

        indices = features.indices()  # (2, nnz)
        values = features.values().float()  # (nnz,)
        keep = residue_rows[indices[0]]  # (nnz,)
        flat_positions = owner[indices[0][keep]] * codebook_dim + indices[1][keep]  # (kept,)

        pooled = torch.zeros(batch_size * codebook_dim, dtype=torch.float32, device=device)
        reduce = 'amax' if pooling == 'max' else 'sum'
        pooled.scatter_reduce_(0, flat_positions, values[keep], reduce=reduce, include_self=True)
        pooled = pooled.reshape(batch_size, codebook_dim)  # (b, c)

        if pooling == 'mean':
            # Two special tokens were dropped, so a length-2 input has no residues to average.
            residue_counts = (token_counts - 2).clamp(min=1).to(pooled.dtype)  # (b,)
            pooled = pooled / residue_counts.unsqueeze(-1)  # (b, c)

        return pooled  # (b, c)


def build_esmc_sae_model(
    preset: str,
    masked_lm: bool = False,
    dtype: torch.dtype | None = None,
    model_path: str | None = None,
    pooling_types: tuple[str, ...] | None = ('max',),
    **kwargs: Any,
) -> tuple[nn.Module, ESMTokenizerWrapper]:
    if masked_lm:
        raise ValueError("SAE models produce codebook features, not masked-language-model logits.")

    selection = parse_sae_model_name(preset)

    if pooling_types is None:
        raise ValueError(
            f"{preset} cannot produce matrix embeddings. One residue of codebook "
            f"{selection.codebook_dim} costs {4 * selection.codebook_dim // 1024} KiB "
            f"dense, so per-residue SAE features do not fit the embedding cache. Drop "
            f"--matrix_embed and pool with --embedding_pooling_types max."
        )

    backbone_path = model_path or ESMC_SAE_COVERAGE[selection.backbone].model_path
    model = EsmcSaeForEmbedding(
        backbone_path, selection, pooling_types=tuple(pooling_types), dtype=dtype
    ).eval()
    return model, get_esmc_tokenizer(preset)
