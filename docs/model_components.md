# Model Components

This page is for developers who need to understand or extend the shared neural building blocks used by transformer-style probes. It summarizes [attention.py](../src/protify/model_components/attention.py), [attention_utils.py](../src/protify/model_components/attention_utils.py), [transformer.py](../src/protify/model_components/transformer.py), and [mlp.py](../src/protify/model_components/mlp.py), and which probes use them.

---

## Overview

The `model_components` package provides:

- **Attention:** Rotary embeddings, multi-head attention with pluggable backends (flex, sdpa, kernels), and attention-based logits.
- **Transformer:** A stack of transformer blocks (attention + SwiGLU FFN) used by the transformer probe.
- **MLP:** SwiGLU and feed-forward helpers used in transformer blocks and in the MLP probe (intermediate sizing).

Probes that use these components:

- **Transformer probe:** Uses `Transformer`, `LayerNorm`, and `intermediate_correction_fn` (from mlp).
- **Linear probe:** Uses only `intermediate_correction_fn` from mlp (for classifier sizing).

---

## attention.py

- **rotate_half(x, interleaved):** Rotates half the dimensions for rotary embeddings.
- **apply_rotary_emb_torch(x, cos, sin, interleaved):** Applies rotary embeddings to a tensor.
- **RotaryEmbedding(dim, base, interleaved, scaling_factor):** Module that caches cos/sin for a max sequence length.
- **MultiHeadAttention(hidden_size, n_heads, rotary, attention_backend, use_bias):** Multi-head attention with optional rotary. Dispatches to flex_attention_func, kernels_attention_func, or sdpa_attention_func from attention_utils. Supports 2D and 4D attention masks and optional BlockMask for flex. Can return attention weights and s_max. (Note: configs that wrap this module now expose `head_size` as the primary knob and derive `n_heads = hidden_size // head_size`. Passing `n_heads` to the *configs* is deprecated; the primitive itself still accepts `n_heads` directly.)
- **AttentionLogitsSequence,** **AttentionLogitsToken:** Cross-attention modules for sequence-level and token-level classification from attention over learned parameters.
- **Linear,** **LayerNorm:** Re-exports of `nn.Linear` and `nn.LayerNorm` for use across probes.

---

## attention_utils.py

- **AttentionBackend:** Enum (KERNELS, FLEX, SDPA) for the attention implementation.
- **resolve_attention_backend(name):** Returns the backend to use.
- **_repeat_kv(keys, values, n_rep):** For grouped-query style repetition.
- **sdpa_attention_func(...):** PyTorch SDPA-based attention.
- **flex_attention_func(...):** Flex attention (e.g. torch.nn.functional.flex_attention when available).
- **kernels_attention_func(...):** Custom kernel-based attention path.
- **build_attention_masks(...):** Builds 2D and 4D masks (and BlockMask for flex) from attention_mask.
- **pad_input,** **_unpad_input:** Padding and unpadding for block-sparse or kernel paths.
- **IndexFirstAxis,** **IndexPutFirstAxis:** Helpers for indexing in attention kernels.

These are used by `MultiHeadAttention` and by the transformer block when calling the chosen attention function.

---

## transformer.py

- **TransformerBlock(hidden_size, n_heads, expansion_ratio, dropout, rotary, use_bias, attention_backend):** Single block: LayerNorm, MultiHeadAttention, residual, then SwiGLU FFN (`swiglu_ln_ffn` from mlp).
- **TransformerOutput:** Dataclass extending ModelOutput (loss, logits, last_hidden_state, hidden_states, attentions, s_max).
- **Transformer(hidden_size, n_heads, n_layers, expansion_ratio, dropout, rotary, use_bias, attention_backend):** Stack of TransformerBlocks. Forward accepts hidden_states, attention_mask_2d/4d, flex_block_mask, output_attentions, output_s_max.
- **TransformerForMaskedLM:** Wrapper for masked LM; not used by the standard probes.

The **transformer probe** uses `Transformer` as the backbone above embeddings with a pooler + linear classifier.

---

## mlp.py

- **intermediate_correction_fn(hidden_size, classifier_size):** Returns an intermediate dimension (used for classifier sizing in the MLP and transformer probes).
- **SwiGLU:** Swish-gated linear unit module.
- **swiglu_ln_ffn(hidden_size, expansion_ratio, dropout, use_bias):** LayerNorm + SwiGLU feed-forward used in TransformerBlock.

---

## Dependency summary

| Component | Used by |
|-----------|---------|
| Transformer, LayerNorm, swiglu_ln_ffn | Transformer probe |
| MultiHeadAttention, RotaryEmbedding, build_attention_masks, attention backends | Transformer block (hence transformer probe) |
| AttentionLogitsSequence, AttentionLogitsToken | Cross-attention logit heads |
| intermediate_correction_fn | Linear probe, transformer probe (classifier sizing) |

---

## Performance: padding + flex attention + torch.compile

Protify defaults to a configuration optimized for maximum GPU throughput:

- **`--padding max_length`** (default): All sequences in every batch are padded to `--max_length`. This produces uniform tensor shapes across batches, which is critical for torch.compile to cache a single optimized graph instead of recompiling per unique shape.
- **`--attention_backend flex`** (default): Flex attention fuses the attention kernel and ignores padding tokens via block masks. With fixed-length tensors, the block mask structure is identical across batches, avoiding recomputation.
- **torch.compile** (`dynamic=False`): The embedding model is compiled with static shapes. Combined with max_length padding, this means a single compilation that is reused for every batch.

Together, these three settings eliminate dynamic shape overhead and produce the fastest possible inference and training.

**`--padding longest`** pads each batch to its longest sequence only, avoiding wasted compute on padding tokens. When this mode is selected, `maybe_compile()` skips torch.compile entirely because flex attention's `create_block_mask` is incompatible with compiled dynamic shapes (causes CUDA illegal memory access). Sequences are automatically sorted by length before batching to minimize padding waste within each batch.

---

## See also

- [Probes and training](probes_and_training.md) for probe types and training flows
- [Configuration](cli_and_config.md) for attention_backend and related CLI options
