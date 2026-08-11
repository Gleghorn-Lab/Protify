"""Audited, DNA-only tokenizer backend for CARBON models.

The upstream CARBON tokenizer dynamically imports Qwen/Qwen3-4B-Base without a
revision. Protify only accepts raw DNA for CARBON, so this module implements the
documented DNA 6-mer path locally and loads Qwen's vocabulary from an immutable
snapshot. No model-repository Python code is executed.
"""

import itertools
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import BatchEncoding


DNA_OPEN_TOKEN = "<dna>"
DNA_CLOSE_TOKEN = "</dna>"
DNA_OOV_TOKEN = "<oov>"
DNA_KMER_SIZE = 6


class AuditedCarbonTokenizer:
    """Reproduce CARBON's pure-DNA token IDs using a pinned Qwen vocabulary."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        base_tokenizer: Any,
        base_model_id: str,
        base_revision: str,
        preprocessing_schema: str,
    ) -> None:
        self._base_tokenizer = base_tokenizer
        self.base_model_id = base_model_id
        self.base_revision = base_revision
        self.preprocessing_schema = preprocessing_schema
        self.padding_side = "right"

        self._base_vocab = self._base_tokenizer.get_vocab()
        self._base_vocab_size = len(self._base_vocab)
        if max(self._base_vocab.values()) + 1 != self._base_vocab_size:
            raise ValueError("Pinned Qwen vocabulary IDs must be contiguous")

        dna_special_tokens = [DNA_OPEN_TOKEN, DNA_CLOSE_TOKEN, DNA_OOV_TOKEN]
        kmers = [
            "".join(kmer)
            for kmer in itertools.product("ATCG", repeat=DNA_KMER_SIZE)
        ]
        base_dna_tokens = dna_special_tokens + kmers
        total_unpadded = self._base_vocab_size + len(base_dna_tokens)
        target_vocab_size = ((total_unpadded + 127) // 128) * 128
        padding_tokens = [
            f"<unused_{index}>"
            for index in range(target_vocab_size - total_unpadded)
        ]

        all_dna_tokens = base_dna_tokens + padding_tokens
        self.dna_token_to_id = {
            token: self._base_vocab_size + index
            for index, token in enumerate(all_dna_tokens)
        }
        self.dna_id_to_token = {
            token_id: token for token, token_id in self.dna_token_to_id.items()
        }
        self.dna_open_token_id = self.dna_token_to_id[DNA_OPEN_TOKEN]
        self.dna_close_token_id = self.dna_token_to_id[DNA_CLOSE_TOKEN]
        self.dna_oov_token_id = self.dna_token_to_id[DNA_OOV_TOKEN]
        self._vocab_size = target_vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return int(self._base_tokenizer.pad_token_id)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._base_tokenizer.eos_token_id

    @property
    def cls_token_id(self) -> Optional[int]:
        return self._base_tokenizer.cls_token_id

    @property
    def mask_token_id(self) -> Optional[int]:
        return self._base_tokenizer.mask_token_id

    @property
    def sep_token_id(self) -> Optional[int]:
        return self._base_tokenizer.sep_token_id

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        vocab = self._base_vocab.copy()
        for token, token_id in self.dna_token_to_id.items():
            vocab.setdefault(token, token_id)
        return vocab

    def convert_tokens_to_ids(
        self,
        tokens: Union[str, List[str]],
    ) -> Union[int, List[int]]:
        if isinstance(tokens, list):
            return [int(self.convert_tokens_to_ids(token)) for token in tokens]
        if tokens in self.dna_token_to_id:
            return self.dna_token_to_id[tokens]
        return int(self._base_tokenizer.convert_tokens_to_ids(tokens))

    def _encode_dna(self, text: str) -> List[int]:
        if not text.startswith(DNA_OPEN_TOKEN) or not text.endswith(DNA_CLOSE_TOKEN):
            raise ValueError("Audited CARBON tokenizer accepts one <dna>...</dna> region")
        dna = text[len(DNA_OPEN_TOKEN):-len(DNA_CLOSE_TOKEN)]
        if DNA_OPEN_TOKEN in dna.lower() or DNA_CLOSE_TOKEN in dna.lower():
            raise ValueError("Nested CARBON DNA boundaries are not supported")

        token_ids = [self.dna_open_token_id]
        for start in range(0, len(dna), DNA_KMER_SIZE):
            partial = dna[start:start + DNA_KMER_SIZE].upper()
            padded = partial.ljust(DNA_KMER_SIZE, "A")
            if all(base in "ATCG" for base in partial):
                token_ids.append(self.dna_token_to_id[padded])
            else:
                token_ids.append(self.dna_oov_token_id)
        token_ids.append(self.dna_close_token_id)
        return token_ids

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = False,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        pad_to_multiple_of: Optional[int] = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        if add_special_tokens:
            raise ValueError("CARBON DNA boundaries are added explicitly")
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        all_ids = [self._encode_dna(item) for item in texts]

        if truncation and max_length is not None:
            all_ids = [ids[:max_length] for ids in all_ids]

        should_pad = bool(padding)
        if should_pad:
            if padding == "max_length":
                if max_length is None:
                    raise ValueError("max_length padding requires max_length")
                target_length = max_length
            else:
                target_length = max(len(ids) for ids in all_ids)
                # Protify's 'longest' collator asks for 8-aligned batches. Honor it here so
                # CARBON batches have the same shape discipline as every other model.
                if pad_to_multiple_of:
                    remainder = target_length % pad_to_multiple_of
                    if remainder:
                        target_length += pad_to_multiple_of - remainder
        else:
            target_length = None

        padded_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        for ids in all_ids:
            if target_length is None:
                padded_ids.append(ids)
                attention_masks.append([1] * len(ids))
                continue
            if len(ids) > target_length:
                ids = ids[:target_length]
            pad_length = target_length - len(ids)
            if self.padding_side == "left":
                padded_ids.append([self.pad_token_id] * pad_length + ids)
                attention_masks.append([0] * pad_length + [1] * len(ids))
            else:
                padded_ids.append(ids + [self.pad_token_id] * pad_length)
                attention_masks.append([1] * len(ids) + [0] * pad_length)

        result: Dict[str, Any] = {
            "input_ids": padded_ids if is_batch else padded_ids[0],
            "attention_mask": attention_masks if is_batch else attention_masks[0],
        }
        if return_tensors == "pt":
            rows = padded_ids
            if len({len(row) for row in rows}) != 1:
                raise ValueError("Tensor output requires padding for variable-length batches")
            result = {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        elif return_tensors is not None:
            raise ValueError(f"Unsupported CARBON tensor type: {return_tensors}")
        return BatchEncoding(result, tensor_type=return_tensors)

    def save_pretrained(self, save_directory: str) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        saved_files = list(self._base_tokenizer.save_pretrained(save_directory))
        metadata_path = os.path.join(save_directory, "carbon_preprocessing.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "base_model_id": self.base_model_id,
                    "base_revision": self.base_revision,
                    "preprocessing_schema": self.preprocessing_schema,
                    "kmer_size": DNA_KMER_SIZE,
                    "partial_kmer_padding": "right_A",
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        return tuple(saved_files + [metadata_path])
