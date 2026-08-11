import ast
import os
import random
import re
import sqlite3
from dataclasses import dataclass
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from pandas import read_csv, read_excel

try:
    from utils import print_message, embedding_blob_to_tensor
    from seed_utils import get_global_seed
    from embedder import get_embedding_filename
    from data.dataset_classes import EmbeddingStandardizer
    from metrics_balanced import compute_sample_weights, apply_weights_from_reference, _default_bin_borders
except ImportError:
    from ..utils import print_message, embedding_blob_to_tensor
    from ..seed_utils import get_global_seed
    from ..embedder import get_embedding_filename
    from .dataset_classes import EmbeddingStandardizer
    from ..metrics_balanced import compute_sample_weights, apply_weights_from_reference, _default_bin_borders
from .supported_datasets import (
    get_dataset_source,
    resolve_dataset_name,
    standard_data_benchmark,
    supported_datasets,
    vector_benchmark,
)
from .utils import (
    AA_SET,
    CODON_SET,
    DNA_SET,
    RNA_SET,
    NONCANONICAL_AMINO_ACIDS,
    AMINO_ACID_TO_HUMAN_CODON,
    NONCANONICAL_ALANINE_CODON,
    AA_TO_CODON_TOKEN,
    CODON_TO_AA,
    DNA_CODON_TO_AA,
    RNA_CODON_TO_AA,
)


FASTA_EXTENSIONS = {'.fa', '.fasta', '.faa', '.fna', '.ffn', '.frn'}
SPLIT_ALIASES = {
    'train': ('train',),
    'valid': ('valid', 'validation', 'val', 'dev'),
    'test': ('test', 'testing'),
}


def _to_numpy(embedding: torch.Tensor) -> np.ndarray:
    """Convert an embedding to numpy, widening bfloat16 which numpy cannot represent."""
    if embedding.dtype == torch.bfloat16:
        embedding = embedding.float()
    return embedding.numpy()


def _as_rows(embedding: np.ndarray, seq: str, full: bool) -> np.ndarray:
    """Reshape one embedding into rows so concatenation yields an (n, d) matrix.

    The SQL backend returns a pooled vector at its stored rank, (d,), while the PTH
    backend already returns (1, d). Normalizing here keeps both paths building the same
    matrix.
    """
    # embedding: (d,) or (1, d) pooled, or (l, d) per-residue
    return embedding.reshape(len(seq), -1) if full else embedding.reshape(1, -1)



@dataclass
class DataArguments:
    """
    Args:
    data_paths: List[str]
        paths to the datasets
    max_length: int
        max length of sequences
    trim: bool
        whether to trim sequences to max_length
    """
    def __init__(
            self,
            data_names: List[str],
            delimiter: str = ',',
            col_names: List[str] = ['seqs', 'labels'],
            max_length: int = 1024,
            trim: bool = False,
            data_dirs: Optional[List[str]] = [],
            multi_column: Optional[List[str]] = None,
            aa_to_dna: bool = False,
            aa_to_rna: bool = False,
            dna_to_aa: bool = False,
            rna_to_aa: bool = False,
            codon_to_aa: bool = False,
            aa_to_codon: bool = False,
            **kwargs
        ):
        self.data_names = data_names
        self.data_dirs = data_dirs
        self.delimiter = delimiter
        self.col_names = col_names
        self.max_length = max_length
        self.trim = trim
        self.protein_gym = False
        self.multi_column = multi_column
        self.aa_to_dna = aa_to_dna
        self.aa_to_rna = aa_to_rna
        self.dna_to_aa = dna_to_aa
        self.rna_to_aa = rna_to_aa
        self.codon_to_aa = codon_to_aa
        self.aa_to_codon = aa_to_codon

        if len(data_names) > 0:
            if data_names[0] == 'standard_benchmark':
                self.data_paths = [get_dataset_source(data_name) for data_name in standard_data_benchmark]
            elif data_names[0] == 'vector_benchmark':
                self.data_paths = [get_dataset_source(data_name) for data_name in vector_benchmark]
            else:
                self.data_paths = []
                for data_name in data_names:
                    if data_name == 'protein_gym':
                        # For special handling in the main workflow
                        self.protein_gym = True
                        continue
                    resolved_name = resolve_dataset_name(data_name)
                    if resolved_name in supported_datasets:
                        self.data_paths.append(get_dataset_source(resolved_name))
                    else:
                        print(f'{data_name} not found in supported datasets')
                        print('We will attempt to load it from huggingface anyways, but this may not work')
                        self.data_paths.append(data_name)
        else:
            self.data_paths = []
        
        if data_dirs is not None:
            for dir in data_dirs:
                if not os.path.exists(dir):
                    raise FileNotFoundError(f'{dir} does not exist')


class DataMixin:
    def __init__(self, data_args: Optional[DataArguments] = None) -> None:
        # intialize defaults
        self.balanced_weights: Dict[str, Dict] = {}
        self._sql = False
        self._full = False
        self._max_length = 1024
        self._trim = False
        self._delimiter = ','
        self._col_names = ['seqs', 'labels']
        self._aa_to_dna = False
        self._aa_to_rna = False
        self._dna_to_aa = False
        self._rna_to_aa = False
        self._codon_to_aa = False
        self._aa_to_codon = False
        self.data_args = data_args
        self._multi_column = None if data_args is None else getattr(data_args, 'multi_column', None)
        if data_args is not None:
            self._aa_to_dna = data_args.aa_to_dna
            self._aa_to_rna = data_args.aa_to_rna
            self._dna_to_aa = data_args.dna_to_aa
            self._rna_to_aa = data_args.rna_to_aa
            self._codon_to_aa = data_args.codon_to_aa
            self._aa_to_codon = data_args.aa_to_codon

    def _not_regression(self, labels: list) -> bool: # not a great assumption but works most of the time
        if isinstance(labels, list):
            # Check if first element is itself a list (multilabel case)
            if isinstance(labels[0], list):
                # For multilabel: check all elements in all sublists
                return all(isinstance(element, (int, float)) and element == int(element) 
                          for label in labels for element in label)
            else:
                # For single label: check all elements in the list
                return all(isinstance(label, (int, float)) and label == int(label) 
                          for label in labels)
        else:
            # Fallback for non-list input
            return all(isinstance(label, (int, float)) and label == int(label) for label in labels)

    def _encode_labels(self, labels: list, tag2id: Dict[str, int]) -> List[torch.Tensor]:
        return [torch.tensor([tag2id[tag] for tag in doc], dtype=torch.long) for doc in labels]

    def _label_type_checker(self, labels: list) -> str:
        ex = labels[0]
        assert len(labels) > 0, f'Labels is empty: {labels}'
        if self._not_regression(labels):
            if isinstance(ex, list):
                label_type = 'multilabel'
            else:
                label_type = 'singlelabel' # binary or multiclass
        elif isinstance(ex, str):
            label_type = 'string'
        else:
            label_type = 'regression'
        return label_type

    def _is_sigmoid_regression(self, labels: list) -> bool:
        """Heuristic: labels within [0, 1] and cover the range approximately.
        Uses 10-bin histogram coverage and span threshold.
        """
        arr = []
        for label in labels:
            try:
                arr.extend(label)
            except:
                arr.append(label)
        arr = np.array(arr, dtype=float).flatten()        

        min_val, max_val = float(arr.min()), float(arr.max())
        cond1 = min_val > 0.0 - 1e-6 and max_val < 1.0 + 1e-6

        # Require substantial span across [0,1]
        cond2 = (max_val - min_val) > 0.75

        # Histogram coverage: at least 7 of 10 bins non-empty
        hist, _ = np.histogram(arr, bins=10, range=(0.0, 1.0))
        cond3 = int((hist > 0).sum()) >= 7

        sigmoid_regression_status = cond1 and cond2 and cond3
        return sigmoid_regression_status

    def _select_from_sql(self, c: sqlite3.Cursor, seq: str, cast_to_torch: bool = True) -> torch.Tensor:
        c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (seq,))
        raw = c.fetchone()[0]
        fallback_shape = (1, -1) if not self._full else (len(seq), -1)
        embedding = embedding_blob_to_tensor(raw, fallback_shape=fallback_shape)
        if not cast_to_torch:
            embedding = _to_numpy(embedding)
        return embedding

    def _select_from_pth(self, emb_dict: Dict[str, torch.Tensor], seq: str, cast_to_np: bool = False) -> torch.Tensor:
        embedding = emb_dict[seq].reshape(1, -1)
        if self._full:
            embedding = embedding.reshape(len(seq), -1)
        if cast_to_np:
            embedding = _to_numpy(embedding)
        return embedding

    def _labels_to_numpy(self, labels: list) -> np.ndarray:
        if isinstance(labels[0], list):
            return np.array(labels).flatten()
        else:
            return np.array([labels]).flatten()

    def _random_order(self, seq_a: str, seq_b: str) -> Tuple[str, str]:
        if random.random() < 0.5:
            return seq_a, seq_b
        else:
            return seq_b, seq_a

    def _truncate_pairs(self, ex: Dict[str, str]) -> Dict[str, str]:
        # Truncate longest first, but if that makes it shorter than the other, truncate that one
        seq_a, seq_b = ex['SeqA'], ex['SeqB']
        trunc_a, trunc_b = seq_a, seq_b
        while len(trunc_a) + len(trunc_b) > self._max_length:
            if len(trunc_a) > len(trunc_b):
                trunc_a = trunc_a[:-1]
            else:
                trunc_b = trunc_b[:-1]
        ex['SeqA'] = trunc_a
        ex['SeqB'] = trunc_b
        return ex

    def _canonical_split_for_name(self, split_name: str) -> Optional[str]:
        lowered = split_name.lower()
        for canonical_split, aliases in SPLIT_ALIASES.items():
            if lowered in aliases:
                return canonical_split
        return None

    def _resolve_split_name(self, available_splits: List[str], canonical_split: str) -> Optional[str]:
        lowercase_to_actual = {split.lower(): split for split in available_splits}
        for alias in SPLIT_ALIASES[canonical_split]:
            if alias in lowercase_to_actual:
                return lowercase_to_actual[alias]
        return None

    def _select_train_valid_test_splits(self, dataset, data_name: str) -> Tuple[Dataset, Dataset, Dataset]:
        available_splits = list(dataset.keys())
        train_name = self._resolve_split_name(available_splits, 'train')
        valid_name = self._resolve_split_name(available_splits, 'valid')
        test_name = self._resolve_split_name(available_splits, 'test')

        assert train_name is not None, f'{data_name} does not have a train set'
        assert valid_name is not None or test_name is not None, (
            f'{data_name} does not have a valid or test set, needs at least one'
        )

        seed = get_global_seed() if get_global_seed() is not None else 42
        if valid_name is None:
            train_set = dataset[train_name]
            train_valid_set = train_set.train_test_split(test_size=0.1, seed=seed + 1)
            print_message(f'{data_name} does not have a valid set, created a 10% validation set')
            return train_valid_set['train'], train_valid_set['test'], dataset[test_name]
        if test_name is None:
            train_set = dataset[train_name]
            train_test_set = train_set.train_test_split(test_size=0.1, seed=seed + 2)
            print_message(f'{data_name} does not have a test set, created a 10% test set')
            return train_test_set['train'], dataset[valid_name], train_test_set['test']

        print_message(f'{data_name} has a valid and test set')
        return dataset[train_name], dataset[valid_name], dataset[test_name]

    def _is_fasta_path(self, file_path: str) -> bool:
        suffix = os.path.splitext(file_path)[1].lower()
        return suffix in FASTA_EXTENSIONS

    def _coerce_fasta_label(self, label: str):
        if re.fullmatch(r'-?\d+', label) is not None:
            return int(label)
        if re.fullmatch(r'-?((\d+\.\d*)|(\.\d+)|(\d+))([eE][+-]?\d+)?', label) is not None:
            return float(label)
        if label.startswith('[') and label.endswith(']'):
            return ast.literal_eval(label)
        return label

    def _label_from_fasta_header(self, header: str):
        match = re.search(r'(?:^|[\s|;,])label=([^\s|;,]+)', header)
        assert match is not None, f'FASTA header is missing explicit label= metadata: {header}'
        return self._coerce_fasta_label(match.group(1))

    def _read_fasta_file(self, file_path: str) -> Dataset:
        sequences, labels = [], []
        header = None
        seq_chunks = []

        def _flush_record() -> None:
            if header is None:
                return
            sequence = ''.join(seq_chunks)
            assert len(sequence) > 0, f'FASTA record has no sequence: {header}'
            sequences.append(sequence)
            labels.append(self._label_from_fasta_header(header))

        with open(file_path, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('>'):
                    _flush_record()
                    header = line[1:].strip()
                    seq_chunks = []
                else:
                    assert header is not None, f'FASTA sequence line found before header in {file_path}'
                    seq_chunks.append(line)
            _flush_record()

        assert len(sequences) > 0, f'No FASTA records found in {file_path}'
        return Dataset.from_dict({'seqs': sequences, 'labels': labels})

    def _read_local_split_file(self, file_path: str) -> Dataset:
        suffix = os.path.splitext(file_path)[1].lower()
        if self._is_fasta_path(file_path):
            return self._read_fasta_file(file_path)
        if suffix in {'.xlsx', '.xls'}:
            frame = read_excel(file_path)
        elif suffix == '.tsv':
            frame = read_csv(file_path, delimiter='\t')
        else:
            frame = read_csv(file_path, delimiter=self._delimiter)
        return Dataset.from_pandas(frame)

    def _load_local_dataset_dir(self, data_dir: str):
        split_files = {}
        for file_path in glob(os.path.join(data_dir, '*')):
            if not os.path.isfile(file_path):
                continue
            stem = os.path.splitext(os.path.basename(file_path))[0]
            canonical_split = self._canonical_split_for_name(stem)
            if canonical_split is None:
                continue
            assert canonical_split not in split_files, (
                f'Multiple files matched {canonical_split} split in {data_dir}: '
                f'{split_files[canonical_split]} and {file_path}'
            )
            split_files[canonical_split] = file_path

        assert 'train' in split_files, f'{data_dir} does not contain a train split file'
        assert 'valid' in split_files or 'test' in split_files, (
            f'{data_dir} does not contain a valid/validation/val/dev or test/testing split file'
        )
        dataset = {
            split_name: self._read_local_split_file(file_path)
            for split_name, file_path in split_files.items()
        }
        return self._select_train_valid_test_splits(dataset, os.path.basename(os.path.normpath(data_dir)))

    def _active_translation_mode(self) -> Optional[str]:
        mode_to_flag = {
            'aa_to_dna': self._aa_to_dna,
            'aa_to_rna': self._aa_to_rna,
            'dna_to_aa': self._dna_to_aa,
            'rna_to_aa': self._rna_to_aa,
            'codon_to_aa': self._codon_to_aa,
            'aa_to_codon': self._aa_to_codon,
        }
        active_modes = [mode for mode, enabled in mode_to_flag.items() if enabled]
        assert len(active_modes) <= 1, f'Only one translation mode can be enabled at a time, found: {active_modes}'
        return active_modes[0] if len(active_modes) == 1 else None

    def _assert_characters_in_set(self, seq: str, allowed_chars: set, mode: str) -> None:
        bad_chars = sorted({char for char in seq if char.upper() not in allowed_chars})
        assert len(bad_chars) == 0, f'Invalid characters for {mode}: {bad_chars}.'

    def _validate_translated_output(self, translated_seq: str, allowed_chars: set, mode: str) -> None:
        bad_chars = sorted({char for char in translated_seq if char not in allowed_chars})
        assert len(bad_chars) == 0, f'Translation output for {mode} contains unexpected characters: {bad_chars}.'

    def _normalize_aa_for_nucleotide_translation(self, seq: str) -> str:
        canonical_aas = set(AMINO_ACID_TO_HUMAN_CODON.keys())
        normalized = []
        for residue in seq:
            residue = residue.upper()
            if residue in canonical_aas:
                normalized.append(residue)
            else:
                normalized.append('X')
        return ''.join(normalized)

    def _translate_aa_to_dna(self, seq: str) -> str:
        seq = self._normalize_aa_for_nucleotide_translation(seq)
        dna_codons: List[str] = []
        for residue in seq:
            residue = residue.upper()
            if residue in AMINO_ACID_TO_HUMAN_CODON:
                dna_codons.append(AMINO_ACID_TO_HUMAN_CODON[residue])
            elif residue in NONCANONICAL_AMINO_ACIDS:
                dna_codons.append(NONCANONICAL_ALANINE_CODON)
            else:
                raise AssertionError(f'Unexpected amino acid token "{residue}" while converting aa_to_dna.')
        translated = ''.join(dna_codons)
        self._validate_translated_output(translated, DNA_SET, 'aa_to_dna')
        return translated

    def _translate_aa_to_rna(self, seq: str) -> str:
        dna_translated = self._translate_aa_to_dna(seq)
        translated = dna_translated.replace('T', 'U')
        self._validate_translated_output(translated, RNA_SET, 'aa_to_rna')
        return translated

    def _translate_dna_to_aa(self, seq: str) -> str:
        dna_seq = seq.upper()
        self._assert_characters_in_set(dna_seq, DNA_SET, 'dna_to_aa')
        assert len(dna_seq) % 3 == 0, f'dna_to_aa requires sequence length multiple of 3, got {len(dna_seq)}.'
        aa_seq = []
        for idx in range(0, len(dna_seq), 3):
            codon = dna_seq[idx:idx + 3]
            assert codon in DNA_CODON_TO_AA, f'Unknown DNA codon for dna_to_aa: {codon}'
            translated_char = DNA_CODON_TO_AA[codon]
            if translated_char != '*':
                aa_seq.append(translated_char)
        translated = ''.join(aa_seq)
        self._validate_translated_output(translated, AA_SET - {'*'}, 'dna_to_aa')
        return translated

    def _translate_rna_to_aa(self, seq: str) -> str:
        rna_seq = seq.upper()
        self._assert_characters_in_set(rna_seq, RNA_SET, 'rna_to_aa')
        assert len(rna_seq) % 3 == 0, f'rna_to_aa requires sequence length multiple of 3, got {len(rna_seq)}.'
        aa_seq = []
        for idx in range(0, len(rna_seq), 3):
            codon = rna_seq[idx:idx + 3]
            assert codon in RNA_CODON_TO_AA, f'Unknown RNA codon for rna_to_aa: {codon}'
            translated_char = RNA_CODON_TO_AA[codon]
            if translated_char != '*':
                aa_seq.append(translated_char)
        translated = ''.join(aa_seq)
        self._validate_translated_output(translated, AA_SET - {'*'}, 'rna_to_aa')
        return translated

    def _translate_codon_to_aa(self, seq: str) -> str:
        aa_seq = []
        for token in seq:
            assert token in CODON_TO_AA, f'Unknown codon token for codon_to_aa: {token}'
            translated_char = CODON_TO_AA[token]
            if translated_char != '*':
                aa_seq.append(translated_char)
        translated = ''.join(aa_seq)
        self._validate_translated_output(translated, AA_SET - {'*'}, 'codon_to_aa')
        return translated

    def _translate_aa_to_codon(self, seq: str) -> str:
        codon_tokens = []
        for residue in seq:
            residue = residue.upper()
            if residue in AA_TO_CODON_TOKEN:
                codon_tokens.append(AA_TO_CODON_TOKEN[residue])
            elif residue in NONCANONICAL_AMINO_ACIDS:
                codon_tokens.append(AA_TO_CODON_TOKEN['A'])
            else:
                raise AssertionError(f'Unexpected amino acid token "{residue}" while converting aa_to_codon.')
        translated = ''.join(codon_tokens)
        self._validate_translated_output(translated, CODON_SET, 'aa_to_codon')
        return translated

    def _translate_sequence_for_mode(self, seq: str, mode: str) -> str:
        if mode == 'aa_to_dna':
            return self._translate_aa_to_dna(seq)
        if mode == 'aa_to_rna':
            return self._translate_aa_to_rna(seq)
        if mode == 'dna_to_aa':
            return self._translate_dna_to_aa(seq)
        if mode == 'rna_to_aa':
            return self._translate_rna_to_aa(seq)
        if mode == 'codon_to_aa':
            return self._translate_codon_to_aa(seq)
        if mode == 'aa_to_codon':
            return self._translate_aa_to_codon(seq)
        raise AssertionError(f'Unsupported translation mode: {mode}')

    def _find_first_present_column(self, available_columns: List[str], candidates_ordered: List[str]) -> str:
        """Return the first column from candidates_ordered that exists in available_columns (case-insensitive)."""
        lowercase_to_actual = {col.lower(): col for col in available_columns}
        for candidate in candidates_ordered:
            actual = lowercase_to_actual.get(candidate.lower())
            if actual is not None:
                return actual
        raise KeyError(f"None of the candidate columns were found. Candidates: {candidates_ordered}. Available: {available_columns}")

    def _is_ppi_from_columns(self, available_columns: List[str]) -> bool:
        """Detect if dataset contains paired sequence inputs (SeqA/SeqB variants)."""
        lowercase_columns = set(col.lower() for col in available_columns)
        base_candidates = ['seqs', 'seq', 'sequence', 'sequences']
        for base in base_candidates:
            if (base + 'a') in lowercase_columns and (base + 'b') in lowercase_columns:
                return True
        return False

    def _find_ppi_sequence_columns(self, available_columns: List[str]) -> Tuple[str, str]:
        """Return the actual column names for A and B sequences in PPI datasets based on priority."""
        lowercase_to_actual = {col.lower(): col for col in available_columns}
        # Try specific common pairs first (in order)
        specific_pairs = [
            ('SeqA', 'SeqB'),
            ('seqa', 'seqb'),
            ('SeqsA', 'SeqsB'),
        ]
        for cand_a, cand_b in specific_pairs:
            a_actual = lowercase_to_actual.get(cand_a.lower())
            b_actual = lowercase_to_actual.get(cand_b.lower())
            if a_actual is not None and b_actual is not None:
                return a_actual, b_actual

        # Generalized search using base tokens
        base_candidates = ['seqs', 'seq', 'sequence', 'sequences']
        for base in base_candidates:
            a_key = (base + 'a').lower()
            b_key = (base + 'b').lower()
            a_actual = lowercase_to_actual.get(a_key)
            b_actual = lowercase_to_actual.get(b_key)
            if a_actual is not None and b_actual is not None:
                return a_actual, b_actual

        raise KeyError(f"Could not find paired sequence columns for PPI. Available: {available_columns}")

    def _is_missing_value(self, v: object) -> bool:
        if v is None:
            return True
        # float/np.nan handling
        try:
            if isinstance(v, float) and np.isnan(v):
                return True
        except Exception:
            pass
        # list/array handling (check any element)
        if isinstance(v, (list, tuple, np.ndarray)):
            for el in v:
                if el is None:
                    return True
                if isinstance(el, float) and np.isnan(el):
                    return True
        return False

    def _compute_balanced_weights_for(self, data_name, train_set, valid_set, test_set) -> None:
        """Pre-compute EpHod-style sample weights for train/valid/test once up front."""
        if 'balanced_weights' not in self.__dict__:
            self.balanced_weights = {}
        if 'trainer_args' not in self.__dict__ or self.trainer_args is None:
            return
        ta = self.trainer_args
        if 'balanced_regression_metrics' not in ta.__dict__ or not ta.balanced_regression_metrics:
            return

        def _extract(split):
            labels = np.asarray(list(split['labels']), dtype=object)
            flat = []
            for item in labels:
                if isinstance(item, (list, tuple, np.ndarray)):
                    arr = np.asarray(item).flatten()
                    flat.extend([float(v) for v in arr if v != -100])
                else:
                    flat.append(float(item))
            return np.asarray(flat, dtype=np.float64)

        train_labels = _extract(train_set)
        valid_labels = _extract(valid_set)
        test_labels = _extract(test_set)

        if len(train_labels) < 2 or float(np.max(train_labels)) == float(np.min(train_labels)):
            print_message(f'Skipping balanced weights for {data_name} (degenerate label distribution)')
            return

        bin_borders = ta.balanced_bin_borders if ta.balanced_bin_borders else _default_bin_borders(train_labels)
        method = ta.balanced_weight_method

        train_w = compute_sample_weights(
            train_labels,
            method=method,
            bin_borders=bin_borders,
            lds_bins=ta.balanced_lds_bins,
            lds_ks=ta.balanced_lds_ks,
            lds_sigma=ta.balanced_lds_sigma,
        )
        valid_w = apply_weights_from_reference(
            valid_labels, train_labels,
            method=method, bin_borders=bin_borders,
            lds_bins=ta.balanced_lds_bins, lds_ks=ta.balanced_lds_ks, lds_sigma=ta.balanced_lds_sigma,
        )
        test_w = apply_weights_from_reference(
            test_labels, train_labels,
            method=method, bin_borders=bin_borders,
            lds_bins=ta.balanced_lds_bins, lds_ks=ta.balanced_lds_ks, lds_sigma=ta.balanced_lds_sigma,
        )

        self.balanced_weights[data_name] = {
            'train': train_w,
            'valid': valid_w,
            'test': test_w,
            'bin_borders': list(bin_borders),
            'method': method,
        }
        print_message(
            f'Balanced weights for {data_name}: method={method} bin_borders={bin_borders} '
            f'sizes train/valid/test={len(train_w)}/{len(valid_w)}/{len(test_w)}'
        )

    def process_datasets(
            self,
            hf_datasets: List[Tuple[Dataset, Dataset, Dataset, bool]],
            data_names: List[str],
        )-> Tuple[Dict[str, Tuple[Dataset, Dataset, Dataset, int, str, bool]], List[str]]:
        max_length = self._max_length
        datasets, all_seqs = {}, set()
        translation_mode = self._active_translation_mode()
        for dataset, data_name in zip(hf_datasets, data_names):
            print_message(f'Processing {data_name}')
            train_set, valid_set, test_set, ppi = dataset
            print(train_set)
            print(valid_set)
            print(test_set)
            ### sanitize
            # 1) Drop rows with None or NaN in any sequence column(s) or labels
            before_train, before_valid, before_test = len(train_set), len(valid_set), len(test_set)
            if ppi:
                train_set = train_set.filter(lambda x: not (self._is_missing_value(x['SeqA']) or self._is_missing_value(x['SeqB']) or self._is_missing_value(x['labels'])))
                valid_set = valid_set.filter(lambda x: not (self._is_missing_value(x['SeqA']) or self._is_missing_value(x['SeqB']) or self._is_missing_value(x['labels'])))
                test_set = test_set.filter(lambda x: not (self._is_missing_value(x['SeqA']) or self._is_missing_value(x['SeqB']) or self._is_missing_value(x['labels'])))
            elif self.data_args.multi_column:
                cols = self.data_args.multi_column
                # assert columns exist
                for col in cols:
                    assert col in train_set.column_names or col in valid_set.column_names or col in test_set.column_names, f"Column {col} not found in dataset {data_name}"

                def _filter_row(x):
                    return (not self._is_missing_value(x['labels'])) and all(not self._is_missing_value(x[col]) for col in cols)

                train_set = train_set.filter(_filter_row)
                valid_set = valid_set.filter(_filter_row)
                test_set = test_set.filter(_filter_row)
            else:
                train_set = train_set.filter(lambda x: not (self._is_missing_value(x['seqs']) or self._is_missing_value(x['labels'])))
                valid_set = valid_set.filter(lambda x: not (self._is_missing_value(x['seqs']) or self._is_missing_value(x['labels'])))
                test_set = test_set.filter(lambda x: not (self._is_missing_value(x['seqs']) or self._is_missing_value(x['labels'])))
            if any([
                len(train_set) != before_train,
                len(valid_set) != before_valid,
                len(test_set) != before_test,
            ]):
                print_message(
                    f"Removed None / NaN rows - train: {before_train - len(train_set)}, valid: {before_valid - len(valid_set)}, test: {before_test - len(test_set)}"
                )

            # 2) Legacy sanitization for non-translation workflows
            if translation_mode is None:
                if ppi:
                    train_set = train_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AA_SET),
                                                         'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AA_SET)})
                    valid_set = valid_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AA_SET),
                                                         'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AA_SET)})
                    test_set = test_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AA_SET),
                                                        'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AA_SET)})
                elif self.data_args.multi_column:
                    cols = self.data_args.multi_column
                    for col in cols:
                        train_set = train_set.map(lambda x, _col=col: {_col: ''.join(aa for aa in x[_col] if aa in AA_SET)})
                        valid_set = valid_set.map(lambda x, _col=col: {_col: ''.join(aa for aa in x[_col] if aa in AA_SET)})
                        test_set = test_set.map(lambda x, _col=col: {_col: ''.join(aa for aa in x[_col] if aa in AA_SET)})
                else:
                    train_set = train_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AA_SET)})
                    valid_set = valid_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AA_SET)})
                    test_set = test_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AA_SET)})

            # 3) Remove any length 0 sequences
            before_train, before_valid, before_test = len(train_set), len(valid_set), len(test_set)
            if ppi:
                train_set = train_set.filter(lambda x: len(x['SeqA']) > 0 and len(x['SeqB']) > 0)
                valid_set = valid_set.filter(lambda x: len(x['SeqA']) > 0 and len(x['SeqB']) > 0)
                test_set = test_set.filter(lambda x: len(x['SeqA']) > 0 and len(x['SeqB']) > 0)
            elif self.data_args.multi_column:
                cols = self.data_args.multi_column
                train_set = train_set.filter(lambda x: all(len(x[col]) > 0 for col in cols))
                valid_set = valid_set.filter(lambda x: all(len(x[col]) > 0 for col in cols))
                test_set = test_set.filter(lambda x: all(len(x[col]) > 0 for col in cols))
            else:
                train_set = train_set.filter(lambda x: len(x['seqs']) > 0)
                valid_set = valid_set.filter(lambda x: len(x['seqs']) > 0)
                test_set = test_set.filter(lambda x: len(x['seqs']) > 0)

            if any([
                len(train_set) != before_train,
                len(valid_set) != before_valid,
                len(test_set) != before_test,
            ]): 
                print_message(
                    f"Removed length 0 rows - train: {before_train - len(train_set)}, valid: {before_valid - len(valid_set)}, test: {before_test - len(test_set)}"
                )

            # 4) Trim or truncate by length if necessary
            before_train, before_valid, before_test = len(train_set), len(valid_set), len(test_set)
            if self._trim: # trim by length
                if ppi:
                    train_set = train_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                    valid_set = valid_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                    test_set = test_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                elif self.data_args.multi_column:
                    cols = self.data_args.multi_column
                    train_set = train_set.filter(lambda x: all(len(x[col]) <= max_length for col in cols))
                    valid_set = valid_set.filter(lambda x: all(len(x[col]) <= max_length for col in cols))
                    test_set = test_set.filter(lambda x: all(len(x[col]) <= max_length for col in cols))
                else:
                    train_set = train_set.filter(lambda x: len(x['seqs']) <= max_length)
                    valid_set = valid_set.filter(lambda x: len(x['seqs']) <= max_length)
                    test_set = test_set.filter(lambda x: len(x['seqs']) <= max_length)

            else: # truncate to max_length
                if ppi:
                    train_set = train_set.map(self._truncate_pairs)
                    valid_set = valid_set.map(self._truncate_pairs)
                    test_set = test_set.map(self._truncate_pairs)
                elif self.data_args.multi_column:
                    cols = self.data_args.multi_column
                    for col in cols:
                        train_set = train_set.map(lambda x, _col=col: { _col: x[_col][:max_length] })
                        valid_set = valid_set.map(lambda x, _col=col: { _col: x[_col][:max_length] })
                        test_set = test_set.map(lambda x, _col=col: { _col: x[_col][:max_length] })
                else:
                    train_set = train_set.map(lambda x: {'seqs': x['seqs'][:max_length]})
                    valid_set = valid_set.map(lambda x: {'seqs': x['seqs'][:max_length]})
                    test_set = test_set.map(lambda x: {'seqs': x['seqs'][:max_length]})

            if any([
                len(train_set) != before_train,
                len(valid_set) != before_valid,
                len(test_set) != before_test,
            ]):
                print_message(
                    f"Trimmed rows - train: {(before_train - len(train_set)) / before_train * 100:.2f}%, \
                    valid: {(before_valid - len(valid_set)) / before_valid * 100:.2f}%, \
                    test: {(before_test - len(test_set)) / before_test * 100:.2f}%"
                )

            # 5) Optional sequence translation (post-trim/truncate)
            if translation_mode is not None:
                if ppi:
                    train_set = train_set.map(lambda x: {'SeqA': self._translate_sequence_for_mode(x['SeqA'], translation_mode),
                                                         'SeqB': self._translate_sequence_for_mode(x['SeqB'], translation_mode)})
                    valid_set = valid_set.map(lambda x: {'SeqA': self._translate_sequence_for_mode(x['SeqA'], translation_mode),
                                                         'SeqB': self._translate_sequence_for_mode(x['SeqB'], translation_mode)})
                    test_set = test_set.map(lambda x: {'SeqA': self._translate_sequence_for_mode(x['SeqA'], translation_mode),
                                                       'SeqB': self._translate_sequence_for_mode(x['SeqB'], translation_mode)})
                elif self.data_args.multi_column:
                    cols = self.data_args.multi_column
                    for col in cols:
                        train_set = train_set.map(lambda x, _col=col: {_col: self._translate_sequence_for_mode(x[_col], translation_mode)})
                        valid_set = valid_set.map(lambda x, _col=col: {_col: self._translate_sequence_for_mode(x[_col], translation_mode)})
                        test_set = test_set.map(lambda x, _col=col: {_col: self._translate_sequence_for_mode(x[_col], translation_mode)})
                else:
                    train_set = train_set.map(lambda x: {'seqs': self._translate_sequence_for_mode(x['seqs'], translation_mode)})
                    valid_set = valid_set.map(lambda x: {'seqs': self._translate_sequence_for_mode(x['seqs'], translation_mode)})
                    test_set = test_set.map(lambda x: {'seqs': self._translate_sequence_for_mode(x['seqs'], translation_mode)})
                print_message(f"Translated sequences with mode {translation_mode} (post-trim/truncate).")

            # 6) Record all_seqs
            if ppi:
                all_seqs.update(list(train_set['SeqA']) + list(train_set['SeqB']))
                all_seqs.update(list(valid_set['SeqA']) + list(valid_set['SeqB']))
                all_seqs.update(list(test_set['SeqA']) + list(test_set['SeqB']))
            elif self.data_args.multi_column:
                cols = self.data_args.multi_column
                for col in cols:
                    all_seqs.update(list(train_set[col]))
                    all_seqs.update(list(valid_set[col]))
                    all_seqs.update(list(test_set[col]))
            else:
                all_seqs.update(list(train_set['seqs']))
                all_seqs.update(list(valid_set['seqs']))
                all_seqs.update(list(test_set['seqs']))

            # confirm the type of labels
            check_labels = list(train_set['labels'])
            label_type = self._label_type_checker(check_labels)

            if label_type == 'string': # might be string or multilabel
                example = list(valid_set['labels'])[0]
                try:
                    import ast
                    new_ex = ast.literal_eval(example)
                    if isinstance(new_ex, list): # if ast runs correctly and is now a list it is multilabel labels
                        label_type = 'multilabel'
                        train_set = train_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                        valid_set = valid_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                        test_set = test_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                except Exception:
                    label_type = 'string' # if ast throws error it is actually string

            if label_type == 'string': # if still string, it's for tokenwise classification
                train_labels = list(train_set['labels'])
                unique_tags = set(tag for doc in train_labels for tag in doc)
                tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
                # add cls token to labels
                train_set = train_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
                valid_set = valid_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
                test_set = test_set.map(lambda ex: {'labels': self._encode_labels(ex['labels'], tag2id=tag2id)})
                label_type = 'tokenwise'
                num_labels = len(unique_tags)
            else:
                if label_type == 'regression':
                    # Detect sigmoid_regression (values in [0,1] covering the range)
                    if self._is_sigmoid_regression(list(train_set['labels'])):
                        label_type = 'sigmoid_regression'
                    num_labels = 1
                elif label_type == 'multilabel':
                    train_labels_list = list(train_set['labels'])
                    label_lengths = set(len(lab) for lab in train_labels_list)
                    if len(label_lengths) > 1:
                        # Variable-length sparse index lists: convert to multi-hot
                        all_labels = (
                            list(train_set['labels'])
                            + list(valid_set['labels'])
                            + list(test_set['labels'])
                        )
                        num_labels = max(idx for lab in all_labels for idx in lab) + 1
                        def _to_multihot(ex, n=num_labels):
                            vec = [0] * n
                            for idx in ex['labels']:
                                vec[idx] = 1
                            return {'labels': vec}
                        train_set = train_set.map(_to_multihot)
                        valid_set = valid_set.map(_to_multihot)
                        test_set = test_set.map(_to_multihot)
                    else:
                        num_labels = len(train_labels_list[0])
                else: # singlelabel classification
                    try:
                        train_labels_list = list(train_set['labels'])
                        num_labels = len(train_labels_list[0])
                    except:
                        unique = np.unique(list(train_set['labels']))
                        max_label = max(unique) # sometimes there are missing labels
                        full_list = np.arange(0, max_label+1)
                        num_labels = len(full_list)
            datasets[data_name] = (train_set, valid_set, test_set, num_labels, label_type, ppi)

            if label_type in ('regression', 'sigmoid_regression'):
                self._compute_balanced_weights_for(data_name, train_set, valid_set, test_set)

        print(f'Label type: {label_type}')
        print(f'Number of labels: {num_labels}')

        all_seqs = list(all_seqs)
        all_seqs = sorted(all_seqs, key=len, reverse=True) # longest first
        return datasets, all_seqs

    def get_data(self):
        """
        Supports tabular local splits and labeled FASTA files.
        """
        datasets, data_names = [], []
        label_candidates = ['labels', 'label', 'Labels', 'Label']
        seq_candidates = ['seqs', 'Seqs', 'seq', 'Seq', 'sequence', 'Sequence', 'sequences', 'Sequences']

        for data_path in self.data_args.data_paths:
            data_name = data_path.split('/')[-1]
            print_message(f'Loading {data_name}')
            dataset = load_dataset(data_path)
            if 'inverse' in data_name.lower():
                dataset = dataset.rename_columns({'seqs': 'labels', 'labels': 'seqs'})
            train_name = self._resolve_split_name(list(dataset.keys()), 'train')
            assert train_name is not None, f'{data_name} does not have a train set'
            ppi = 'SeqA' in dataset[train_name].column_names
            # Fallback PPI detection based on available columns
            if not ppi:
                ppi = self._is_ppi_from_columns(dataset[train_name].column_names)
            print_message(f'PPI (or dual sequence input dataset): {ppi}')

            train_set, valid_set, test_set = self._select_train_valid_test_splits(dataset, data_name)

            print_message(f'Train set: {len(train_set)}, Valid set: {len(valid_set)}, Test set: {len(test_set)}')
            if ppi:
                # Standardize PPI columns to 'SeqA', 'SeqB', and 'labels'
                print('Standardizing PPI column names')
                try:
                    a_col, b_col = self._find_ppi_sequence_columns(train_set.column_names)
                except KeyError:
                    # Retry with validation/test in case train is empty or missing columns
                    try:
                        a_col, b_col = self._find_ppi_sequence_columns(valid_set.column_names)
                    except KeyError:
                        a_col, b_col = self._find_ppi_sequence_columns(test_set.column_names)
                
                try:
                    lbl_col = self._find_first_present_column(train_set.column_names, label_candidates)
                except KeyError:
                    try:
                        lbl_col = self._find_first_present_column(valid_set.column_names, label_candidates)
                    except KeyError:
                        lbl_col = self._find_first_present_column(test_set.column_names, label_candidates)

                train_set = train_set.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'})
                valid_set = valid_set.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'})
                test_set = test_set.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'})

                print('Removing extras')
                train_set = train_set.remove_columns([col for col in train_set.column_names if col not in ['SeqA', 'SeqB', 'labels']])
                valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in ['SeqA', 'SeqB', 'labels']])
                test_set = test_set.remove_columns([col for col in test_set.column_names if col not in ['SeqA', 'SeqB', 'labels']])
            else:
                print('Standardizing column names')
                use_multi = self.data_args.multi_column is not None
                if not use_multi:
                    try:
                        seq_col = self._find_first_present_column(train_set.column_names, seq_candidates)
                    except KeyError:
                        try:
                            seq_col = self._find_first_present_column(valid_set.column_names, seq_candidates)
                        except KeyError:
                            seq_col = self._find_first_present_column(test_set.column_names, seq_candidates)

                try:
                    label_col = self._find_first_present_column(train_set.column_names, label_candidates)
                except KeyError:
                    try:
                        label_col = self._find_first_present_column(valid_set.column_names, label_candidates)
                    except KeyError:
                        label_col = self._find_first_present_column(test_set.column_names, label_candidates)

                # Always standardize label column to 'labels'
                train_set = train_set.rename_columns({label_col: 'labels'})
                valid_set = valid_set.rename_columns({label_col: 'labels'})
                test_set = test_set.rename_columns({label_col: 'labels'})

                if not use_multi:
                    train_set = train_set.rename_columns({seq_col: 'seqs'})
                    valid_set = valid_set.rename_columns({seq_col: 'seqs'})
                    test_set = test_set.rename_columns({seq_col: 'seqs'})
                    # drop everything else
                    print('Removing extras')
                    train_set = train_set.remove_columns([col for col in train_set.column_names if col not in ['seqs', 'labels']])
                    valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in ['seqs', 'labels']])
                    test_set = test_set.remove_columns([col for col in test_set.column_names if col not in ['seqs', 'labels']])
                else:
                    # Validate requested multi columns exist (assert exact match)
                    for col in self.data_args.multi_column:
                        assert col in train_set.column_names or col in valid_set.column_names or col in test_set.column_names, f"Column {col} not found in dataset {data_name}"
                    # Keep only requested columns and labels
                    keep_cols = set(self.data_args.multi_column + ['labels'])
                    train_set = train_set.remove_columns([col for col in train_set.column_names if col not in keep_cols])
                    valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in keep_cols])
                    test_set = test_set.remove_columns([col for col in test_set.column_names if col not in keep_cols])

            datasets.append((train_set, valid_set, test_set, ppi))
            data_names.append(data_name)

        for data_dir in self.data_args.data_dirs:
            # local_data/taxon
            data_name = os.path.basename(os.path.normpath(data_dir))
            # Determine PPI by directory hint or columns
            ppi = 'ppi' in data_dir.lower()
            train_set, valid_set, test_set = self._load_local_dataset_dir(data_dir)

            # If not indicated by directory, infer from columns
            if not ppi:
                ppi = self._is_ppi_from_columns(train_set.column_names)

            if ppi:
                print('Standardizing PPI column names')
                try:
                    a_col, b_col = self._find_ppi_sequence_columns(train_set.column_names)
                except KeyError:
                    try:
                        a_col, b_col = self._find_ppi_sequence_columns(valid_set.column_names)
                    except KeyError:
                        a_col, b_col = self._find_ppi_sequence_columns(test_set.column_names)

                try:
                    lbl_col = self._find_first_present_column(train_set.column_names, label_candidates)
                except KeyError:
                    try:
                        lbl_col = self._find_first_present_column(valid_set.column_names, label_candidates)
                    except KeyError:
                        lbl_col = self._find_first_present_column(test_set.column_names, label_candidates)

                train_set = train_set.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'})
                valid_set = valid_set.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'})
                test_set = test_set.rename_columns({a_col: 'SeqA', b_col: 'SeqB', lbl_col: 'labels'})

                print('Removing extras')
                train_set = train_set.remove_columns([col for col in train_set.column_names if col not in ['SeqA', 'SeqB', 'labels']])
                valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in ['SeqA', 'SeqB', 'labels']])
                test_set = test_set.remove_columns([col for col in test_set.column_names if col not in ['SeqA', 'SeqB', 'labels']])
            else:
                print('Standardizing column names')
                use_multi = self.data_args.multi_column is not None
                if not use_multi:
                    try:
                        seq_col = self._find_first_present_column(train_set.column_names, seq_candidates)
                    except KeyError:
                        try:
                            seq_col = self._find_first_present_column(valid_set.column_names, seq_candidates)
                        except KeyError:
                            seq_col = self._find_first_present_column(test_set.column_names, seq_candidates)

                try:
                    label_col = self._find_first_present_column(train_set.column_names, label_candidates)
                except KeyError:
                    try:
                        label_col = self._find_first_present_column(valid_set.column_names, label_candidates)
                    except KeyError:
                        label_col = self._find_first_present_column(test_set.column_names, label_candidates)

                # Always standardize label column to 'labels'
                train_set = train_set.rename_columns({label_col: 'labels'})
                valid_set = valid_set.rename_columns({label_col: 'labels'})
                test_set = test_set.rename_columns({label_col: 'labels'})

                if not use_multi:
                    train_set = train_set.rename_columns({seq_col: 'seqs'})
                    valid_set = valid_set.rename_columns({seq_col: 'seqs'})
                    test_set = test_set.rename_columns({seq_col: 'seqs'})
                    # drop everything else
                    print('Removing extras')
                    train_set = train_set.remove_columns([col for col in train_set.column_names if col not in ['seqs', 'labels']])
                    valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in ['seqs', 'labels']])
                    test_set = test_set.remove_columns([col for col in test_set.column_names if col not in ['seqs', 'labels']])
                else:
                    # Validate requested multi columns exist
                    for col in self.data_args.multi_column:
                        assert col in train_set.column_names or col in valid_set.column_names or col in test_set.column_names, f"Column {col} not found in dataset {data_name}"
                    # Keep only requested columns and labels
                    keep_cols = set(self.data_args.multi_column + ['labels'])
                    train_set = train_set.remove_columns([col for col in train_set.column_names if col not in keep_cols])
                    valid_set = valid_set.remove_columns([col for col in valid_set.column_names if col not in keep_cols])
                    test_set = test_set.remove_columns([col for col in test_set.column_names if col not in keep_cols])

            datasets.append((train_set, valid_set, test_set, ppi))
            data_names.append(data_name)

        return self.process_datasets(hf_datasets=datasets, data_names=data_names)

    def get_embedding_dim_sql(self, save_path: str, test_seq: str, tokenizer: object) -> int:
        import sqlite3

        with sqlite3.connect(save_path) as conn:
            c = conn.cursor()
            c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (test_seq,))
            row = c.fetchone()
            assert row is not None, f"Test sequence not found in embeddings database: {test_seq}"
            test_embedding = row[0]
            test_embedding = embedding_blob_to_tensor(test_embedding, fallback_shape=(1, -1))
        embedding_dim = test_embedding.shape[-1]
        return embedding_dim

    def get_embedding_dim_pth(self, emb_dict: Dict[str, torch.Tensor], test_seq: str, tokenizer: object) -> int:
        test_embedding = emb_dict[test_seq]
        embedding_dim = test_embedding.shape[-1]
        return embedding_dim

    def build_vector_numpy_dataset_from_embeddings(
            self,
            model_name,
            train_seqs,
            valid_seqs,
            test_seqs,
        ):
        save_dir = self.embedding_args.embedding_save_dir
        train_array, valid_array, test_array = [], [], []
        # Get pooling types from embedding_args, default to ['mean'] if not available
        pooling_types = self.embedding_args.pooling_types
        hidden_state_index = self.embedding_args.hidden_state_index
        if self._sql:
            import sqlite3
            filename = get_embedding_filename(model_name, self._full, pooling_types, 'db', hidden_state_index)
            save_path = os.path.join(save_dir, filename)
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq in train_seqs:
                    embedding = _as_rows(self._select_from_sql(c, seq, cast_to_torch=False), seq, self._full)
                    train_array.append(embedding)

                for seq in valid_seqs:
                    embedding = _as_rows(self._select_from_sql(c, seq, cast_to_torch=False), seq, self._full)
                    valid_array.append(embedding)

                for seq in test_seqs:
                    embedding = _as_rows(self._select_from_sql(c, seq, cast_to_torch=False), seq, self._full)
                    test_array.append(embedding)
        else:
            filename = get_embedding_filename(model_name, self._full, pooling_types, 'pth', hidden_state_index)
            save_path = os.path.join(save_dir, filename)
            emb_dict = torch.load(save_path)
            for seq in train_seqs:
                embedding = _as_rows(self._select_from_pth(emb_dict, seq, cast_to_np=True), seq, self._full)
                train_array.append(embedding)
                
            for seq in valid_seqs:
                embedding = _as_rows(self._select_from_pth(emb_dict, seq, cast_to_np=True), seq, self._full)
                valid_array.append(embedding)

            for seq in test_seqs:
                embedding = _as_rows(self._select_from_pth(emb_dict, seq, cast_to_np=True), seq, self._full)
                test_array.append(embedding)
            del emb_dict

        train_array = np.concatenate(train_array, axis=0)
        valid_array = np.concatenate(valid_array, axis=0)
        test_array = np.concatenate(test_array, axis=0)
        
        if self._full: # average over the length of the sequence
            train_array = np.mean(train_array, axis=1)
            valid_array = np.mean(valid_array, axis=1)
            test_array = np.mean(test_array, axis=1)

        print_message('Numpy dataset shapes')
        print_message(f'Train: {train_array.shape}')
        print_message(f'Valid: {valid_array.shape}')
        print_message(f'Test: {test_array.shape}')
        return train_array, valid_array, test_array

    def build_pair_vector_numpy_dataset_from_embeddings(
            self,
            model_name,
            train_seqs_a,
            train_seqs_b,
            valid_seqs_a,
            valid_seqs_b,
            test_seqs_a,
            test_seqs_b,
            flip_train_pairs: bool = False,
        ):
        # Interaction is symmetric but concatenation is not, so swapping A and B augments
        # the training set. Validation and test keep the dataset order, because swapping
        # there would make a score depend on the random state rather than the model.
        save_dir = self.embedding_args.embedding_save_dir
        train_array, valid_array, test_array = [], [], []
        pooling_types = self.embedding_args.pooling_types
        hidden_state_index = self.embedding_args.hidden_state_index
        if self._sql:
            filename = get_embedding_filename(model_name, self._full, pooling_types, 'db', hidden_state_index)
            save_path = os.path.join(save_dir, filename)
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq_a, seq_b in zip(train_seqs_a, train_seqs_b):
                    if flip_train_pairs:
                        seq_a, seq_b = self._random_order(seq_a, seq_b)
                    embedding_a = _as_rows(self._select_from_sql(c, seq_a, cast_to_torch=False), seq_a, self._full)
                    embedding_b = _as_rows(self._select_from_sql(c, seq_b, cast_to_torch=False), seq_b, self._full)
                    train_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

                for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b):
                    embedding_a = _as_rows(self._select_from_sql(c, seq_a, cast_to_torch=False), seq_a, self._full)
                    embedding_b = _as_rows(self._select_from_sql(c, seq_b, cast_to_torch=False), seq_b, self._full)
                    valid_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

                for seq_a, seq_b in zip(test_seqs_a, test_seqs_b):
                    embedding_a = _as_rows(self._select_from_sql(c, seq_a, cast_to_torch=False), seq_a, self._full)
                    embedding_b = _as_rows(self._select_from_sql(c, seq_b, cast_to_torch=False), seq_b, self._full)
                    test_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))
        else:
            filename = get_embedding_filename(model_name, self._full, pooling_types, 'pth', hidden_state_index)
            save_path = os.path.join(save_dir, filename)
            emb_dict = torch.load(save_path)
            for seq_a, seq_b in zip(train_seqs_a, train_seqs_b):
                if flip_train_pairs:
                    seq_a, seq_b = self._random_order(seq_a, seq_b)
                embedding_a = _as_rows(self._select_from_pth(emb_dict, seq_a, cast_to_np=True), seq_a, self._full)
                embedding_b = _as_rows(self._select_from_pth(emb_dict, seq_b, cast_to_np=True), seq_b, self._full)
                train_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

            for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b):
                embedding_a = _as_rows(self._select_from_pth(emb_dict, seq_a, cast_to_np=True), seq_a, self._full)
                embedding_b = _as_rows(self._select_from_pth(emb_dict, seq_b, cast_to_np=True), seq_b, self._full)
                valid_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

            for seq_a, seq_b in zip(test_seqs_a, test_seqs_b):
                embedding_a = _as_rows(self._select_from_pth(emb_dict, seq_a, cast_to_np=True), seq_a, self._full)
                embedding_b = _as_rows(self._select_from_pth(emb_dict, seq_b, cast_to_np=True), seq_b, self._full)
                test_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))
            del emb_dict

        train_array = np.concatenate(train_array, axis=0)
        valid_array = np.concatenate(valid_array, axis=0)
        test_array = np.concatenate(test_array, axis=0)
        
        if self._full: # average over the length of the sequence
            train_array = np.mean(train_array, axis=1)
            valid_array = np.mean(valid_array, axis=1)
            test_array = np.mean(test_array, axis=1)

        print_message('Numpy dataset shapes')
        print_message(f'Train: {train_array.shape}')
        print_message(f'Valid: {valid_array.shape}')
        print_message(f'Test: {test_array.shape}')
        return train_array, valid_array, test_array

    def prepare_scikit_dataset(
        self, model_name, dataset, standardize: bool = True, flip_train_pairs: bool = False
    ):
        """Materialize pooled embeddings and labels as numpy arrays.

        `standardize` exists for axis-aligned learners: a per-feature shift and scale
        cannot change which splits a decision tree can express, and centering would turn
        a sparse SAE feature matrix dense.

        `flip_train_pairs` carries --random_pair_flipping into the paired builder and is
        ignored for single-sequence datasets.
        """
        train_set, valid_set, test_set, _, label_type, ppi = dataset

        if ppi:
            X_train, X_valid, X_test = self.build_pair_vector_numpy_dataset_from_embeddings(
                model_name,
                list(train_set['SeqA']),
                list(train_set['SeqB']),
                list(valid_set['SeqA']),
                list(valid_set['SeqB']),
                list(test_set['SeqA']),
                list(test_set['SeqB']),
                flip_train_pairs=flip_train_pairs,
            )
        else:
            X_train, X_valid, X_test = self.build_vector_numpy_dataset_from_embeddings(
                model_name,
                list(train_set['seqs']),
                list(valid_set['seqs']),
                list(test_set['seqs']),
            )

        embedding_scaler = self.embedding_args.embedding_scaler and standardize
        assert isinstance(embedding_scaler, bool), f"Invalid embedding_scaler: {embedding_scaler}"
        if not self._full and embedding_scaler:
            print_message('Fitting StandardScaler on scikit training embeddings')
            embedding_standardizer = EmbeddingStandardizer.fit_numpy(X_train)
            X_train = embedding_standardizer.transform_numpy(X_train)
            X_valid = embedding_standardizer.transform_numpy(X_valid)
            X_test = embedding_standardizer.transform_numpy(X_test)

        y_train = self._labels_to_numpy(list(train_set['labels']))
        y_valid = self._labels_to_numpy(list(valid_set['labels']))
        y_test = self._labels_to_numpy(list(test_set['labels']))

        print_message('Numpy dataset shapes with labels')
        print_message(f'Train: {X_train.shape}, {y_train.shape}')
        print_message(f'Valid: {X_valid.shape}, {y_valid.shape}')
        print_message(f'Test: {X_test.shape}, {y_test.shape}')
        return X_train, y_train, X_valid, y_valid, X_test, y_test, label_type
