# Data

This page documents how Protify loads and prepares data: `DataArguments`, supported datasets, local directories (`data_dirs`), the `get_data()` flow, column normalization, translation flags, and dataset classes. To list supported datasets from the CLI or Python, see [Resource listing](resource_listing.md).

---

## Overview

Data is specified either by **dataset names** (HuggingFace IDs or special presets like `standard_benchmark`) or by **local directories** containing split files. After loading, columns are normalized (e.g. to `seqs`/`labels` or `SeqA`/`SeqB`/`labels` for PPI), optional sequence translation is applied, and the translated output is trimmed or truncated by the target model's raw-sequence budget. Valid lowercase biological sequences are preserved and normalized to uppercase. The result is a dictionary of datasets keyed by name, each value being `(train_set, valid_set, test_set, num_labels, label_type, ppi)`.

---

## How it works

1. **DataArguments** is built from config (`data_names`, `data_dirs`, `delimiter`, `max_length`, etc.). From `data_names`, the code resolves `data_paths` (HuggingFace dataset IDs) and sets `protein_gym` when the name is `protein_gym`.
2. **DataMixin.get_data()** loads each path: for HuggingFace it uses `load_dataset(path)`; for `data_dirs` it globs split files and reads tabular files with pandas or labeled FASTA directly into HuggingFace `Dataset`.
3. **Splits:** Train is required; at least one of valid or test is required. Valid aliases are `valid`, `validation`, `val`, and `dev`; test aliases are `test` and `testing`. If valid is missing, 10% of train is used; if test is missing, 10% of train is used.
4. **process_datasets()** normalizes column names, drops missing sequence/label, optionally translates sequences, removes zero-length translated outputs, applies trim or truncation to the tokenizer-facing sequence, and infers `label_type` (e.g. singlelabel, multilabel, regression, tokenwise).
5. For **embedding-based training**, datasets are later built from precomputed embeddings (SQLite or `.pth`) via `build_vector_numpy_dataset_from_embeddings` or the PPI/multi-column variants.

---

## DataArguments

Defined in [data_mixin.py](../src/protify/data/data_mixin.py). All arguments that affect data loading and preprocessing:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_names` | List[str] | (required for HF) | Dataset names. Can be keys from `supported_datasets`, `standard_benchmark`, `vector_benchmark`, or literal HuggingFace IDs. |
| `data_dirs` | Optional[List[str]] | [] | Local directories; each must contain `train.*` and at least one valid/test split (CSV, TSV, Excel, or labeled FASTA). |
| `delimiter` | str | ',' | Delimiter when loading from `data_dirs`. |
| `col_names` | List[str] | ['seqs', 'labels'] | Column names (legacy; columns are often inferred from the data). |
| `max_length` | int | 1024 | Maximum sequence length for trim/truncation. |
| `trim` | bool | False | If True, drop rows exceeding `max_length`; if False, truncate to `max_length`. |
| `multi_column` | Optional[List[str]] | None | Names of sequence columns for multi-input tasks (e.g. PPI mutation effect). |
| `aa_to_dna` | bool | False | Translate amino acid sequences to DNA codons. |
| `aa_to_rna` | bool | False | Translate amino acid sequences to RNA. |
| `dna_to_aa` | bool | False | Translate DNA to amino acid. |
| `rna_to_aa` | bool | False | Translate RNA to amino acid. |
| `codon_to_aa` | bool | False | Map codon tokens to amino acid. |
| `aa_to_codon` | bool | False | Map amino acid to codon tokens. |

Only one of the translation flags may be True. Setting `data_names[0]` to `standard_benchmark` or `vector_benchmark` expands to the corresponding list in [supported_datasets.py](../src/protify/data/supported_datasets.py).

---

## Supported datasets and benchmark lists

- **supported_datasets:** A dict mapping dataset name to HuggingFace dataset ID or path (e.g. `'EC': 'GleghornLab/EC_reg'`).
- **standard_data_benchmark:** List of dataset names used when `data_names=['standard_benchmark']`.
- **vector_benchmark:** List used when `data_names=['vector_benchmark']`.
- **possible_with_vector_reps:** Datasets that can be used with precomputed vector representations.
- **testing:** Small set used in tests.

To list datasets and get info from the CLI:

```bash
py -m src.protify.data.dataset_utils --list
py -m src.protify.data.dataset_utils --info EC
```

See [Resource listing](resource_listing.md) for programmatic access and combined model/dataset listing.

---

## data_dirs and file layout

For each directory in `data_dirs`, `get_data()` looks for files whose stem matches:

- `train.*`
- `valid.*`, `validation.*`, `val.*`, or `dev.*`
- `test.*` or `testing.*`

Tabular files are read with pandas (`read_csv` or `read_excel` by extension) and converted to HuggingFace `Dataset`. FASTA files with extensions `.fa`, `.fasta`, `.faa`, `.fna`, `.ffn`, or `.frn` are supported for supervised single-sequence datasets when every header has explicit `label=` metadata, for example `>seq_001 label=1`. Column names are inferred (e.g. sequence column as `seqs` or `Seq`/`sequence`, label as `labels` or `label`). PPI data is normalized to `SeqA`, `SeqB`, `labels`.

---

## Column normalization and label_type

- **Single-sequence:** Columns become `seqs` and `labels` (or `multi_column` names plus `labels`).
- **PPI:** Columns become `SeqA`, `SeqB`, `labels`.
- **label_type** is inferred: e.g. `singlelabel`, `multilabel`, `regression`, `sigmoid_regression`, `tokenwise`, `string`. This drives the loss and metrics during training.

---

## Dataset classes

Used when building PyTorch datasets from embeddings or raw sequences ([dataset_classes.py](../src/protify/data/dataset_classes.py)):

| Class | Use case |
|-------|----------|
| **EmbedsLabelsDataset** | Single-sequence, embeddings from in-memory dict. |
| **EmbedsLabelsDatasetFromDisk** | Single-sequence, embeddings from SQLite (batched reads). |
| **PairEmbedsLabelsDataset** | PPI, embeddings from in-memory dict. |
| **PairEmbedsLabelsDatasetFromDisk** | PPI, embeddings from SQLite; optional pair flipping. |
| **MultiEmbedsLabelsDataset** / **MultiEmbedsLabelsDatasetFromDisk** | Multi-column sequence inputs, embeddings from dict or SQLite. |
| **StringLabelDataset** | Single-sequence, raw sequences and labels. |
| **PairStringLabelDataset** | PPI, raw pairs and labels; optional pair flipping. |
| **SimpleProteinDataset** | Wrapper over a list of sequence strings (e.g. for embedding). |

Collators in [data_collators.py](../src/protify/data/data_collators.py) pair with these for batching (e.g. `EmbedsLabelsCollator`, `PairEmbedsLabelsCollator`, `StringLabelsCollator`).

---

## prepare_scikit_dataset

`DataMixin.prepare_scikit_dataset(model_name, dataset)` builds `X_train`, `y_train`, `X_valid`, `y_valid`, `X_test`, `y_test` and `label_type` from embedding-backed datasets, for use with the scikit-learn path (`run_scikit_scheme`).

---

## Examples

### Single dataset by name

```yaml
data_names: [DeepLoc-2]
max_length: 1024
trim: false
```

### Standard benchmark (many datasets)

```yaml
data_names: [standard_benchmark]
```

### Local directory

```bash
py -m src.protify.main --data_dirs path/to/my_data --delimiter "," --model_names ESM2-8 --data_names []
```

Ensure `path/to/my_data` contains `train.csv` and at least one valid/test split such as `validation.csv` or `testing.tsv`.

### Local FASTA directory

```text
train.fasta
validation.fasta
test.fasta
```

Each record must include `label=` in the header:

```text
>seq_001 label=1
MKTAYIAKQRQISFVKSHFSRQ
```

### Translation (DNA to amino acid)

```yaml
data_names: [my_dna_dataset]
dna_to_aa: true
```

---

## See also

- [Configuration](cli_and_config.md) for data-related CLI flags
- [Resource listing](resource_listing.md) for listing datasets and programmatic access
- [Models and embeddings](models_and_embeddings.md) for how embeddings are built from sequences
- [Probes and training](probes_and_training.md) for how datasets are used in training
