import io
import os
import struct
import torch
import shutil
import pyfiglet
from functools import partial
from typing import List, Optional, Tuple

import numpy as np

torch_load = partial(torch.load, map_location='cpu', weights_only=True)

# Compact blob serialization constants
# Canonical source: core/embed/blob.py. Keep in sync with fastplms/embedding_mixin.py.
_COMPACT_VERSION = 0x01
# Code 1 is the original bfloat16 encoding, which converted to float16 bytes. bfloat16
# carries a wider exponent than float16, so that conversion silently flushed values below
# the float16 normal range: 1.0e-6 came back as 1.01e-6. Code 3 stores the bfloat16 bit
# pattern instead and is exact. Code 1 is still read so existing caches keep working.
_LEGACY_BFLOAT16_CODE = 1
_DTYPE_TO_CODE = {torch.float16: 0, torch.float32: 2, torch.bfloat16: 3}
_CODE_TO_DTYPE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32, 3: torch.bfloat16}
_CODE_TO_NP_DTYPE = {0: np.float16, 1: np.float16, 2: np.float32, 3: np.int16}

# Coordinate-list blob format, selected by the model that produces the embedding.
# Sparse autoencoder features are the case that pays: per-residue features have exactly
# k of codebook_dim active whatever the sequence length, and max-pooled features measured
# 3.6% dense at codebook width 131072. Ordinary PLM embeddings have no exact zeros.
_SPARSE_VERSION = 0x02
_INDEX_CODE_TO_NP = {0: np.uint16, 1: np.uint32}
_UINT16_INDEX_LIMIT = 1 << 16


def _index_code(numel: int) -> int:
    """Narrowest index width that addresses a flattened embedding of this size."""
    return 0 if numel <= _UINT16_INDEX_LIMIT else 1


def _raw_value_bytes(t: torch.Tensor) -> bytes:
    # numpy has no bfloat16, so those tensors travel as their own 2-byte bit pattern.
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().tobytes()

    return t.numpy().tobytes()


def _restore_dtype(t: torch.Tensor, dtype_code: int) -> torch.Tensor:
    """Recover the stored dtype from the raw array a blob decodes to."""
    if dtype_code == _LEGACY_BFLOAT16_CODE:
        return t.to(torch.bfloat16)  # float16 values written before code 3 existed
    if _CODE_TO_DTYPE[dtype_code] == torch.bfloat16:
        return t.view(torch.bfloat16)  # raw bit pattern

    return t


def _sparse_blob(t: torch.Tensor, dtype_code: int, index_code: int) -> bytes:
    """Serialize one tensor as flat indices plus values.

    Format: [version:1][dtype_code:1][index_code:1][ndim:4][shape:4*ndim][nnz:4]
            [indices:nnz*index_width][values:nnz*value_width]
    """
    # t: arbitrary rank; flat: (numel,)
    flat = t.reshape(-1)
    indices = torch.nonzero(flat, as_tuple=False).reshape(-1)  # (nnz,)
    shape = tuple(t.shape)
    header = struct.pack(
        f'<BBBi{len(shape)}ii',
        _SPARSE_VERSION,
        dtype_code,
        index_code,
        len(shape),
        *shape,
        indices.numel(),
    )
    index_bytes = indices.numpy().astype(_INDEX_CODE_TO_NP[index_code]).tobytes()
    return header + index_bytes + _raw_value_bytes(flat[indices])


def tensor_to_embedding_blob(tensor: torch.Tensor, sparse: bool = False) -> bytes:
    """Serialize a tensor to compact binary format for SQLite blob storage.

    Format: [version:1][dtype_code:1][ndim:4][shape:4*ndim][raw_bytes]
    bfloat16 tensors keep their own bit pattern under dtype_code=3.
    Falls back to torch.save for unsupported dtypes.

    `sparse` selects the coordinate-list format. The caller decides it, because whether
    coordinates pay follows from the model producing the embedding rather than from any
    single tensor: see `EsmcSaeForEmbedding.sparse_storage`.
    """
    t = tensor.cpu()
    if t.dtype not in _DTYPE_TO_CODE:
        buffer = io.BytesIO()
        torch.save(t, buffer)
        return buffer.getvalue()
    dtype_code = _DTYPE_TO_CODE[t.dtype]

    if sparse:
        return _sparse_blob(t, dtype_code, _index_code(t.numel()))

    shape = t.shape
    header = struct.pack(f'<BBi{len(shape)}i', _COMPACT_VERSION, dtype_code, len(shape), *shape)
    return header + _raw_value_bytes(t)


def _compact_header(dtype: torch.dtype, shape: tuple) -> bytes:
    """Build just the compact header for a given dtype and shape."""
    dtype_code = _DTYPE_TO_CODE[dtype]
    return struct.pack(f'<BBi{len(shape)}i', _COMPACT_VERSION, dtype_code, len(shape), *shape)


def _dense_blobs(rows: torch.Tensor, shape: tuple) -> list:
    """Serialize a block of rows in the compact dense format, in one numpy conversion."""
    # rows: (n, *shape)
    header = _compact_header(rows.dtype, shape)
    raw = _raw_value_bytes(rows)
    stride = len(raw) // max(rows.shape[0], 1)
    return [header + raw[i * stride:(i + 1) * stride] for i in range(rows.shape[0])]


def _sparse_blobs(rows: torch.Tensor, shape: tuple, dtype_code: int, index_code: int) -> list:
    """Serialize a block of sparse rows in one pass.

    One `nonzero` call and one numpy conversion cover the whole block, then each row
    takes a slice of the shared index and value buffers. `torch.nonzero` orders
    coordinates by row, so those slices are contiguous.
    """
    # rows: (n, numel)
    coordinates = torch.nonzero(rows, as_tuple=False)  # (nnz_total, 2)
    counts = torch.bincount(coordinates[:, 0], minlength=rows.shape[0])  # (n,)
    values = rows[coordinates[:, 0], coordinates[:, 1]]  # (nnz_total,)

    index_dtype = _INDEX_CODE_TO_NP[index_code]
    index_buffer = coordinates[:, 1].numpy().astype(index_dtype).tobytes()
    value_buffer = _raw_value_bytes(values)
    index_width = index_dtype().itemsize
    value_width = len(value_buffer) // max(len(values), 1)

    blobs = []
    start = 0
    for nnz in counts.tolist():
        header = struct.pack(
            f'<BBBi{len(shape)}ii', _SPARSE_VERSION, dtype_code, index_code, len(shape), *shape, nnz
        )
        stop = start + nnz
        blobs.append(
            header
            + index_buffer[start * index_width:stop * index_width]
            + value_buffer[start * value_width:stop * value_width]
        )
        start = stop

    return blobs


def batch_tensor_to_blobs(batch: torch.Tensor, sparse: bool = False) -> list:
    """Serialize a batch of identically-shaped embeddings to compact blobs.

    Input: (b, d) or (b, l, d) tensor already on CPU and in target dtype.
    Returns: list of b bytes objects, one per embedding.

    Both formats are written in one bulk numpy conversion for the whole batch, which is
    much faster than serializing row by row. `sparse` selects the coordinate-list format;
    the caller decides it from the model, so ordinary embeddings pay nothing for it.
    """
    assert batch.dtype in _DTYPE_TO_CODE, f"Unsupported dtype {batch.dtype}"
    single_shape = tuple(batch.shape[1:])

    if not sparse:
        return _dense_blobs(batch, single_shape)

    numel = int(np.prod(single_shape)) if single_shape else 1
    return _sparse_blobs(
        batch.reshape(batch.shape[0], -1),
        single_shape,
        _DTYPE_TO_CODE[batch.dtype],
        _index_code(numel),
    )


def embedding_blob_to_tensor(
    blob: bytes,
    fallback_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Deserialize an embedding blob from SQLite.

    Tries compact binary format first (version byte 0x01), then the coordinate-list
    format (0x02), then PyTorch torch.save format, then legacy raw float32 with
    fallback_shape. Both compact formats return a dense tensor.
    """
    if len(blob) >= 3 and blob[0] == _SPARSE_VERSION and blob[1] in _CODE_TO_DTYPE:
        dtype_code = blob[1]
        index_code = blob[2]
        ndim = struct.unpack_from('<i', blob, 3)[0]
        shape = struct.unpack_from(f'<{ndim}i', blob, 7)
        nnz = struct.unpack_from('<i', blob, 7 + 4 * ndim)[0]
        np_dtype = _CODE_TO_NP_DTYPE[dtype_code]
        np_index_dtype = _INDEX_CODE_TO_NP[index_code]

        index_offset = 11 + 4 * ndim
        indices = np.frombuffer(blob, dtype=np_index_dtype, count=nnz, offset=index_offset)  # (nnz,)
        value_offset = index_offset + nnz * np_index_dtype().itemsize
        values = np.frombuffer(blob, dtype=np_dtype, count=nnz, offset=value_offset)  # (nnz,)

        dense = np.zeros(int(np.prod(shape)) if shape else 1, dtype=np_dtype)  # (numel,)
        dense[indices] = values
        return _restore_dtype(torch.from_numpy(dense.reshape(shape)), dtype_code)

    if len(blob) >= 2 and blob[0] == _COMPACT_VERSION and blob[1] in _CODE_TO_DTYPE:
        dtype_code = blob[1]
        ndim = struct.unpack_from('<i', blob, 2)[0]
        shape = struct.unpack_from(f'<{ndim}i', blob, 6)
        data_offset = 6 + 4 * ndim
        np_dtype = _CODE_TO_NP_DTYPE[dtype_code]
        arr = np.frombuffer(blob, dtype=np_dtype, offset=data_offset).reshape(shape).copy()
        return _restore_dtype(torch.from_numpy(arr), dtype_code)

    try:
        t = torch_load(io.BytesIO(blob))
        if isinstance(t, torch.Tensor):
            return t
    except Exception:
        pass
    if fallback_shape is not None:
        return torch.tensor(
            np.frombuffer(blob, dtype=np.float32).reshape(fallback_shape)
        )
    raise ValueError(
        "Blob is not in compact/PyTorch format and no fallback_shape provided for legacy float32."
    )


class _SQLWriter:
    """Context manager for async SQL embedding writes. Matches core/embed/storage.SQLEmbeddingWriter."""

    def __init__(self, conn, queue_maxsize: int = 4) -> None:
        import queue
        import threading
        self._conn = conn
        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._threading = threading

    def __enter__(self) -> "_SQLWriter":
        self._thread = self._threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        return self

    def write_batch(self, rows) -> None:
        self._queue.put(rows)

    def _writer_loop(self) -> None:
        cursor = self._conn.cursor()
        while True:
            item = self._queue.get()
            if item is None:
                break
            cursor.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", item)
            if self._queue.qsize() == 0:
                self._conn.commit()
        self._conn.commit()

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None


def clear_screen() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def print_message(message: str) -> None:
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 50
    print('\n' + '-' * terminal_width)
    print(f'\n{message}\n')
    print('-' * terminal_width + '\n')


def print_title(title: str) -> None:
    print(pyfiglet.figlet_format(title, font='3d-ascii'))


def print_done() -> None:
    print(pyfiglet.figlet_format('== Done ==', font='js_stick_letters'))


def expand_dms_ids_all(dms_ids: List[str], mode: Optional[str] = None) -> List[str]:
    """
    Expand 'all' to actual DMS IDs from benchmarks.proteingym.dms_ids.
    """
    if any(str(x).lower() == 'all' for x in dms_ids):
        if mode == 'indels':
            from benchmarks.proteingym.dms_ids import ALL_INDEL_DMS_IDS
            dms_ids = list(ALL_INDEL_DMS_IDS)
        else:
            from benchmarks.proteingym.dms_ids import ALL_SUBSTITUTION_DMS_IDS
            dms_ids = list(ALL_SUBSTITUTION_DMS_IDS)
    return dms_ids


def maybe_compile(model: torch.nn.Module, dynamic: bool = False) -> torch.nn.Module:
    if dynamic:
        # dynamic=True (padding='longest') is incompatible with flex attention's
        # create_block_mask under torch.compile, causing CUDA illegal memory access.
        # Skip compilation; the variable-shape batches already avoid wasted padding.
        print_message("Skipping torch.compile (dynamic shapes + flex attention incompatible)")
        return model
    try:
        model = torch.compile(model)
        print_message("Model compiled")
    except Exception as e:
        print_message(f"Skipping torch.compile: {e}")
    return model


if __name__ == '__main__':
    folders_to_clean = ['logs', 'results', 'plots', 'embeddings', 'weights']
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            files = os.listdir(folder)
            if files:
                response = input(f"Do you want to delete all files in '{folder}' folder? ({len(files)} files) [y/N]: ")
                if response.lower() == 'y':
                    for file in files:
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    print(f"All files in '{folder}' have been deleted.")
                else:
                    print(f"Skipped cleaning '{folder}' folder.")
            else:
                print(f"'{folder}' folder is already empty.")
        else:
            print(f"'{folder}' folder does not exist.")
