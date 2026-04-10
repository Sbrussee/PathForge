# src/pathbench/core/io/h5/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import h5py
import numpy as np

STATUS_ATTR = "status"
STATUS_WRITING = "writing"
STATUS_COMPLETE = "complete"


@dataclass(slots=True)
class FileHandleH5:
    path: Path
    mode: str = "a"
    _h5: h5py.File | None = None

    def __enter__(self) -> "FileHandleH5":
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, self.mode)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError("H5 file is not open. Use: with FileHandleH5(...) as f:")
        return self._h5


# ---- generic helpers ---------------------------------------------------------

_UTF8 = h5py.string_dtype(encoding="utf-8")


def exists(h5_file: h5py.File, h5_path: str) -> bool:
    return h5_path in h5_file


def get_dataset(h5_file: h5py.File, h5_path: str) -> h5py.Dataset | None:
    obj = h5_file.get(h5_path)
    if obj is None or not isinstance(obj, h5py.Dataset):
        return None
    return obj


def get_status(h5_obj: h5py.Dataset | h5py.Group) -> str | None:
    value = h5_obj.attrs.get(STATUS_ATTR)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def is_complete(h5_obj: h5py.Dataset | h5py.Group) -> bool:
    return get_status(h5_obj) == STATUS_COMPLETE


def delete_if_exists(h5_file: h5py.File, h5_path: str) -> None:
    if h5_path in h5_file:
        del h5_file[h5_path]


def ensure_group(h5_file: h5py.File, group_path: str) -> h5py.Group:
    return h5_file.require_group(group_path)


def write_json_dataset(h5_file: h5py.File, dataset_path: str, obj: Any) -> None:
    """Write obj as a scalar UTF-8 JSON dataset (overwrite)."""
    payload = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    delete_if_exists(h5_file, dataset_path)
    parent = str(Path(dataset_path).parent).replace("\\", "/")
    if parent and parent != ".":
        ensure_group(h5_file, parent)
    dset = h5_file.create_dataset(dataset_path, shape=(), dtype=_UTF8)
    dset.attrs[STATUS_ATTR] = STATUS_WRITING
    dset[()] = payload

    raw = dset[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    json.loads(raw)
    dset.attrs[STATUS_ATTR] = STATUS_COMPLETE


def read_json_dataset(h5_file: h5py.File, dataset_path: str) -> Any:
    """Read scalar UTF-8 JSON dataset."""
    dset = h5_file[dataset_path]
    raw = dset[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def write_array_dataset(
    h5_file: h5py.File,
    dataset_path: str,
    array: np.ndarray,
    *,
    dtype: Any,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """Write array dataset (overwrite), including UTF-8 string arrays."""
    arr = np.asarray(array, dtype=dtype)
    arr_dtype = np.asarray(arr).dtype
    h5_dtype: Any = arr_dtype

    # h5py cannot create datasets directly from NumPy unicode dtypes (kind 'U').
    # Store text-like arrays as variable-length UTF-8 string datasets instead.
    if arr_dtype.kind in {"U", "S"}:
        h5_dtype = h5py.string_dtype(encoding="utf-8")
        arr = np.asarray(arr, dtype=object)

    delete_if_exists(h5_file, dataset_path)
    parent = str(Path(dataset_path).parent).replace("\\", "/")
    if parent and parent != ".":
        ensure_group(h5_file, parent)
    # h5py does not allow chunk/filter options for scalar datasets.
    # Keep compression for non-scalar arrays only.
    use_compression = compression if arr.ndim >= 1 else None
    use_compression_opts = compression_opts if use_compression else None
    use_chunks = True if arr.ndim >= 1 else None

    dset = h5_file.create_dataset(
        dataset_path,
        shape=arr.shape,
        dtype=h5_dtype,
        compression=use_compression,
        compression_opts=use_compression_opts,
        chunks=use_chunks,
    )
    dset.attrs[STATUS_ATTR] = STATUS_WRITING
    if arr.ndim == 0:
        dset[()] = arr[()]
    else:
        dset[...] = arr
    if tuple(dset.shape) != tuple(arr.shape):
        raise ValueError(f"Stored dataset has shape {dset.shape}, expected {arr.shape}.")
    dset.attrs[STATUS_ATTR] = STATUS_COMPLETE


def read_array_dataset(h5_file: h5py.File, dataset_path: str) -> np.ndarray:
    return np.asarray(h5_file[dataset_path][()])
