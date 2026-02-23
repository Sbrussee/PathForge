# src/pathbench/core/io/h5/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import h5py
import numpy as np


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
    h5_file.create_dataset(dataset_path, data=payload, dtype=_UTF8)


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
    """Write numeric array dataset (overwrite)."""
    arr = np.asarray(array, dtype=dtype)
    delete_if_exists(h5_file, dataset_path)
    parent = str(Path(dataset_path).parent).replace("\\", "/")
    if parent and parent != ".":
        ensure_group(h5_file, parent)
    h5_file.create_dataset(
        dataset_path,
        data=arr,
        dtype=dtype,
        compression=compression,
        compression_opts=compression_opts if compression else None,
        chunks=True if arr.ndim >= 1 else None,
    )


def read_array_dataset(h5_file: h5py.File, dataset_path: str) -> np.ndarray:
    return np.asarray(h5_file[dataset_path][()])
