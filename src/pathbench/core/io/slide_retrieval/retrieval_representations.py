from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from pathbench.core.io.slide_artifacts.base import (
    FileHandleH5,
    exists,
    read_array_dataset,
    write_array_dataset,
)
from pathbench.core.io.slide_retrieval.layout import (
    DEFAULT_LAYOUT,
    RetrievalH5Layout,
)


# ---------------------------------------------------------------------------
# Entry-level API
# ---------------------------------------------------------------------------


def retrieval_representation_entry_exists(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether one retrieval entry contains the required stored objects."""
    paths = _retrieval_representation_entry_paths(
        tile_id,
        representation_id,
        entry_id,
        layout=layout,
    )
    return all(
        exists(retrieval_artifact.h5, path)
        for path in (
            paths["embedding"],
            paths["metadata"],
        )
    )


def read_retrieval_representation_entry(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Read all stored objects for one retrieval entry."""
    paths = _retrieval_representation_entry_paths(
        tile_id,
        representation_id,
        entry_id,
        layout=layout,
    )
    if not retrieval_representation_entry_exists(
        retrieval_artifact,
        tile_id,
        representation_id,
        entry_id,
        layout=layout,
    ):
        raise KeyError(
            "Missing required retrieval entry objects for "
            f"tile_id={tile_id!r}, representation_id={representation_id!r}, entry_id={entry_id!r}."
        )

    additional_data: dict[str, np.ndarray] = {}
    additional_group_path = layout.retrieval_representation_additional_data_group(
        tile_id,
        representation_id,
        entry_id,
    )
    if additional_group_path in retrieval_artifact.h5:
        for name in retrieval_artifact.h5[additional_group_path].keys():
            additional_data[name] = read_additional_retrieval_representation_data(
                retrieval_artifact,
                tile_id,
                representation_id,
                entry_id,
                name,
                layout=layout,
            )

    return {
        "metadata": _read_json_dataset(retrieval_artifact.h5, paths["metadata"]),
        "params": (
            _read_json_dataset(retrieval_artifact.h5, paths["params"])
            if exists(retrieval_artifact.h5, paths["params"])
            else {}
        ),
        "embedding": read_array_dataset(retrieval_artifact.h5, paths["embedding"]),
        "additional_data": additional_data,
    }


def write_retrieval_representation_entry(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    metadata: dict[str, Any],
    embedding: np.ndarray,
    params: dict[str, Any] | None = None,
    additional_data: dict[str, np.ndarray] | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write all stored objects for one retrieval entry."""
    write_retrieval_representation_metadata(
        retrieval_artifact,
        tile_id,
        representation_id,
        entry_id,
        metadata,
        layout=layout,
    )
    write_retrieval_representation_params(
        retrieval_artifact,
        tile_id,
        representation_id,
        entry_id,
        dict(params or {}),
        layout=layout,
    )
    write_retrieval_representation(
        retrieval_artifact,
        tile_id,
        representation_id,
        entry_id,
        embedding,
        layout=layout,
    )

    additional_group_path = layout.retrieval_representation_additional_data_group(
        tile_id,
        representation_id,
        entry_id,
    )
    if additional_group_path in retrieval_artifact.h5:
        del retrieval_artifact.h5[additional_group_path]

    for name, value in dict(additional_data or {}).items():
        write_additional_retrieval_representation_data(
            retrieval_artifact,
            tile_id,
            representation_id,
            entry_id,
            name,
            value,
            layout=layout,
        )


# ---------------------------------------------------------------------------
# Field-level wrappers
# ---------------------------------------------------------------------------


def retrieval_representation_exists(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether the main embedding exists for one retrieval entry."""
    path = layout.retrieval_representation_embedding(
        tile_id,
        representation_id,
        entry_id,
    )
    if not exists(retrieval_artifact.h5, path):
        return False

    if expected_shape is None:
        return True

    dset = retrieval_artifact.h5[path]
    shape = getattr(dset, "shape", None)
    return tuple(shape) == tuple(expected_shape)


def retrieval_representation_item_exists(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Backward-compatible alias for the full-entry existence check."""
    return retrieval_representation_entry_exists(
        retrieval_artifact,
        tile_id,
        representation_id,
        entry_id,
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Core entry fields
# ---------------------------------------------------------------------------


def read_retrieval_representation(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Read the main embedding array for one retrieval entry."""
    path = layout.retrieval_representation_embedding(
        tile_id,
        representation_id,
        entry_id,
    )
    return read_array_dataset(retrieval_artifact.h5, path)


def write_retrieval_representation(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    data: np.ndarray,
    *,
    dtype: np.dtype | str | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write the main embedding array for one retrieval entry."""
    array = np.asarray(data)
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    write_array_dataset(
        retrieval_artifact.h5,
        layout.retrieval_representation_embedding(
            tile_id,
            representation_id,
            entry_id,
        ),
        array,
        dtype=array.dtype,
    )


def read_retrieval_representation_type(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> str:
    """Read the representation-type label for one retrieval entry."""
    path = layout.retrieval_representation_type(
        tile_id,
        representation_id,
        entry_id,
    )
    return _read_string_dataset(retrieval_artifact.h5, path)


def write_retrieval_representation_type(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    representation_type: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write the representation-type label for one retrieval entry."""
    path = layout.retrieval_representation_type(
        tile_id,
        representation_id,
        entry_id,
    )
    _write_string_dataset(retrieval_artifact.h5, path, representation_type)


def read_retrieval_representation_metadata(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Read the metadata payload for one retrieval entry."""
    path = layout.retrieval_representation_metadata(
        tile_id,
        representation_id,
        entry_id,
    )
    return _read_json_dataset(retrieval_artifact.h5, path)


def write_retrieval_representation_metadata(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    metadata: dict[str, Any],
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write the metadata payload for one retrieval entry."""
    path = layout.retrieval_representation_metadata(
        tile_id,
        representation_id,
        entry_id,
    )
    _write_json_dataset(retrieval_artifact.h5, path, metadata)


def read_retrieval_representation_params(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Read the readable params payload for one retrieval entry."""
    path = layout.retrieval_representation_params(
        tile_id,
        representation_id,
        entry_id,
    )
    return _read_json_dataset(retrieval_artifact.h5, path)


def write_retrieval_representation_params(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    params: dict[str, Any],
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write the readable params payload for one retrieval entry."""
    path = layout.retrieval_representation_params(
        tile_id,
        representation_id,
        entry_id,
    )
    _write_json_dataset(retrieval_artifact.h5, path, params)


# ---------------------------------------------------------------------------
# Additional data arrays
# ---------------------------------------------------------------------------


def additional_retrieval_representation_data_exists(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    name: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether one additional-data array exists for a retrieval entry."""
    path = layout.retrieval_representation_additional_data(
        tile_id,
        representation_id,
        entry_id,
        name,
    )
    if not exists(retrieval_artifact.h5, path):
        return False

    if expected_shape is None:
        return True

    dset = retrieval_artifact.h5[path]
    shape = getattr(dset, "shape", None)
    return tuple(shape) == tuple(expected_shape)


def read_additional_retrieval_representation_data(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    name: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Read one additional-data array for a retrieval entry."""
    path = layout.retrieval_representation_additional_data(
        tile_id,
        representation_id,
        entry_id,
        name,
    )
    return read_array_dataset(retrieval_artifact.h5, path)


def write_additional_retrieval_representation_data(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    name: str,
    data: np.ndarray,
    *,
    dtype: np.dtype | str | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write one additional-data array for a retrieval entry."""
    array = np.asarray(data)
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    write_array_dataset(
        retrieval_artifact.h5,
        layout.retrieval_representation_additional_data(
            tile_id,
            representation_id,
            entry_id,
            name,
        ),
        array,
        dtype=array.dtype,
    )


# ---------------------------------------------------------------------------
# Deletion helpers
# ---------------------------------------------------------------------------


def delete_retrieval_representation(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Delete one retrieval entry group and all stored objects under it."""
    path = layout.retrieval_representation_entry_group(
        tile_id,
        representation_id,
        entry_id,
    )
    if path in retrieval_artifact.h5:
        del retrieval_artifact.h5[path]


def delete_all_retrieval_representations_for_representation(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Delete all entry groups for one retrieval configuration."""
    path = layout.retrieval_representation_group(tile_id, representation_id)
    if path in retrieval_artifact.h5:
        del retrieval_artifact.h5[path]


def delete_all_retrieval_representations_for_bag(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Delete all retrieval representations stored under one tile group."""
    path = layout.retrieval_representations_group(tile_id)
    if path in retrieval_artifact.h5:
        del retrieval_artifact.h5[path]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _retrieval_representation_entry_paths(
    tile_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: RetrievalH5Layout,
) -> dict[str, str]:
    """Return the main leaf paths for one retrieval entry."""
    return {
        "metadata": layout.retrieval_representation_metadata(
            tile_id,
            representation_id,
            entry_id,
        ),
        "params": layout.retrieval_representation_params(
            tile_id,
            representation_id,
            entry_id,
        ),
        "embedding": layout.retrieval_representation_embedding(
            tile_id,
            representation_id,
            entry_id,
        ),
    }


def _read_json_dataset(h5: h5py.File, path: str) -> dict[str, Any]:
    raw = h5[path][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _write_json_dataset(h5: h5py.File, path: str, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    _ensure_parent_group(h5, path)

    if path in h5:
        del h5[path]

    dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(path, data=payload, dtype=dtype)


def _ensure_parent_group(h5: h5py.File, path: str) -> None:
    parent_path = path.rsplit("/", 1)[0]
    h5.require_group(parent_path)


def _read_string_dataset(h5: h5py.File, path: str) -> str:
    raw = h5[path][()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _write_string_dataset(h5: h5py.File, path: str, value: str) -> None:
    _ensure_parent_group(h5, path)

    if path in h5:
        del h5[path]

    dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(path, data=str(value), dtype=dtype)
