from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from pathbench.core.io.h5.base import (
    FileHandleH5,
    exists,
    read_array_dataset,
    write_array_dataset,
)
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT, H5Layout


def retrieval_representation_exists(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    path = layout.retrieval_representation_main_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    if not exists(slide_artifact.h5, path):
        return False

    if expected_shape is None:
        return True

    dset = slide_artifact.h5[path]
    shape = getattr(dset, "shape", None)
    return tuple(shape) == tuple(expected_shape)


def retrieval_representation_item_exists(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether a full retrieval representation entry exists."""
    main_path = layout.retrieval_representation_main_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    sample_id_path = layout.retrieval_representation_sample_id_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    representation_type_path = layout.retrieval_representation_type_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    metadata_path = layout.retrieval_representation_metadata_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    slide_ids_path = layout.retrieval_representation_slide_ids_dataset(
        bag_id,
        representation_id,
        entry_id,
    )

    return all(
        exists(slide_artifact.h5, path)
        for path in (
            main_path,
            sample_id_path,
            representation_type_path,
            metadata_path,
            slide_ids_path,
        )
    )


def read_retrieval_representation(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    path = layout.retrieval_representation_main_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    return read_array_dataset(slide_artifact.h5, path)


def read_retrieval_representation_sample_id(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> str:
    path = layout.retrieval_representation_sample_id_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    return _read_string_dataset(slide_artifact.h5, path)


def write_retrieval_representation_sample_id(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    sample_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_sample_id_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    _write_string_dataset(slide_artifact.h5, path, sample_id)


def read_retrieval_representation_type(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> str:
    path = layout.retrieval_representation_type_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    return _read_string_dataset(slide_artifact.h5, path)


def write_retrieval_representation_type(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    representation_type: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_type_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    _write_string_dataset(slide_artifact.h5, path, representation_type)


def write_retrieval_representation(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    data: np.ndarray,
    *,
    dtype: np.dtype | str | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    array = np.asarray(data)
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    write_array_dataset(
        slide_artifact.h5,
        layout.retrieval_representation_main_dataset(
            bag_id,
            representation_id,
            entry_id,
        ),
        array,
        dtype=array.dtype,
    )


def read_retrieval_representation_metadata(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    path = layout.retrieval_representation_metadata_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    return _read_json_dataset(slide_artifact.h5, path)


def write_retrieval_representation_metadata(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    metadata: dict[str, Any],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_metadata_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    _write_json_dataset(slide_artifact.h5, path, metadata)


def read_retrieval_representation_params(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    path = layout.retrieval_representation_params_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    return _read_json_dataset(slide_artifact.h5, path)


def write_retrieval_representation_params(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    params: dict[str, Any],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_params_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    _write_json_dataset(slide_artifact.h5, path, params)


def read_retrieval_representation_slide_ids(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> list[str]:
    path = layout.retrieval_representation_slide_ids_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    return _read_string_array_dataset(slide_artifact.h5, path)


def write_retrieval_representation_slide_ids(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    slide_ids: list[str],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_slide_ids_dataset(
        bag_id,
        representation_id,
        entry_id,
    )
    _write_string_array_dataset(slide_artifact.h5, path, slide_ids)


def additional_retrieval_representation_data_exists(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    name: str,
    *,
    expected_shape: tuple[int, ...] | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    path = layout.retrieval_representation_additional_data_dataset(
        bag_id,
        representation_id,
        entry_id,
        name,
    )
    if not exists(slide_artifact.h5, path):
        return False

    if expected_shape is None:
        return True

    dset = slide_artifact.h5[path]
    shape = getattr(dset, "shape", None)
    return tuple(shape) == tuple(expected_shape)


def read_additional_retrieval_representation_data(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    path = layout.retrieval_representation_additional_data_dataset(
        bag_id,
        representation_id,
        entry_id,
        name,
    )
    return read_array_dataset(slide_artifact.h5, path)


def write_additional_retrieval_representation_data(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    name: str,
    data: np.ndarray,
    *,
    dtype: np.dtype | str | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    array = np.asarray(data)
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    write_array_dataset(
        slide_artifact.h5,
        layout.retrieval_representation_additional_data_dataset(
            bag_id,
            representation_id,
            entry_id,
            name,
        ),
        array,
        dtype=array.dtype,
    )


def delete_retrieval_representation(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_entry_group(
        bag_id,
        representation_id,
        entry_id,
    )
    if path in slide_artifact.h5:
        del slide_artifact.h5[path]


def delete_all_retrieval_representations_for_representation(
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representation_group(bag_id, representation_id)
    if path in slide_artifact.h5:
        del slide_artifact.h5[path]


def delete_all_retrieval_representations_for_bag(
    slide_artifact: FileHandleH5,
    bag_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    path = layout.retrieval_representations_group(bag_id)
    if path in slide_artifact.h5:
        del slide_artifact.h5[path]


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


def _read_string_array_dataset(h5: h5py.File, path: str) -> list[str]:
    values = h5[path][()]
    if values.ndim != 1:
        raise ValueError(f"slide_ids must have shape (N,). Got {values.shape}.")

    out: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _write_string_array_dataset(h5: h5py.File, path: str, values: list[str]) -> None:
    array = np.asarray(
        [str(value) for value in values],
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    _ensure_parent_group(h5, path)

    if path in h5:
        del h5[path]

    h5.create_dataset(path, data=array, dtype=array.dtype)


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
