from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathforge.core.io.h5 import descriptors as descriptors_io
from pathforge.core.io.h5.base import FileHandleH5, write_array_dataset


def test_descriptor_roundtrip_and_validation(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"
    descriptor_name = "mean_rgb"
    descriptor_in = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=np.float32)

    with FileHandleH5(h5_path, mode="a") as slide_artifact:
        assert (
            descriptors_io.descriptor_exists(slide_artifact, bag_id, descriptor_name)
            is False
        )

        descriptors_io.write_descriptor(
            slide_artifact,
            bag_id,
            descriptor_name,
            descriptor_in,
        )

        assert descriptors_io.descriptor_exists(
            slide_artifact,
            bag_id,
            descriptor_name,
            expected_rows=2,
            expected_dim=3,
        )

        descriptor_out = descriptors_io.read_descriptor(
            slide_artifact,
            bag_id,
            descriptor_name,
        )
        np.testing.assert_allclose(descriptor_out, descriptor_in)


def test_read_descriptor_rejects_non_2d_dataset(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"
    descriptor_name = "mean_rgb"

    with FileHandleH5(h5_path, mode="a") as slide_artifact:
        write_array_dataset(
            slide_artifact.h5,
            f"bags/{bag_id}/descriptors/{descriptor_name}",
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            dtype=np.float32,
        )

        with pytest.raises(ValueError, match="must have shape"):
            _ = descriptors_io.read_descriptor(slide_artifact, bag_id, descriptor_name)
