from pathlib import Path

import numpy as np
import pytest

from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval.input_inspection import inspect_retrieval_inputs


def _artifact(path: Path, *, feature_shape: tuple[int, int], coord_rows: int) -> Path:
    with FileHandleH5(path, mode="a") as artifact:
        tiles_io.write_coords(artifact, "tiles", np.zeros((coord_rows, 5), dtype=np.int32))
        features_io.write_features(artifact, "tiles", "uni", np.zeros(feature_shape, dtype=np.float32))
    return path


def test_inspect_retrieval_inputs_classifies_aligned_multi_row_features_as_patch(tmp_path: Path) -> None:
    inspection = inspect_retrieval_inputs([_artifact(tmp_path / "patch.h5", feature_shape=(2, 3), coord_rows=2)], tiling_id="tiles", extractor_name="uni")

    assert inspection.feature_level == "patch"
    assert inspection.feature_dimension == 3


def test_inspect_retrieval_inputs_classifies_one_row_features_as_slide_regardless_of_coords(tmp_path: Path) -> None:
    inspection = inspect_retrieval_inputs([_artifact(tmp_path / "slide.h5", feature_shape=(1, 3), coord_rows=1)], tiling_id="tiles", extractor_name="uni")

    assert inspection.feature_level == "slide"


@pytest.mark.parametrize("feature_shape,coord_rows", [((0, 3), 1), ((2, 0), 2), ((2, 3), 1)])
def test_inspect_retrieval_inputs_rejects_empty_or_misaligned_inputs(tmp_path: Path, feature_shape: tuple[int, int], coord_rows: int) -> None:
    artifact = _artifact(tmp_path / "invalid.h5", feature_shape=feature_shape, coord_rows=coord_rows)

    with pytest.raises(ValueError):
        inspect_retrieval_inputs([artifact], tiling_id="tiles", extractor_name="uni")


def test_inspect_retrieval_inputs_rejects_patch_slide_mix(tmp_path: Path) -> None:
    paths = [_artifact(tmp_path / "patch.h5", feature_shape=(2, 3), coord_rows=2), _artifact(tmp_path / "slide.h5", feature_shape=(1, 3), coord_rows=2)]

    with pytest.raises(ValueError, match="mixed"):
        inspect_retrieval_inputs(paths, tiling_id="tiles", extractor_name="uni")
