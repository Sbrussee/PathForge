from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4

import logging
import os
import shutil
import tempfile

import h5py

from pathbench.core.io.slide_artifacts.base import FileHandleH5

logger = logging.getLogger(__name__)

ArtifactValidator = Callable[[FileHandleH5], None]


def validate_h5_artifact(path: str | Path) -> None:
    artifact_path = Path(path)
    with h5py.File(artifact_path, "r") as h5_file:
        _ = list(h5_file.keys())
        h5_file.visititems(lambda _name, _obj: None)


def quarantine_corrupt_artifact(artifact_path: str | Path, *, reason: Exception | None = None) -> Path:
    path = Path(artifact_path)
    corrupt_dir = path.parent / "_corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    quarantined_path = corrupt_dir / (
        f"{path.stem}.corrupt-{timestamp}-pid{os.getpid()}-{uuid4().hex[:8]}{path.suffix}"
    )
    os.replace(path, quarantined_path)

    if reason is None:
        logger.warning("[H5] Quarantined corrupt artifact: %s -> %s", path, quarantined_path)
    else:
        logger.warning(
            "[H5] Quarantined corrupt artifact: %s -> %s (%s)",
            path,
            quarantined_path,
            reason,
        )
    return quarantined_path


def ensure_artifact_readable_or_quarantine(artifact_path: str | Path) -> bool:
    path = Path(artifact_path)
    if not path.is_file():
        return False

    try:
        validate_h5_artifact(path)
    except (OSError, RuntimeError, ValueError) as exc:
        quarantine_corrupt_artifact(path, reason=exc)
        return False

    return True


@contextmanager
def atomic_slide_artifact_write(
    artifact_path: str | Path,
    *,
    validate: ArtifactValidator | None = None,
) -> Iterator[FileHandleH5]:
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix or ".h5",
    )
    os.close(fd)
    temp_path = Path(temp_name)
    seeded_from_live = False

    try:
        if path.exists():
            try:
                validate_h5_artifact(path)
            except (OSError, RuntimeError, ValueError) as exc:
                quarantine_corrupt_artifact(path, reason=exc)
            else:
                shutil.copy2(path, temp_path)
                seeded_from_live = True

        if not seeded_from_live and temp_path.exists():
            temp_path.unlink()

        with FileHandleH5(temp_path, mode="a") as slide_artifact:
            yield slide_artifact

        validate_h5_artifact(temp_path)
        if validate is not None:
            with FileHandleH5(temp_path, mode="r") as slide_artifact:
                validate(slide_artifact)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
