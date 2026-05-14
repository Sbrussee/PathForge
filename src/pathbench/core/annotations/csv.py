from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pathbench.core.annotations.base import AnnotationsBase

logger = logging.getLogger(__name__)


class CSVAnnotations(AnnotationsBase):
    """
    CSV-backed annotation adapter.

    The adapter loads a CSV file into a column-oriented dictionary where each key
    is a column name and each value is a list of row values for that column.

    Example:
        >>> annotations = CSVAnnotations("annotations.csv")
        >>> sorted(annotations.annotations)
        ['category', 'dataset', 'patient', 'slide']
    """

    def __init__(self, path_to_csv: str):
        self.contains_wsi_path = False
        self.contains_patient = False
        self.contains_dataset = False

        self.annotations = self.load_annotations(path_to_csv)
        if not self.validate_annotations(self.annotations):
            raise ValueError("Invalid annotations provided, see logs above.")

        logger.info("CSV annotations successfully loaded and validated.")

    def load_annotations(self, id: str) -> dict[str, list[Any]]:
        """
        Load annotations from a CSV file.

        Args:
            id: Path to a CSV file.

        Returns:
            Column-oriented annotation dictionary where each value list has shape
            ``[num_rows]``.
        """

        return pd.read_csv(id, sep=None, engine="python").to_dict(orient="list")

    def save_annotations(self, id: str, annotations: dict[str, list[Any]]) -> None:
        """
        Persist annotations to a CSV file.

        Args:
            id: Output CSV path.
            annotations: Column-oriented annotation dictionary.
        """

        pd.DataFrame(annotations).to_csv(id, index=False)

    def validate_annotations(self, annotations: dict[str, list[Any]]) -> bool:
        """
        Validate and normalize a PathBench annotation table.

        Required columns:
            - ``slide``: slide identifier per row
            - ``category``: task label per row

        Optional columns filled or tracked here:
            - ``patient``: defaults to ``slide`` when missing
            - ``dataset``: defaults to ``"default"`` when missing
            - ``wsi_path``: tracked for direct slide resolution when present
        """

        assert isinstance(annotations, dict), "Annotations should be a dictionary."
        for key, value in annotations.items():
            assert isinstance(value, list), (
                f"Annotation for key '{key}' should be a list."
            )

        assert "slide" in annotations, (
            "Column 'slide' is missing from your annotations file."
        )
        assert "category" in annotations, (
            "Column 'category' is missing from your annotations file."
        )

        if "patient" not in annotations:
            logger.warning(
                "Column 'patient' is missing from the annotations file; "
                "defaulting patient IDs to the slide column."
            )
            annotations["patient"] = list(annotations["slide"])
            self.contains_patient = False
        else:
            logger.info("Column 'patient' found in annotations; using it for patient IDs.")
            self.contains_patient = True

        if "wsi_path" not in annotations:
            logger.warning(
                "Column 'wsi_path' is missing from the annotations file; "
                "PathBench will resolve slides from the configured slides_dir."
            )
            self.contains_wsi_path = False
        else:
            logger.info("Column 'wsi_path' found in annotations; using direct slide paths.")
            self.contains_wsi_path = True

        if "dataset" not in annotations:
            logger.warning(
                "Column 'dataset' is missing from the annotations file; "
                "assigning all rows to dataset='default'."
            )
            annotations["dataset"] = ["default"] * len(
                next(iter(annotations.values()), [])
            )
            self.contains_dataset = False
        else:
            logger.info(
                "Found datasets in annotations: %s",
                ", ".join(sorted({str(value) for value in annotations["dataset"]})),
            )
            self.contains_dataset = True

        return True

    def inspect_annotations(self, annotations: dict[str, list[Any]]) -> None:
        """
        Log a compact summary of the loaded annotations.

        Args:
            annotations: Column-oriented annotation dictionary with lists shaped
                ``[num_rows]``.
        """

        n_entries = len(next(iter(annotations.values()), []))
        logger.info(
            "Annotations contain %d entries with columns: %s",
            n_entries,
            list(annotations.keys()),
        )
