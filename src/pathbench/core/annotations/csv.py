from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pathbench.core.annotations.base import AnnotationsBase

logger = logging.getLogger(__name__)


class CSVAnnotations(AnnotationsBase):
    """
    Read, validate, and persist annotation dictionaries from CSV files.
    """

    def __init__(self, path_to_csv: str):
        self.contains_wsi_path = False
        self.contains_patient = False
        self.contains_dataset = False

        self.annotations = self.load_annotations(path_to_csv)
        valid = self.validate_annotations(self.annotations)
        if not valid:
            raise ValueError("Invalid annotations provided, see logs above.")
        logger.info("CSV annotations successfully loaded and validated.")

    def load_annotations(self, path: str) -> dict[str, list[Any]]:
        """Load annotations from a CSV file."""
        return pd.read_csv(path, sep=None, engine="python").to_dict(orient="list")

    def save_annotations(self, path: str, annotations: dict[str, list[Any]]) -> None:
        """Save annotations to a CSV file."""
        pd.DataFrame(annotations).to_csv(path, index=False)

    def validate_annotations(self, annotations: dict[str, list[Any]]) -> bool:
        """Validate and normalize the annotation dictionary in-place."""
        if not isinstance(annotations, dict):
            raise TypeError("Annotations should be a dictionary.")
        for key, value in annotations.items():
            if not isinstance(value, list):
                raise TypeError(f"Annotation for key '{key}' should be a list.")

        if "slide" not in annotations:
            raise ValueError("Column 'slide' is missing from the annotations file.")
        if "category" not in annotations:
            raise ValueError("Column 'category' is missing from the annotations file.")

        if "patient" not in annotations:
            logger.warning(
                "Column 'patient' is missing from annotations; remapping to 'slide'."
            )
            annotations["patient"] = annotations["slide"]
            self.contains_patient = False
        else:
            logger.info("Column 'patient' found in annotations.")
            self.contains_patient = True

        if "wsi_path" not in annotations:
            logger.warning(
                "Column 'wsi_path' is missing from annotations; slide paths will be resolved from config."
            )
            self.contains_wsi_path = False
        else:
            logger.info("Column 'wsi_path' found in annotations.")
            self.contains_wsi_path = True

        if "dataset" not in annotations:
            logger.warning(
                "Column 'dataset' is missing from annotations; all slides mapped to 'default'."
            )
            annotations["dataset"] = ["default"] * len(
                next(iter(annotations.values()), [])
            )
            self.contains_dataset = False
        else:
            logger.info(
                "Found datasets in annotations: %s",
                ", ".join(sorted(set(annotations["dataset"]))),
            )
            self.contains_dataset = True

        return True

    def inspect_annotations(self, annotations: dict[str, list[Any]]) -> None:
        """Print a summary of the annotations."""
        n_entries = len(next(iter(annotations.values()), []))
        print(
            f"Annotations contain {n_entries} entries with columns: {list(annotations.keys())}"
        )
