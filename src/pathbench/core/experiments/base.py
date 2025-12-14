from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List
from itertools import product
import pandas as pd
import logging
import shutil
from datetime import datetime
import json

from ...config.config import Config
from ...utils.constants import EXPERIMENTS_DIR

import os

#from ..tasks.base import ClassificationTask, RegressionTask, ContinuousSurvivalTask, DiscreteSurvivalTask
#from ..annotations.annotations import (
#ClassificationAnnotation, RegressionAnnotation, SurvivalAnnotation, DiscreteSurvivalAnnotation,
#)

from pathbench.core.datasets.slides import SlideDataset

logger = logging.getLogger(__name__)

#TASK_TO_ANN = {
#"classification": (ClassificationTask, ClassificationAnnotation),
#"regression": (RegressionTask, RegressionAnnotation),
#"survival": (ContinuousSurvivalTask, SurvivalAnnotation),
#"survival_discrete": (DiscreteSurvivalTask, DiscreteSurvivalAnnotation),
#}

# inside Experiment.run() implementations, e.g., BenchmarkingExperiment.run():
#TCls, ACls = TASK_TO_ANN[self.cfg.experiment.task]
#assert isinstance(self.cfg.task_obj, TCls) if hasattr(self.cfg, "task_obj") else True
#ann = ACls()
#rows = ann.read(self.cfg.experiment.annotation_file)

class ComboConfig:
    """
    Generic, dynamically-populated combo.

    Any key you pass in becomes an attribute:
        Combo(feature_extraction="virchow", tile_px=256)
        -> combo.feature_extraction, combo.tile_px
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_keys_values(cls, keys: list[str], values: list[object]) -> "ComboConfig":
        data = {k: v for k, v in zip(keys, values)}
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(slots=True)
class Experiment:
    """
    Experiment class, which manages project structure and metadata.

    Responsibilities:
    - Determine project_root.
    - Ensure project structure exists:
        * project.json
        * annotations.csv (copied into project_root)
        * datasets.json with absolute features_dir / tile_records_dir
    - Provide utilities to compute benchmark parameter combinations.
    """

    cfg: Config
    project_root: str | None = None

    def __post_init__(self) -> None:
        """
        After construction:
        - decide project_root
        - ensure project metadata & dataset config exist
        """
        self.project_root = self._determine_project_root()
        self._prepare_project()

    def _determine_project_root(self) -> str:
        """
        Determine absolute path to the project root.

        Rules:
        - cfg.experiment.project_name must be a non-empty string.
        - If cfg.experiment.project_root exists and is not None:
            * it MUST be an absolute path; otherwise error.
        - Else: use <cwd>/experiments/{project_name}.
        """
        exp_cfg = self.cfg.experiment

        project_name = getattr(exp_cfg, "project_name", None)
        if not project_name or not isinstance(project_name, str):
            raise ValueError(
                "cfg.experiment.project_name must be a non-empty string."
            )

        # Optional: explicit project_root in config
        project_root = getattr(exp_cfg, "project_root", None)

        if project_root is not None:
            # Strict: must be absolute, no clever guessing
            if not os.path.isabs(project_root):
                raise ValueError(
                    "cfg.experiment.project_root must be an absolute path "
                    f"(got: {project_root!r})."
                )
            root = project_root
        else:
            base = os.path.join(os.getcwd(), "experiments")
            root = os.path.join(base, project_name)

        root_abs = os.path.abspath(root)
        logger.info("Using project_root: %s", root_abs)
        return root_abs

    def _prepare_project(self) -> None:
        """
        Ensure the project structure exists and is consistent.

        Steps (idempotent):
        1. Create project_root directory if it does not exist.
        2. Ensure project.json exists (create if missing).
        3. Ensure annotations.csv exists in project_root
           (copy from cfg.experiment.annotation_file if missing).
        4. Ensure datasets.json exists:
           - if present: load it and sync cfg.datasets from it.
           - if absent: create it from cfg.datasets and write absolute
             features_dir / tile_records_dir paths.
        """
        if self.project_root is None:
            raise RuntimeError("project_root is not set.")

        root = self.project_root
        os.makedirs(root, exist_ok=True)

        # 1) project.json
        project_json_path = os.path.join(root, "project.json")
        if not os.path.exists(project_json_path):
            self._write_project_json(project_json_path)
        else:
            # We only sanity-check; not using these values for logic yet.
            self._load_project_json(project_json_path)

        # 2) annotations.csv
        annotations_target = os.path.join(root, "annotations.csv")
        if not os.path.exists(annotations_target):
            self._copy_annotations(annotations_target)
    
    def _write_project_json(self, path: str) -> None:
        """
        Create a minimal project.json with basic metadata.

        Raises:
            FileNotFoundError if the configured annotation_file does not exist.
        """
        ann_src = self.cfg.experiment.annotation_file
        if not os.path.isfile(ann_src):
            raise FileNotFoundError(
                f"experiment.annotation_file does not exist: {ann_src}"
            )

        data = {
            "project_name": self.cfg.experiment.project_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "annotation_source": os.path.abspath(ann_src),
        }

        logger.info("Creating project.json at %s", path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_project_json(self, path: str) -> None:
        """
        Load and lightly validate project.json.

        Currently only checks that project_name matches cfg.experiment.project_name.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pj_name = data.get("project_name")
        cfg_name = self.cfg.experiment.project_name
        if pj_name != cfg_name:
            raise ValueError(
                f"project.json project_name={pj_name!r} does not match "
                f"cfg.experiment.project_name={cfg_name!r}."
            )

        # Could add more checks later if needed.
        logger.debug("Loaded project.json from %s", path)

    def _copy_annotations(self, target: str) -> None:
        """
        Copy annotations CSV from cfg.experiment.annotation_file into project_root.

        Raises:
            FileNotFoundError if the source annotation file does not exist.
        """
        src = self.cfg.experiment.annotation_file
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"experiment.annotation_file does not exist: {src}"
            )

        logger.info("Copying annotations from %s to %s", src, target)
        shutil.copy2(src, target)
        
    def load_annotations(self) -> pd.DataFrame:
        """
        Load annotations CSV from project_root/annotations.csv.

        Returns:
            DataFrame with annotations.
        """
        if self.project_root is None:
            raise RuntimeError("project_root is not set.")
        
        ann_path = os.path.join(self.project_root, "annotations.csv")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")
        
        df = pd.read_csv(ann_path)
        return df
    
    def build_combinations(self, keys: List[str]) -> List[ComboConfig]:
        """
        Build all combinations of benchmark parameters for the given keys.
        Args:
            keys: List of field names in cfg.benchmark_parameters to build combinations for.
        Returns:
            List of ComboConfig instances representing all combinations.
        """
        bp = self.cfg.benchmark_parameters

        value_lists: List[list[Any]] = []
        for key in keys:
            if not hasattr(bp, key):
                raise AttributeError(f"benchmark_parameters has no field '{key}'")
            values = getattr(bp, key)

            if not values:
                raise ValueError(f"benchmark_parameters.{key} is empty; cannot build grid.")
            value_lists.append(values)

        combos: List[ComboConfig] = []
        for vals in product(*value_lists):
            combos.append(ComboConfig.from_keys_values(keys, list(vals)))

        return combos
    
    def build_datasets(self) -> list[SlideDataset]:
        """
        Build SlideDataset instances for all datasets in cfg.datasets.
        Returns:
            List of SlideDataset instances.
        """
        if self.project_root is None:
            raise RuntimeError("project_root is not set.")
        
        annotations = self.load_annotations()
        
        datasets = []
        for ds in self.cfg.datasets:
            if ds.used_for == "ignore":
                continue
            datasets.append(SlideDataset(ds, annotations))
        return datasets