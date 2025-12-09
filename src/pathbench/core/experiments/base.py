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
    cfg: Config

    def initialize_project(self) -> tuple[str, pd.DataFrame, list[SlideDataset]]:
        """
        Standard entry point for all experiments.

        - Determine project_root.
        - If dataset_config.json exists:
            * Load annotations from the copied CSV in project_root.
            * Rebuild SlideDatasets from dataset_config.json.
        - Else:
            * Load annotations from cfg.experiment.annotation_file.
            * Copy annotations into project_root.
            * Build SlideDatasets via _build_slide_datasets.
            * Write dataset_config.json.
        """
        project_root = self.get_project_root()
        ds_cfg_path = os.path.join(project_root, "dataset_config.json")

        # ---- reuse existing project ----
        if os.path.exists(ds_cfg_path):
            logger.info("[EXP] Found existing dataset_config.json at %s, reusing it", ds_cfg_path)
            with open(ds_cfg_path, "r") as f:
                meta = json.load(f)

            ann_file = meta.get("annotations_file")
            if ann_file:
                ann_path = os.path.join(project_root, ann_file)
                if os.path.exists(ann_path):
                    logger.info("[EXP] Loading annotations from %s", ann_path)
                    annotations = pd.read_csv(ann_path)
                else:
                    logger.warning(
                        "[EXP] Copied annotations file missing (%s), "
                        "falling back to original path %s",
                        ann_path,
                        self.cfg.experiment.annotation_file,
                    )
                    annotations = pd.read_csv(self.cfg.experiment.annotation_file)
            else:
                logger.warning(
                    "[EXP] 'annotations_file' missing in dataset_config.json, "
                    "falling back to original path %s",
                    self.cfg.experiment.annotation_file,
                )
                annotations = pd.read_csv(self.cfg.experiment.annotation_file)

            # rebuild slide datasets from meta
            name_to_ds_cfg = {d.name: d for d in self.cfg.datasets}
            datasets: list[SlideDataset] = []

            for ds_entry in meta.get("datasets", []):
                name = ds_entry["name"]
                ds_cfg = name_to_ds_cfg.get(name)
                if ds_cfg is None:
                    logger.warning(
                        "[EXP] Dataset '%s' in dataset_config.json not found in current cfg.datasets; skipping",
                        name,
                    )
                    continue
                samples_data = ds_entry.get("samples", [])
                ds = SlideDataset.from_samples(ds_cfg, samples_data)
                datasets.append(ds)

            return project_root, annotations, datasets

        # ---- first run: build from scratch ----
        logger.info("[EXP] No dataset_config.json found; initializing project from scratch")
        annotations = pd.read_csv(self.cfg.experiment.annotation_file)
        ann_dst = self._copy_annotations_to_project_root(project_root)
        ann_filename = os.path.basename(ann_dst)

        datasets = self._build_slide_datasets(annotations)
        self._write_dataset_config(project_root, datasets, ann_filename)
        return project_root, annotations, datasets

    def get_project_root(self) -> str:
        """
        Return the root folder for this project/experiment.

        Default:
            <repo-root>/experiments/{project_name}

        If cfg.experiment.project_root exists and is non-empty,
        that is used instead.
        """
        exp_cfg = self.cfg.experiment

        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        root = os.path.join(repo_root, EXPERIMENTS_DIR, exp_cfg.project_name)

        os.makedirs(root, exist_ok=True)
        logger.info("[EXP] Using project_root=%s", root)

        return root
    
    def _copy_annotations_to_project_root(self, project_root: str) -> str:
        """
        Copy the annotations CSV into the project folder.

        Returns the destination path (inside project_root).
        """
        src = self.cfg.experiment.annotation_file
        dst = os.path.join(project_root, os.path.basename(src))

        if not os.path.isfile(src):
            logger.warning("[EXP] Annotation file does not exist: %s", src)
            return dst

        # Avoid copying onto itself if already in project_root
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
            logger.info("[EXP] Copied annotations: %s -> %s", src, dst)
        else:
            logger.info("[EXP] Annotation file already in project_root: %s", dst)

        return dst
    
    def _write_dataset_config(
        self,
        project_root: str,
        datasets: list["SlideDataset"],
        annotations_filename: str,
    ) -> str:
        """
        Create dataset_config.json in the project folder.

        Contains basic info so we can later reconstruct datasets
        without globbing from scratch.
        """
        cfg = {
            "project_name": self.cfg.experiment.project_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "annotations_file": annotations_filename,
            "datasets": [],
        }

        for ds in datasets:
            ds_entry = {
                "name": ds.name,
                "used_for": ds.used_for,
                "slide_path": ds.config.slide_path,
                "num_samples": len(ds),
                "samples": [],
            }

            for sample in ds.samples:
                ds_entry["samples"].append(
                    {
                        "slide": sample.slide,
                        "patient": sample.patient,
                        "category": sample.category,
                        "wsi_path": sample.wsi_path,
                    }
                )

            cfg["datasets"].append(ds_entry)

        out_path = os.path.join(project_root, "dataset_config.json")
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)

        logger.info("[EXP] Wrote dataset_config.json to %s", out_path)

        return out_path
    
    def _compute_combinations(self, keys: List[str]) -> List[ComboConfig]:
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
    
    def _build_slide_datasets(self, annotations: pd.DataFrame) -> list[SlideDataset]:
        datasets = []
        for ds in self.cfg.datasets:
            if ds.used_for == "ignore":
                continue
            datasets.append(SlideDataset(ds, annotations))
        return datasets

    def run(self) -> dict[str, Any]:
        raise NotImplementedError

class FeatureExtractionExperiment(Experiment):
    
    def run(self) -> dict[str, Any]:
        from pathbench.policy.feature_extraction import FeatureExtractionPolicy
        
        # 1) Create-or-load project root, annotations, and datasets
        self.project_root, self.annotations, self.datasets = self.initialize_project()
        logger.info("[FE] Project root: %s", self.project_root)
        logger.info("[FE] Built %d SlideDatasets", len(self.datasets))
        for ds in self.datasets:
            logger.info(
                "[FE] Dataset '%s' (used_for=%s) -> %d slides",
                ds.name,
                ds.used_for,
                len(ds),
            )

        # 2) Compute benchmark parameter combinations
        bp_combos = self._compute_combinations(["feature_extraction", "tile_px", "tile_mpp"])
        logger.info("[FE] Number of benchmark parameter combos: %d", len(bp_combos))
        for i, combo in enumerate(bp_combos):
            logger.debug("[FE] Combo %d: %s", i, combo.to_dict())

        # 3) Run policy
        policy = FeatureExtractionPolicy(
            config=self.cfg,
            datasets=self.datasets,
        )

        for i, combo_cfg in enumerate(bp_combos):
            logger.info(
                "[FE] === Running combo %d/%d: %s ===",
                i + 1,
                len(bp_combos),
                combo_cfg.to_dict(),
            )
            policy.execute(combo_cfg)
        
        logger.info("[FE] Feature extraction DONE.")

        return {"status": "feature_extraction_done"}

class BenchmarkingExperiment(Experiment):
    def run(self):
        # prepare data splits, train/eval, aggregate reports
        return {"status": "benchmark_done"}

class OptimizationExperiment(Experiment):
    def run(self):
        # build optuna study based on cfg.search_space_path
        return {"status": "optimize_done"}