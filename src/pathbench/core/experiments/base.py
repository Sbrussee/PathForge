from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
import json
import logging
import shutil
from pathlib import Path
from typing import Any, List

import pandas as pd

from ...config.config import Config
from ...utils.constants import EXPERIMENTS_DIR
from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.core.datasets.bag_dataset import BagDataset

logger = logging.getLogger(__name__)


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
    Manage experiment project_root and experiment-level metadata files.

    Creates/validates:
    - project.json
    - annotations.csv (copied into project_root)

    Provides helpers to load annotations, build parameter combos, and build datasets.
    Dataset-scoped artifacts (per-slide .h5) are managed via dataset config, not here.
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
        Resolve absolute project_root as:
        - <experiment.project_root>/<project_name> if provided (must be absolute), else
        - <repo_root>/{EXPERIMENTS_DIR}/<project_name>.
        """
        exp_cfg = self.cfg.experiment

        project_name = getattr(exp_cfg, "project_name", None)
        if not project_name or not isinstance(project_name, str):
            raise ValueError("cfg.experiment.project_name must be a non-empty string.")

        project_root = getattr(exp_cfg, "project_root", None)

        if project_root is not None:
            base = Path(project_root)
            if not base.is_absolute():
                raise ValueError(
                    "cfg.experiment.project_root must be an absolute path "
                    f"(got: {project_root!r})."
                )
        else:
            # Resolve repo root robustly (avoid cwd). Assumes repo layout: <repo>/src/pathbench/...
            this_file = Path(__file__).resolve()
            cur = this_file.parent
            repo_root: Path | None = None

            # Walk up until we find the 'src' directory, then take its parent as repo root
            while True:
                if cur.name == "src":
                    repo_root = cur.parent
                    break
                parent = cur.parent
                if parent == cur:
                    break
                cur = parent

            if repo_root is None:
                # Fallback: use current working directory as last resort
                repo_root = Path.cwd()

            base = repo_root / str(EXPERIMENTS_DIR)

        root_abs = (base / project_name).resolve()

        logger.info("Using project_root: %s", root_abs)
        return str(root_abs)

    def _prepare_project(self) -> None:
        """Ensure project_root exists and contains project.json and annotations.csv."""
        if self.project_root is None:
            raise RuntimeError("project_root is not set.")

        root = Path(self.project_root)
        root.mkdir(parents=True, exist_ok=True)

        # 1) project.json
        project_json_path = root / "project.json"
        if not project_json_path.exists():
            self._write_project_json(project_json_path)
        else:
            # We only sanity-check; not using these values for logic yet.
            self._load_project_json(project_json_path)

        # 2) annotations.csv
        annotations_target = root / "annotations.csv"
        if not annotations_target.exists():
            self._copy_annotations(annotations_target)

    def _write_project_json(self, path: Path) -> None:
        """
        Create a minimal project.json with basic metadata.

        Raises:
            FileNotFoundError if the configured annotation_file does not exist.
        """
        ann_src = Path(self.cfg.experiment.annotation_file)
        if not ann_src.is_file():
            raise FileNotFoundError(
                f"experiment.annotation_file does not exist: {ann_src}"
            )

        data = {
            "project_name": self.cfg.experiment.project_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "annotation_source": str(ann_src.resolve()),
        }

        logger.info("Creating project.json at %s", path)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_project_json(self, path: Path) -> None:
        """
        Load and lightly validate project.json.

        Currently only checks that project_name matches cfg.experiment.project_name.
        """
        data = json.loads(path.read_text(encoding="utf-8"))

        pj_name = data.get("project_name")
        cfg_name = self.cfg.experiment.project_name
        if pj_name != cfg_name:
            raise ValueError(
                f"project.json project_name={pj_name!r} does not match "
                f"cfg.experiment.project_name={cfg_name!r}."
            )

        logger.debug("Loaded project.json from %s", path)

    def _copy_annotations(self, target: Path) -> None:
        """
        Copy annotations CSV from cfg.experiment.annotation_file into project_root.

        Raises:
            FileNotFoundError if the source annotation file does not exist.
        """
        src = Path(self.cfg.experiment.annotation_file)
        if not src.is_file():
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

        ann_path = Path(self.project_root) / "annotations.csv"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")

        return pd.read_csv(ann_path)

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

    def build_datasets(self) -> list[WSIDataset]:
        """
        Build WSIDataset instances for all datasets in cfg.datasets.
        Returns:
            List of SlideDataset instances.
        """
        if self.project_root is None:
            raise RuntimeError("project_root is not set.")

        annotations = self.load_annotations()

        datasets: list[WSIDataset] = []
        for ds in self.cfg.datasets:
            if ds.used_for == "ignore":
                continue
            datasets.append(WSIDataset(ds, annotations))
        return datasets

    def build_bag_datasets(
        self,
        combo_cfg: ComboConfig,
        target_column: str | None = None,
    ) -> list[BagDataset]:
        """
        Build BagDataset instances for all datasets in cfg.datasets.

        Args:
            combo_cfg: Active benchmark combination.
            target_column: Optional override for the target/category column.

        Returns:
            List of BagDataset instances.
        """
        if self.project_root is None:
            raise RuntimeError("project_root is not set.")

        annotations = self.load_annotations()

        datasets: list[BagDataset] = []
        for ds in self.cfg.datasets:
            if ds.used_for == "ignore":
                continue

            kwargs = {
                "ds_cfg": ds,
                "annotations_df": annotations,
                "combo_cfg": combo_cfg,
                "aggregation_level": self.cfg.experiment.aggregation_level,
            }

            if target_column is not None:
                kwargs["target_column"] = target_column

            datasets.append(BagDataset(**kwargs))

        return datasets