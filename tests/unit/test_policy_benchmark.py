import pandas as pd
from types import SimpleNamespace
from pathbench.policy.benchmarking import BenchmarkingPolicy
from pathbench.config.config import Config


class DummyExperiment:
    def __init__(self, cfg: Config, annotations: pd.DataFrame, project_root: str):
        self.cfg = cfg
        self._annotations = annotations
        self.project_root = project_root

    def build_datasets(self):
        return [SimpleNamespace(name="train", used_for="training", features_dir=self.project_root)]

    def load_annotations(self):
        return self._annotations

    ann_path = tmp_path / "annotations.csv"
    annotations.to_csv(ann_path, index=False)

    cfg = Config.from_dict(
        {
            "experiment": {
                "project_name": "test",
                "annotation_file": str(ann_path),
                "task": "classification",
                "mode": "benchmark",
            },
            "datasets": [
                {
                    "name": "train",
                    "slide_path": str(tmp_path),
                    "used_for": "training",
                }
            ],
            "benchmark_parameters": {
                "mil": ["ModelA", "ModelB"],
                "loss": ["LossX", "LossY"],
            },
        }
    )
    
    exp = DummyExperiment(cfg, annotations, str(tmp_path))
    policy = BenchmarkingPolicy(exp)
    combos = policy._build_combo_grid()

    assert len(combos) == 4  # 2 models * 2 losses
    model_names = {getattr(c, "mil", None) for c in combos}
    assert {"ModelA", "ModelB"}.issubset(model_names)