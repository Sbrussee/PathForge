from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from ...config.config import Config


from ..tasks.base import ClassificationTask, RegressionTask, ContinuousSurvivalTask, DiscreteSurvivalTask
from ..annotations.annotations import (
ClassificationAnnotation, RegressionAnnotation, SurvivalAnnotation, DiscreteSurvivalAnnotation,
)


TASK_TO_ANN = {
"classification": (ClassificationTask, ClassificationAnnotation),
"regression": (RegressionTask, RegressionAnnotation),
"survival": (ContinuousSurvivalTask, SurvivalAnnotation),
"survival_discrete": (DiscreteSurvivalTask, DiscreteSurvivalAnnotation),
}


# inside Experiment.run() implementations, e.g., BenchmarkingExperiment.run():
TCls, ACls = TASK_TO_ANN[self.cfg.experiment.task]
assert isinstance(self.cfg.task_obj, TCls) if hasattr(self.cfg, "task_obj") else True
ann = ACls()
rows = ann.read(self.cfg.experiment.annotation_file)

@dataclass(slots=True)
class Experiment:
    cfg: Config
    def run(self) -> Any:  # pragma: no cover
        raise NotImplementedError

class FeatureExtractionExperiment(Experiment):
    def run(self):
        # iterate slides, run extractor, persist features
        return {"status": "features_done"}

class BenchmarkingExperiment(Experiment):
    def run(self):
        # prepare data splits, train/eval, aggregate reports
        return {"status": "benchmark_done"}

class OptimizationExperiment(Experiment):
    def run(self):
        # build optuna study based on cfg.search_space_path
        return {"status": "optimize_done"}