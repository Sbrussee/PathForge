import argparse
from ..config.config import Config
from ..core.experiments.base import Experiment
from ..policy.benchmarking import BenchmarkingPolicy

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = Config.from_yaml(args.config)
    experiment = Experiment(cfg)
    out = BenchmarkingPolicy(experiment).execute()
    print(out)