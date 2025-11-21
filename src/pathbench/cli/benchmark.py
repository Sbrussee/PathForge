import argparse
from ..config.config import Config
from ..core.experiments.base import BenchmarkingExperiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = Config.from_yaml(args.config)
    out = BenchmarkingExperiment(cfg).run()
    print(out)