from pathbench.config.config import Config


def test_yaml_loads(example_yaml_path="configs/config.example.yaml"):
cfg = Config.from_yaml(example_yaml_path)
assert cfg.experiment.task in {"classification","regression","survival","survival_discrete"}