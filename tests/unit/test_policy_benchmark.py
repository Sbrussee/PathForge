import pytest
from pathbench.config.config import Config, BenchmarkParameters
from pathbench.policy.benchmarking import BenchmarkingPolicy

def test_benchmark_grid_generation():
    # Mock a config with multiple parameters
    bp = BenchmarkParameters(
        mil=["ModelA", "ModelB"],
        loss=["LossX", "LossY"]
    )
    # Create minimal dummy config
    cfg = Config.from_dict({
        "experiment": {
            "project_name": "test", "annotation_file": "x", 
            "task": "classification", "mode": "benchmark"
        },
        "benchmark_parameters": {
            "mil": ["ModelA", "ModelB"],
            "loss": ["LossX", "LossY"]
        }
    })
    
    policy = BenchmarkingPolicy(cfg)
    configs = policy._generate_configs()
    
    # We expect len(mil) * len(loss) * len(seeds=3)
    # 2 * 2 * 3 = 12 configs
    assert len(configs) == 12
    
    # Verify naming injection
    model_names = set(c._active_model_name for c in configs)
    assert "ModelA" in model_names
    assert "ModelB" in model_names