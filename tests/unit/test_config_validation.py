import pytest
from pydantic import ValidationError
from pathbench.config.config import Config, BenchmarkParameters
from pathbench.utils.registries import FEATURE_EXTRACTORS, LAZYSLIDE_MODEL_NAMES, MODELS
from pathbench.core.models.mil_base import MILModelBase

# --- Mocks ---
LAZYSLIDE_MODEL_NAMES.add("lazy_specific_model")

@FEATURE_EXTRACTORS.register("lazy_specific_model")
def lazy_model(): return "lazy"

@FEATURE_EXTRACTORS.register("generic_model")
def generic_model(): return "generic"

@MODELS.register("MockMIL")
class MockMIL(MILModelBase):
    @property
    def bag_size(self): return None
    def forward_bag(self, x, **kwargs): return x

# --- Tests ---

def test_valid_benchmark_params():
    bp = BenchmarkParameters(
        tile_px=[256, 512],
        tile_um=[0.5],
        feature_extraction=["generic_model"],
        mil=["MockMIL"],
        activation_function=["ReLU"],
        optimizer=["Adam"]
    )
    assert bp.tile_px == [256, 512]

def test_invalid_tile_px():
    with pytest.raises(ValidationError) as excinfo:
        BenchmarkParameters(tile_px=[255])
    assert "divisible by 2" in str(excinfo.value)

def test_invalid_tile_um():
    with pytest.raises(ValidationError) as excinfo:
        # Passing string to float field raises parsing error or validator error
        BenchmarkParameters(tile_um=["20x"]) 
    # Pydantic will complain about 'Input should be a valid number'
    assert "Input should be a valid number" in str(excinfo.value)

def test_backend_constraint_failure():
    cfg_data = {
        "experiment": {"project_name": "test", "annotation_file": "x"},
        "slide_processing": {"backend": "openslide"},
        "benchmark_parameters": {
            "feature_extraction": ["lazy_specific_model"], # Requires lazyslide
            "mil": ["MockMIL"]
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Config.model_validate(cfg_data)
    assert "requires 'lazyslide' backend" in str(excinfo.value)

def test_backend_constraint_success():
    cfg_data = {
        "experiment": {"project_name": "test", "annotation_file": "x"},
        "slide_processing": {"backend": "lazyslide"},
        "benchmark_parameters": {
            "feature_extraction": ["lazy_specific_model"], 
            "mil": ["MockMIL"]
        }
    }
    # Should not raise
    cfg = Config.model_validate(cfg_data)
    assert cfg.slide_processing.backend == "lazyslide"