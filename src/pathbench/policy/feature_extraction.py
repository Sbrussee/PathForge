from __future__ import annotations
from typing import Any
from tqdm import tqdm
from pathlib import Path
from dataclasses import asdict

from pathbench.policy.base import PolicyBase
from pathbench.config.config import Config
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.slide_processing.base import SlideProcessorBase

class FeatureExtractionPolicy(PolicyBase):
    """
    Policy for extracting features from WSI slides.
    DEPENDENCIES: Depends ONLY on core abstractions (SlideProcessorBase).
    """
    def __init__(self, config: Config):
        self.config = config
    
    def execute(self) -> None:
        # 1. Resolve Backend (Factory Pattern)
        backend_name = self.config.slide_processing.backend
        ProcessorClass = SLIDE_PROCESSORS.get(backend_name)
        
        if not ProcessorClass:
            raise ValueError(f"Slide processing backend '{backend_name}' not found in registry.")
            
        # Instantiate via abstract interface
        processor: SlideProcessorBase = ProcessorClass()
        
        # 2. Prepare Configuration Dictionaries
        # We convert dataclasses to dicts to pass to the abstract methods
        seg_config = {
            "method": "otsu", # Default or from config
            "params": self.config.slide_processing.qc_filters[0] if self.config.slide_processing.qc_filters else {} 
        }
        # Note: In a real scenario, map Config attributes strictly to these dicts
        
        model_name = self.config.benchmark_parameters.feature_extraction[0]
        feat_config = {
            "model": model_name,
            "params": {} # Add batch_size etc from config if available
        }

        # 3. Execution Loop
        for ds in self.config.datasets:
            if ds.used_for == "ignore": continue
            
            slide_dir = Path(ds.slide_path)
            slides = list(slide_dir.glob("*.svs")) + list(slide_dir.glob("*.ndpi")) + list(slide_dir.glob("*.tiff"))
            
            print(f"Processing dataset '{ds.name}' using {processor}: {len(slides)} slides.")
            
            for slide_path in tqdm(slides):
                try:
                    # Pure Abstract Usage
                    wsi = processor.load_slide(str(slide_path))
                    
                    wsi = processor.segment_tissue(wsi, config=seg_config)
                    wsi = processor.extract_patches(wsi, config={"params": {}}) # Add tile size config
                    wsi = processor.extract_features(wsi, config=feat_config)
                    
                    # Persistence is handled by the backend or explicit save
                    # processor.save_slide(wsi, str(ds.tile_path)) 
                    
                except Exception as e:
                    print(f"Error processing {slide_path.name}: {e}")