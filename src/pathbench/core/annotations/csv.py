from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Dict
from pathbench.core.annotations.base import AnnotationsBase
import pandas as pd




class CSVAnnotations(AnnotationsBase):
"""
CSVAnnotations expects annotations to be read from CSV format, building a dictionary
where the keys are column names and values are the row values for that column.
"""

    def __init__(self, path_to_csv: str):
        self.contains_wsi_path = False
        self.contains_patient = False
        self.contains_dataset = False
        
        self.annotations = self.load_annotations(path_to_csv)
        valid = self.validate_annotations(self.annotations)
        if not valid:
            raise ValueError("Invalid annotations provided, see logs above.")
        else:
            logging.info("CSV Annotations successfully loaded and validated.")
        
    def load_annotations(self, id: str) -> Dict:
        """Load annotations from a CSV file."""
        # Using pandas to read CSV, with automatic separator detection, and converting to dict.
        # Keys should be column names, values lists of column values.
        return pd.read_csv(id, sep=None, engine='python').to_dict(orient='list')
    
    def save_annotations(self, id: str, annotations: Dict):
        """Save annotations to a CSV file."""
        # Using pandas to save DataFrame to CSV
        df = pd.DataFrame(annotations)
        df.to_csv(id, index=False)
    
    def validate_annotations(self, annotations: Dict) -> bool:
        """Validate the given annotations."""
        # Annotation should be a dictionary with lists as values
        assert isinstance(annotations, dict), "Annotations should be a dictionary."
        for key, value in annotations.items():
            assert isinstance(value, list), f"Annotation for key '{key}' should be a list."
            
        # At least needs a column 'slide', 'category' to be valid for SlideDataset
        assert "slide" in annotations.keys(), "Column 'slide' is missing from your annotations file."
        assert "category" in annotations.keys(), "Column 'category' is missing from your annotations file."
        
        if "patient" not in annotations.keys():
            logging.warning("Warning: Column 'patient' is missing from your annotations file, will remap to 'slide'.")
            annotations["patient"] = annotations["slide"]
            self.contains_patient = False
        else:
            logging.info("Column 'patient' found in annotations, will use this for patient mapping.")
            self.contains_patient = True    
        
        if "wsi_path" not in annotations.keys():
            logging.warning("Warning: Column 'wsi_path' is missing from your annotations file, will search for slide names in 'slide_path' from your config.")
            self.contains_wsi_path = False
        else:
            logging.info("Column 'wsi_path' found in annotations, will use this for slide paths")
            self.contains_wsi_path = True
        
        if "dataset" not in annotations.keys():
            logging.warning("Warning: Column 'dataset' is missing from your annotations file, all slides will be mapped to the same dataset.")
            annotations["dataset"] = ["default"] * len(next(iter(annotations.values()), []))
            self.contains_dataset = False
        else:
            logging.info("Found the following datasets in annotations: " + ", ".join(set(annotations["dataset"])))
            self.contains_dataset = True
        
        return True # If all checks passed
    
    def inspect_annotations(self, annotations: Dict) -> None:
        """Print a summary of the annotations."""
        n_entries = len(next(iter(annotations.values()), []))
        print(f"Annotations contain {n_entries} entries with columns: {list(annotations.keys())}")
    