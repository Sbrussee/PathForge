from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class Report:
    """Base report payload wrapper used by generated PathForge artifacts."""

    payload: dict

class ProcessingReport(Report):
    """Report payload for slide-processing and feature-extraction workflows."""


class DebugReport(Report):
    """Report payload for debugging and inspection workflows."""


class PredictionReport(Report):
    """Report payload for inference or evaluation prediction outputs."""
