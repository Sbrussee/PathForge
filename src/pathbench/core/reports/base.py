from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class Report:
    payload: dict

class ProcessingReport(Report): ...
class DebugReport(Report): ...
class PredictionReport(Report): ...