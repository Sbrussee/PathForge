from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class SlideMap:
    slide_id: str

class HeatMap(SlideMap): ...
class GraphMap(SlideMap): ...