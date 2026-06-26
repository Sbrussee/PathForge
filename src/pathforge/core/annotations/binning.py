from __future__ import annotations

from typing import Sequence

def bin_times(times: Sequence[float], n_bins: int = 3, method: str = "quantile") -> list[int]:
    """Bin continuous survival times into integer classes for discrete survival tasks."""
    assert n_bins >= 2
    if method == "quantile":
        qs = [i / n_bins for i in range(1, n_bins)]
        sorted_t = sorted(times)
        cuts = [sorted_t[int(q * (len(times)-1))] for q in qs]
    else:
        tmin, tmax = min(times), max(times)
        step = (tmax - tmin) / n_bins
        cuts = [tmin + step*i for i in range(1, n_bins)]
        
    def to_bin(t: float) -> int:
        b = 1
        for c in cuts:
            if t > c:
                b += 1
        return b
        
    return [to_bin(t) for t in times]
