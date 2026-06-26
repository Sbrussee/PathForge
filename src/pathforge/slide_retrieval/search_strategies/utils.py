from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_patch_dicts_pickle(
    pickle_path: str | Path,
    *,
    reconstruct_features: bool = True,
) -> dict[str, Any]:
    """
    Load one SISH mosaic pickle into the dictionary shape expected by the paper port.

    Inputs:
        pickle_path:
            Path to a pickle file expected to store at least ``{"patches": ...}``.
        reconstruct_features:
            If ``True``, cast patch feature arrays to ``float32`` NumPy arrays.

    Output:
        dict[str, Any]:
            Dictionary containing ``patches`` and ``properties`` keys.
    """
    with Path(pickle_path).open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "Expected mosaic pickle payload to be a dictionary with 'patches'. "
            f"Got {type(payload).__name__}."
        )

    patches = list(payload.get("patches", []))
    properties = dict(payload.get("properties", {}))
    if reconstruct_features:
        for patch in patches:
            if isinstance(patch, dict) and patch.get("feature") is not None:
                patch["feature"] = np.asarray(patch["feature"], dtype=np.float32)

    return {
        "patches": patches,
        "properties": properties,
    }
