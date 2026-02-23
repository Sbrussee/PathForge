import json
from pathlib import Path
from typing import Dict, Any

def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path, *, indent: int = 2, sort_keys: bool = True) -> None:
    """
    Save a dict to JSON.

    Args:
        data: Dictionary to write.
        path: Destination path.
        indent: JSON indentation.
        sort_keys: Whether to sort keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)
        f.write("\n")