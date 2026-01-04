# src/pathbench/core/io/base.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def ensure_base_path(base: Path) -> Path:
    """Validate and return a base path (must have no suffix)."""
    if not isinstance(base, Path):
        base = Path(base)
    if base.suffix:
        raise ValueError(f"Expected base path without suffix, got: {base}")
    return base

def ensure_suffix(path: str | Path, suffix: str) -> Path:
    """
    Ensure `path` ends with `suffix`.
    - If `path` has no suffix: add `suffix`
    - If `path` already has a suffix: require it matches `suffix`
    """
    p = Path(path)

    if not suffix.startswith("."):
        raise ValueError(f"Suffix must start with '.', got: {suffix}")

    if p.suffix == "":
        return p.with_suffix(suffix)

    if p.suffix != suffix:
        raise ValueError(f"Expected suffix '{suffix}', got '{p.suffix}' for: {p}")

    return p

def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for `path` if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def list_stem_matches(base: Path) -> list[Path]:
    """List files next to `base` that share the same stem."""
    base = ensure_base_path(base)
    return sorted(base.parent.glob(base.name + ".*"))


def resolve_existing_path(
    *,
    base: Path,
    candidates: Iterable[Path],
    allowed_suffixes: Iterable[str],
    kind: str = "artifact",
    prefer_suffixes: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """Resolve a base path to a single existing file with a supported suffix."""
    base = ensure_base_path(base)
    allowed = tuple(allowed_suffixes)

    supported = [p for p in candidates if p.suffix in allowed]
    if not supported:
        return None
    if len(supported) == 1:
        return supported[0]

    if prefer_suffixes is not None:
        for suf in prefer_suffixes:
            for p in supported:
                if p.suffix == suf:
                    return p

    raise ValueError(
        f"Ambiguous {kind} format for base '{base}'. Found: {[str(p) for p in supported]}. "
        f"Allowed suffixes: {list(allowed)}"
    )

def detect_artifact_path(
    base: Path,
    *,
    allowed_suffixes: tuple[str, ...],
    kind: str,
    prefer_suffixes: tuple[str, ...],
) -> Optional[Path]:
    """Resolve a base path to an existing artifact file."""
    base = ensure_base_path(base)
    return resolve_existing_path(
        base=base,
        candidates=list_stem_matches(base),
        allowed_suffixes=allowed_suffixes,
        kind=kind,
        prefer_suffixes=prefer_suffixes,
    )

