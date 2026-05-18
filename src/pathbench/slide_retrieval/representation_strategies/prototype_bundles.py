from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SalvPrototypeBundle:
    """Resolved Salv prototype discovery run used by PBMS retrieval."""

    run_dir: Path
    prototypes_path: Path
    labels_path: Path
    summary_path: Path
    prototype_matrix: np.ndarray
    labels: tuple[str, ...]
    feature_dim: int
    content_hash: str

    @property
    def n_proto(self) -> int:
        return int(self.prototype_matrix.shape[0])

    @property
    def bundle_id(self) -> str:
        return f"{self.run_dir.name}_{self.content_hash[:12]}"


class SalvPrototypeBundleResolver:
    """Resolve Salv prototype discovery outputs for a PathBench combo."""

    def __init__(self, prototypes_root: str | Path) -> None:
        self.prototypes_root = Path(prototypes_root).expanduser().resolve()

    def resolve(
        self,
        *,
        tile_px: int,
        tile_mpp: float,
        feature_extraction: str,
        color_norm: str | None,
        prototype_run_name: str | None = None,
    ) -> SalvPrototypeBundle:
        if not self.prototypes_root.exists():
            raise FileNotFoundError(
                f"Prototype root does not exist: {self.prototypes_root}"
            )

        feature_roots = self._build_feature_roots(
            tile_px=tile_px,
            tile_mpp=tile_mpp,
            feature_extraction=feature_extraction,
            color_norm=color_norm,
        )
        candidate_run_dirs = _dedupe_paths(
            candidate_run_dir
            for feature_root in feature_roots
            for candidate_run_dir in self._candidate_run_dirs(
                feature_root=feature_root,
                prototype_run_name=prototype_run_name,
            )
        )

        matching_run_dirs: list[Path] = []
        for run_dir in candidate_run_dirs:
            summary_path = run_dir / "prototype_discovery_summary.json"
            try:
                summary = _read_json(summary_path)
            except Exception:
                continue

            if _summary_matches_combo(
                summary=summary,
                tile_px=tile_px,
                tile_mpp=tile_mpp,
                feature_extraction=feature_extraction,
                color_norm=color_norm,
            ):
                matching_run_dirs.append(run_dir)

        if not matching_run_dirs:
            requested = (
                f", prototype_run_name={prototype_run_name!r}"
                if prototype_run_name is not None
                else ""
            )
            raise FileNotFoundError(
                "No Salv prototype run matched the active retrieval combo "
                f"(tile_px={tile_px}, tile_mpp={tile_mpp}, "
                f"feature_extraction={feature_extraction!r}, "
                f"color_norm={color_norm!r}{requested}). Expected under "
                f"{', '.join(str(path) for path in feature_roots)}, "
                "or pass an explicit prototype_run_name relative "
                "to that folder."
            )

        if len(matching_run_dirs) > 1:
            matches = "\n".join(f"- {path}" for path in sorted(matching_run_dirs))
            raise ValueError(
                "Multiple Salv prototype runs matched the active retrieval combo. "
                "Set prototype_run_name to disambiguate.\n"
                f"{matches}"
            )

        return load_salv_prototype_bundle(matching_run_dirs[0])

    def _build_feature_roots(
        self,
        *,
        tile_px: int,
        tile_mpp: float,
        feature_extraction: str,
        color_norm: str | None,
    ) -> list[Path]:
        tiling_id = f"{int(tile_px)}px_{float(tile_mpp):g}mpp"
        feature_name = _build_feature_name(
            feature_extraction=feature_extraction,
            color_norm=color_norm,
        )
        return _dedupe_paths(
            [
                self.prototypes_root / tiling_id / feature_name,
                self.prototypes_root / f"{tiling_id}_{feature_name}",
            ]
        )

    def _candidate_run_dirs(
        self,
        *,
        feature_root: Path,
        prototype_run_name: str | None,
    ) -> list[Path]:
        if _is_prototype_run_dir(self.prototypes_root):
            return [self.prototypes_root]

        normalized_run_name = (
            None if prototype_run_name is None else prototype_run_name.strip()
        )
        if normalized_run_name:
            return self._explicit_candidate_run_dirs(
                feature_root=feature_root,
                prototype_run_name=normalized_run_name,
            )

        return self._default_candidate_run_dirs(feature_root=feature_root)

    def _explicit_candidate_run_dirs(
        self,
        *,
        feature_root: Path,
        prototype_run_name: str,
    ) -> list[Path]:
        run_path = Path(prototype_run_name).expanduser()
        if run_path.is_absolute():
            return [run_path.resolve()]

        candidates = [
            feature_root / run_path,
            self.prototypes_root / run_path,
        ]

        if run_path.parent == Path(".") and feature_root.is_dir():
            candidates.extend(
                path / run_path
                for path in sorted(feature_root.iterdir())
                if path.is_dir()
            )
            method_dir = feature_root / run_path
            if method_dir.is_dir():
                candidates.extend(
                    path for path in sorted(method_dir.iterdir()) if path.is_dir()
                )

        return _dedupe_paths(candidates)

    def _default_candidate_run_dirs(self, *, feature_root: Path) -> list[Path]:
        if not feature_root.exists():
            return []

        candidates = [feature_root]
        if feature_root.is_dir():
            method_dirs = sorted(path for path in feature_root.iterdir() if path.is_dir())
            candidates.extend(method_dirs)
            for method_dir in method_dirs:
                candidates.extend(
                    path for path in sorted(method_dir.iterdir()) if path.is_dir()
                )

        return _dedupe_paths(
            path for path in candidates if _is_prototype_run_dir(path)
        )


def load_salv_prototype_bundle(run_dir: str | Path) -> SalvPrototypeBundle:
    run_path = Path(run_dir).expanduser().resolve()
    prototypes_path = run_path / "prototype_discovery.json"
    labels_path = run_path / "proto_labels.json"
    summary_path = run_path / "prototype_discovery_summary.json"

    if not prototypes_path.exists():
        raise FileNotFoundError(f"Missing prototype_discovery.json: {prototypes_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing manual proto_labels.json: {labels_path}")
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing prototype_discovery_summary.json: {summary_path}"
        )

    prototype_payload = _read_json(prototypes_path)
    prototype_matrix = _load_prototype_matrix(prototype_payload, prototypes_path)
    labels = _load_proto_labels(labels_path, n_proto=int(prototype_matrix.shape[0]))
    feature_dim = int(prototype_matrix.shape[1])
    content_hash = _hash_files([prototypes_path, labels_path, summary_path])

    return SalvPrototypeBundle(
        run_dir=run_path,
        prototypes_path=prototypes_path,
        labels_path=labels_path,
        summary_path=summary_path,
        prototype_matrix=prototype_matrix,
        labels=tuple(labels),
        feature_dim=feature_dim,
        content_hash=content_hash,
    )


def _summary_matches_combo(
    *,
    summary: dict[str, Any],
    tile_px: int,
    tile_mpp: float,
    feature_extraction: str,
    color_norm: str | None,
) -> bool:
    feature_source = (
        summary.get("config", {})
        .get("experiment", {})
        .get("feature_source", {})
    )
    if not isinstance(feature_source, dict):
        return False

    try:
        summary_tile_px = int(feature_source.get("tile_px"))
        summary_mpp = float(feature_source.get("mpp"))
    except (TypeError, ValueError):
        return False

    summary_feature_extraction = str(feature_source.get("feat_extractor", "")).strip()
    summary_color_norm = _normalize_optional_name(feature_source.get("color_norm"))

    return (
        summary_tile_px == int(tile_px)
        and abs(summary_mpp - float(tile_mpp)) <= 1e-9
        and summary_feature_extraction == str(feature_extraction).strip()
        and summary_color_norm == _normalize_optional_name(color_norm)
    )


def _load_prototype_matrix(
    payload: dict[str, Any],
    prototypes_path: Path,
) -> np.ndarray:
    prototype_entries = payload.get("prototypes")
    if not isinstance(prototype_entries, list) or not prototype_entries:
        raise ValueError(
            f"Prototype artifact must contain a non-empty prototypes list: {prototypes_path}"
        )

    normalized: list[tuple[int, np.ndarray]] = []
    for index, entry in enumerate(prototype_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Prototype entry at index {index} must be an object.")
        if "prototype_id" not in entry or "vector" not in entry:
            raise ValueError(
                f"Prototype entry at index {index} must contain prototype_id and vector."
            )

        prototype_id = int(entry["prototype_id"])
        vector = np.asarray(entry["vector"], dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(
                f"Prototype {prototype_id} vector must be 1D, got {vector.shape}."
            )
        normalized.append((prototype_id, vector))

    normalized.sort(key=lambda item: item[0])
    prototype_ids = [prototype_id for prototype_id, _ in normalized]
    expected_ids = list(range(len(normalized)))
    if prototype_ids != expected_ids:
        raise ValueError(
            "Prototype IDs must be contiguous and zero-based. "
            f"Got {prototype_ids}, expected {expected_ids}."
        )

    feature_dims = {int(vector.shape[0]) for _, vector in normalized}
    if len(feature_dims) != 1:
        raise ValueError("All prototype vectors must have the same feature dimension.")

    return np.stack([vector for _, vector in normalized], axis=0).astype(np.float32)


def _load_proto_labels(labels_path: Path, *, n_proto: int) -> list[str]:
    payload = _read_json(labels_path)

    if isinstance(payload.get("prototypes"), list):
        labels_by_id: dict[int, str] = {}
        for index, entry in enumerate(payload["prototypes"]):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"proto_labels.json prototypes[{index}] must be an object."
                )
            if "prototype_id" not in entry or "label" not in entry:
                raise ValueError(
                    "Each proto_labels.json prototype entry must contain "
                    "prototype_id and label."
                )
            labels_by_id[int(entry["prototype_id"])] = _normalize_policy_label(
                entry["label"]
            )
        labels = [labels_by_id.get(prototype_id) for prototype_id in range(n_proto)]
        if any(label is None for label in labels):
            missing = [idx for idx, label in enumerate(labels) if label is None]
            raise ValueError(f"proto_labels.json is missing prototype IDs: {missing}")
        return [str(label) for label in labels]

    if isinstance(payload.get("labels"), list):
        return _validate_label_list(payload["labels"], n_proto=n_proto)

    if isinstance(payload.get("label"), list):
        return _validate_label_list(payload["label"], n_proto=n_proto)

    raise ValueError(
        "proto_labels.json must contain either `prototypes`, `labels`, or `label`."
    )


def _validate_label_list(values: list[Any], *, n_proto: int) -> list[str]:
    if len(values) != n_proto:
        raise ValueError(
            f"Expected {n_proto} prototype labels, got {len(values)}."
        )
    return [_normalize_policy_label(value) for value in values]


def _normalize_policy_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label not in {"include", "exclude"}:
        raise ValueError(
            "Prototype labels must be exactly 'include' or 'exclude'. "
            f"Got {value!r}."
        )
    return label


def _normalize_optional_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_feature_name(
    *,
    feature_extraction: str,
    color_norm: str | None,
) -> str:
    feature_name = str(feature_extraction).strip()
    if not feature_name:
        raise ValueError("feature_extraction must be a non-empty string.")

    normalized_color_norm = _normalize_optional_name(color_norm)
    if normalized_color_norm is None:
        return feature_name

    return f"{feature_name}_{normalized_color_norm}"


def _is_prototype_run_dir(path: Path) -> bool:
    return (path / "prototype_discovery_summary.json").is_file()


def _dedupe_paths(paths) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha1()
    for path in paths:
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
