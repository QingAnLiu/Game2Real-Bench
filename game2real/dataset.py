"""Portable TSV loading and image-path resolution."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


PARTITION_FILES = {
    "toy_game": Path("data/toy_game.tsv"),
    "real_world": Path("data/real_world.tsv"),
}


def _safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe dataset-relative path: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes dataset root: {relative}")
    return resolved


def load_partition(dataset_root: str | Path, partition: str) -> list[dict[str, str]]:
    """Load one public benchmark partition from its TSV file."""
    root = Path(dataset_root)
    try:
        relative = PARTITION_FILES[partition]
    except KeyError as exc:
        raise ValueError(f"unknown partition {partition!r}; choose from {sorted(PARTITION_FILES)}") from exc
    path = _safe_path(root, relative.as_posix())
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = 1000 if partition == "toy_game" else 400
    if len(rows) != expected:
        raise ValueError(f"{partition} must contain {expected} rows, found {len(rows)}")
    return rows


def image_paths(record: dict[str, Any]) -> list[str]:
    """Return the ordered repository-relative image paths for a record."""
    value = record.get("image_path", "[]")
    paths = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError(f"invalid image_path for {record.get('id', '<unknown>')}")
    return paths


def resolve_image_paths(
    dataset_root: str | Path,
    record: dict[str, Any],
    *,
    require_exists: bool = True,
) -> list[Path]:
    """Resolve ordered image paths while preventing traversal outside the dataset."""
    root = Path(dataset_root)
    resolved = [_safe_path(root, path) for path in image_paths(record)]
    if require_exists:
        missing = [str(path) for path in resolved if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing images for {record.get('id', '<unknown>')}: {missing}")
    return resolved


def iter_partition(dataset_root: str | Path, partition: str) -> Iterable[dict[str, str]]:
    yield from load_partition(dataset_root, partition)
