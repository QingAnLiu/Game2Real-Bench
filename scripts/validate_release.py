#!/usr/bin/env python3
"""Validate the local Game2Real-Bench code and dataset release trees."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED = {
    "toy_game": {"rows": 1000, "references": 1586, "unique_images": 1586},
    "real_world": {"rows": 400, "references": 779, "unique_images": 779},
}

CAPABILITY_SLUGS = {
    "Fine-grained Perception": "fine_grained_perception",
    "Spatial Reasoning": "spatial_reasoning",
    "Constraint Reasoning": "constraint_reasoning",
    "Sequential Planning": "sequential_planning",
}

TEXT_SUFFIXES = {".md", ".py", ".ps1", ".toml", ".tsv", ".json", ".jsonl", ".txt", ".sha256"}
FORBIDDEN_LITERALS = (
    "GameBench" + "_V15_" + "Final1000",
    "RealWorld" + "_Final400_" + "v2",
    "cheng" + "yujiet",
    "/root/" + "LMUData",
    "/mnt/" + "afs/",
    "F:/" + "Benchmark/",
    "F:\\" + "Benchmark\\",
)
SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
)


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe path: {value}")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"path escapes dataset root: {value}")
    return candidate


def validate_partition(root: Path, partition: str) -> tuple[list[dict[str, str]], list[str]]:
    rows = load_tsv(root / "data" / f"{partition}.tsv")
    expected = EXPECTED[partition]
    if len(rows) != expected["rows"]:
        raise AssertionError(f"{partition}: expected {expected['rows']} rows, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{partition}: duplicate public ids")
    references: list[str] = []
    for row in rows:
        paths = json.loads(row["image_path"])
        if not isinstance(paths, list) or not paths:
            raise AssertionError(f"{row['id']}: image_path is not a non-empty list")
        if partition == "real_world" and len(paths) != int(row["num_images"]):
            raise AssertionError(f"{row['id']}: num_images does not match image_path")
        if row["image"] != paths[0]:
            raise AssertionError(f"{row['id']}: image must equal the first image_path entry")
        for image_position, value in enumerate(paths, start=1):
            relative = Path(value)
            expected_parent = Path("images") / partition
            if partition == "toy_game":
                try:
                    expected_parent /= CAPABILITY_SLUGS[row["category"]]
                except KeyError as exc:
                    raise AssertionError(f"{row['id']}: unknown capability {row['category']!r}") from exc
            if relative.parent != expected_parent:
                raise AssertionError(
                    f"{row['id']}: expected image directory {expected_parent.as_posix()}, found {relative.parent.as_posix()}"
                )
            if relative.stem != f"{row['id']}-{image_position:02d}":
                raise AssertionError(
                    f"{row['id']}: image {image_position} is not named from its TSV id: {value}"
                )
            path = safe_relative(root, value)
            if not path.is_file():
                raise FileNotFoundError(f"{row['id']}: missing image {value}")
            references.append(value)
        for column in ("accepted_answers", "evaluator_payload"):
            if row.get(column):
                json.loads(row[column])
    if len(references) != expected["references"]:
        raise AssertionError(f"{partition}: expected {expected['references']} references, found {len(references)}")
    if len(set(references)) != expected["unique_images"]:
        raise AssertionError(
            f"{partition}: expected {expected['unique_images']} unique images, found {len(set(references))}"
        )
    image_root = root / "images" / partition
    physical_images = {
        path.relative_to(root).as_posix()
        for path in image_root.rglob("*")
        if path.is_file()
    }
    if physical_images != set(references):
        unreferenced = sorted(physical_images - set(references))
        missing = sorted(set(references) - physical_images)
        raise AssertionError(
            f"{partition}: image tree mismatch; unreferenced={unreferenced[:5]} missing={missing[:5]}"
        )
    if partition == "toy_game":
        actual_capability_dirs = {path.name for path in image_root.iterdir() if path.is_dir()}
        if actual_capability_dirs != set(CAPABILITY_SLUGS.values()):
            raise AssertionError(
                f"toy_game: unexpected capability directories: {sorted(actual_capability_dirs)}"
            )
    image_counts = [len(json.loads(row["image_path"])) for row in rows]
    question_types = {row.get("question_type", "") for row in rows}
    if 1 not in image_counts or not any(count > 1 for count in image_counts):
        raise AssertionError(f"{partition}: smoke coverage requires both single-image and multi-image records")
    if "open" not in question_types or len(question_types - {"open"}) == 0:
        raise AssertionError(f"{partition}: smoke coverage requires both open and choice records")
    return rows, references


def validate_sources(root: Path, real_rows: list[dict[str, str]]) -> None:
    source_rows = load_tsv(root / "manifests" / "sources.tsv")
    manifest_counts = {row["source"]: int(row["num_samples"]) for row in source_rows}
    actual_counts = Counter(row["source"] for row in real_rows)
    if manifest_counts != dict(actual_counts):
        raise AssertionError(f"source manifest mismatch: {manifest_counts} != {dict(actual_counts)}")
    if sum(manifest_counts.values()) != 400:
        raise AssertionError("source manifest does not sum to 400")


def validate_open_evaluator(root: Path, toy_rows: list[dict[str, str]]) -> None:
    path = root / "evaluators" / "toy_game_open.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != 600:
        raise AssertionError(f"expected 600 Toy Game open evaluator records, found {len(records)}")
    if len({record["id"] for record in records}) != 600:
        raise AssertionError("duplicate Toy Game open evaluator ids")
    expected_images = {
        row["id"]: json.loads(row["image_path"])
        for row in toy_rows
        if row["question_type"] == "open"
    }
    if {record["id"] for record in records} != set(expected_images):
        raise AssertionError("Toy Game open evaluator ids do not match open TSV rows")
    for record in records:
        if record["images"] != expected_images[record["id"]]:
            raise AssertionError(f"{record['id']}: evaluator image paths do not match TSV")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksums(root: Path) -> None:
    manifest = root / "manifests" / "files.sha256"
    expected_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest
        and ".cache" not in path.relative_to(root).parts
    }
    recorded: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest
    if set(recorded) != expected_files:
        missing = sorted(expected_files - set(recorded))
        extra = sorted(set(recorded) - expected_files)
        raise AssertionError(f"checksum manifest membership mismatch; missing={missing[:5]} extra={extra[:5]}")
    for relative, expected in recorded.items():
        actual = sha256_file(root / Path(relative))
        if actual != expected:
            raise AssertionError(f"checksum mismatch: {relative}")


def scan_text_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or ".cache" in path.relative_to(root).parts
            or path.suffix.casefold() not in TEXT_SUFFIXES
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        for literal in FORBIDDEN_LITERALS:
            if literal in text:
                raise AssertionError(f"forbidden internal identifier {literal!r} in {path}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise AssertionError(f"possible credential in {path}")


def validate_code_tree(root: Path) -> None:
    required = (
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "game2real/dataset.py",
        "game2real/evaluator.py",
        "scripts/download_data.py",
        "scripts/build_dataset_release.py",
        "scripts/run_evaluation.py",
        "scripts/validate_release.py",
        "vlmevalkit_adapter/prepare_vlmevalkit_tsv.py",
        "tests/test_evaluator.py",
    )
    missing = [path for path in required if not (root / Path(path)).is_file()]
    if missing:
        raise FileNotFoundError(f"code release is missing required files: {missing}")
    scan_text_tree(root)


def validate_gold_replay(code_root: Path, rows: list[dict[str, str]]) -> None:
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from game2real.evaluator import evaluate_record

    failures = []
    for row in rows:
        outcome = evaluate_record(row["answer"], row)
        if not outcome.correct:
            failures.append((row["id"], outcome.evaluator, row["answer"]))
    if failures:
        raise AssertionError(f"gold-answer replay failed for {len(failures)} rows: {failures[:5]}")


def validate_release(
    dataset_root: Path,
    code_root: Path | None,
    *,
    for_publish: bool,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    toy_rows, toy_refs = validate_partition(dataset_root, "toy_game")
    real_rows, real_refs = validate_partition(dataset_root, "real_world")
    validate_sources(dataset_root, real_rows)
    validate_open_evaluator(dataset_root, toy_rows)
    scan_text_tree(dataset_root)
    validate_checksums(dataset_root)
    if code_root is not None:
        resolved_code_root = code_root.resolve()
        validate_code_tree(resolved_code_root)
        validate_gold_replay(resolved_code_root, toy_rows + real_rows)
    return {
        "toy_game": {"rows": len(toy_rows), "references": len(toy_refs), "unique_images": len(set(toy_refs))},
        "real_world": {"rows": len(real_rows), "references": len(real_refs), "unique_images": len(set(real_refs))},
        "publish_validation": for_publish,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--for-publish", action="store_true")
    args = parser.parse_args()
    report = validate_release(
        args.dataset_root,
        args.code_root,
        for_publish=args.for_publish,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
