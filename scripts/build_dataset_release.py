#!/usr/bin/env python3
"""Build the public Game2Real-Bench dataset staging directory.

The builder is intentionally local-release-only: it reads the canonical benchmark
artifacts in this workspace and writes a sanitized, relocatable dataset tree. It
never edits the source TSVs or images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TOY_PUBLIC_COLUMNS = [
    "id",
    "index",
    "partition",
    "question",
    "hint",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "answer",
    "accepted_answers",
    "evaluator_payload",
    "question_type",
    "difficulty",
    "category",
    "subfield",
    "image",
    "image_path",
    "question_image_roles",
    "question_image_note",
]

REAL_PUBLIC_COLUMNS = [
    "id",
    "index",
    "partition",
    "question",
    "hint",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "answer",
    "accepted_answers",
    "evaluator_payload",
    "question_type",
    "original_question_type",
    "evaluator_type",
    "answer_normalization_version",
    "difficulty",
    "category",
    "subfield",
    "evaluation_role",
    "primary_toy_task",
    "mapped_toy_tasks",
    "input_format",
    "num_images",
    "primitive",
    "mechanism_alignment",
    "mechanism_alignment_level",
    "alignment_confidence",
    "derivation_type",
    "source",
    "source_subset",
    "source_split",
    "source_index",
    "unique_scene_id",
    "domain_realism",
    "visual_origin",
    "gt_verification_method",
    "contamination_risk",
    "contamination_reason",
    "source_license",
    "source_revision",
    "redistribution_status",
    "source_url_or_archive",
    "image_sha256",
    "image_phash",
    "image",
    "image_path",
    "question_image_roles",
    "question_image_note",
]

CAPABILITY_SLUGS = {
    "Fine-grained Perception": "fine_grained_perception",
    "Spatial Reasoning": "spatial_reasoning",
    "Constraint Reasoning": "constraint_reasoning",
    "Sequential Planning": "sequential_planning",
}


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def json_list(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"expected JSON string list, received: {value[:120]!r}")
    return parsed


def relative_source_path(raw_path: str, source_root: Path, marker: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    marker_with_slash = f"{marker}/"
    if marker_with_slash in normalized:
        relative = normalized.split(marker_with_slash, 1)[1]
    elif normalized.startswith("assets/"):
        relative = normalized
    else:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(source_root.resolve())
            except ValueError as exc:
                raise ValueError(f"path is outside source root: {raw_path}") from exc
        relative = normalized
    result = Path(*relative.split("/"))
    if result.is_absolute() or ".." in result.parts:
        raise ValueError(f"unsafe relative path: {raw_path}")
    return result


def public_image_path(
    partition: str,
    row_id: str,
    image_position: int,
    source_relative: Path,
    *,
    category: str = "",
) -> str:
    suffix = source_relative.suffix.lower()
    if not suffix:
        raise ValueError(f"referenced image has no file extension: {source_relative}")
    directory = Path("images") / partition
    if partition == "toy_game":
        try:
            directory /= CAPABILITY_SLUGS[category]
        except KeyError as exc:
            raise ValueError(f"unknown Toy Game capability: {category!r}") from exc
    return (directory / f"{row_id}-{image_position:02d}{suffix}").as_posix()


def copy_referenced_images(
    rows: list[dict[str, str]],
    *,
    source_root: Path,
    source_marker: str,
    partition: str,
    output_root: Path,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    references: list[str] = []
    source_references: list[str] = []
    transformed: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        if partition == "toy_game":
            index = int(row["index"])
            row_id = f"toy-{index:04d}"
        elif partition == "real_world":
            index = int(row["benchmark_index"])
            row_id = f"real-{index:04d}"
        else:
            raise ValueError(f"unsupported partition: {partition}")
        item["id"] = row_id
        item["index"] = str(index)
        item["partition"] = partition
        paths = json_list(row["image_path"])
        public_paths: list[str] = []
        for image_position, raw_path in enumerate(paths, start=1):
            relative = relative_source_path(raw_path, source_root, source_marker)
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"referenced image is missing: {source}")
            public = public_image_path(
                partition,
                row_id,
                image_position,
                relative,
                category=row.get("category", ""),
            )
            destination = output_root / Path(*public.split("/"))
            if destination.exists():
                raise FileExistsError(f"duplicate generated image path: {public}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            public_paths.append(public)
            references.append(public)
            source_references.append(relative.as_posix())
        item["image_path"] = json.dumps(public_paths, ensure_ascii=False, separators=(",", ":"))
        item["image"] = public_paths[0]
        transformed.append(item)
    return transformed, references, source_references


def build_toy(
    output_root: Path,
    *,
    source_root: Path,
    source_tsv: Path,
    open_evaluator: Path,
    source_marker: str,
) -> dict[str, int]:
    source_rows = load_tsv(source_tsv)
    if len(source_rows) != 1000:
        raise ValueError(f"Toy Game source must contain 1000 rows, found {len(source_rows)}")
    transformed, references, source_references = copy_referenced_images(
        source_rows,
        source_root=source_root,
        source_marker=source_marker,
        partition="toy_game",
        output_root=output_root,
    )
    write_tsv(output_root / "data" / "toy_game.tsv", transformed, TOY_PUBLIC_COLUMNS)

    images_by_index = {
        int(row["index"]): json_list(row["image_path"])
        for row in transformed
    }

    open_records: list[dict[str, Any]] = []
    with open_evaluator.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            index = int(raw.get("final1000_index") or raw["index"])
            images = images_by_index[index]
            open_records.append(
                {
                    "id": f"toy-{index:04d}",
                    "index": index,
                    "task": raw.get("task"),
                    "capability": raw.get("capability"),
                    "answer": raw.get("answer"),
                    "accepted_answers": raw.get("accepted_answers"),
                    "answer_schema": raw.get("answer_schema"),
                    "evaluator_payload": raw.get("evaluator_payload"),
                    "task_contract": raw.get("task_contract"),
                    "images": images,
                }
            )
    if len(open_records) != 600:
        raise ValueError(f"Toy Game open evaluator must contain 600 rows, found {len(open_records)}")
    evaluator_path = output_root / "evaluators" / "toy_game_open.jsonl"
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    with evaluator_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in open_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "rows": len(transformed),
        "references": len(references),
        "unique_images": len(set(references)),
        "unique_source_images": len(set(source_references)),
        "open_evaluator_rows": len(open_records),
    }


def build_real(
    output_root: Path,
    *,
    source_root: Path,
    source_tsv: Path,
    source_marker: str,
) -> tuple[dict[str, int], list[dict[str, str]]]:
    source_rows = load_tsv(source_tsv)
    if len(source_rows) != 400:
        raise ValueError(f"Real World source must contain 400 rows, found {len(source_rows)}")
    transformed, references, source_references = copy_referenced_images(
        source_rows,
        source_root=source_root,
        source_marker=source_marker,
        partition="real_world",
        output_root=output_root,
    )
    for row in transformed:
        if row["source"] == "Visual-Jigsaw":
            row["source_license"] = "Upstream source terms apply"
            row["redistribution_status"] = "source_terms_apply"
    write_tsv(output_root / "data" / "real_world.tsv", transformed, REAL_PUBLIC_COLUMNS)
    return (
        {
            "rows": len(transformed),
            "references": len(references),
            "unique_images": len(set(references)),
            "unique_source_images": len(set(source_references)),
            "duplicate_source_references": len(source_references) - len(set(source_references)),
        },
        transformed,
    )


def build_source_manifest(output_root: Path, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["source"] for row in rows)
    first_by_source: dict[str, dict[str, str]] = {}
    for row in rows:
        first_by_source.setdefault(row["source"], row)
    manifest_rows = []
    for source in sorted(counts):
        row = first_by_source[source]
        stated_license = row["source_license"]
        manifest_rows.append(
            {
                "source": source,
                "num_samples": counts[source],
                "upstream_url": row["source_url_or_archive"],
                "upstream_revision": row["source_revision"],
                "stated_license_or_terms": stated_license,
                "redistribution_status": row["redistribution_status"],
                "release_note": "Retain upstream attribution and all source-specific restrictions.",
            }
        )
    columns = [
        "source",
        "num_samples",
        "upstream_url",
        "upstream_revision",
        "stated_license_or_terms",
        "redistribution_status",
        "release_note",
    ]
    write_tsv(output_root / "manifests" / "sources.tsv", manifest_rows, columns)


DATASET_CARD = """---
license: other
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - vision-language
  - benchmark
  - multimodal-reasoning
  - game-reasoning
configs:
  - config_name: default
    data_files:
      - split: toy_game
        path: data/toy_game.tsv
      - split: real_world
        path: data/real_world.tsv
---

# Game2Real-Bench

**Game2Real-Bench** is a multimodal benchmark for evaluating whether capabilities demonstrated by vision-language models (VLMs) in controlled visual games extend to aligned real-world tasks.

The benchmark brings together two complementary partitions, **Toy Game** and **Real World**, organized around four shared capability groups: **Fine-grained Perception**, **Spatial Reasoning**, **Constraint Reasoning**, and **Sequential Planning**. This design supports capability-level comparisons across game and real-world settings, helping assess how performance in structured visual games relates to performance on corresponding real-world tasks.

## Dataset Overview

Game2Real-Bench contains **1,400 samples** and **2,365 image files** across its two evaluation partitions.

| Partition | Hugging Face Split | Samples | Image Files |
|---|---|---:|---:|
| Toy Game | `toy_game` | 1,000 | 1,586 |
| Real World | `real_world` | 400 | 779 |
| **Total** | — | **1,400** | **2,365** |

### Toy Game

The Toy Game partition comprises **18 game tasks**, with **400 single-answer multiple-choice questions** and **600 open-answer questions**. It provides a controlled setting for assessing visual perception, reasoning, and planning.

### Real World

The Real World partition contains **400 questions grounded in their source materials** and aligned with the same four capability groups. It complements the game-based evaluation by assessing the corresponding capabilities in real-world contexts. Source records are provided in `manifests/sources.tsv`.

## Repository Structure

```text
.
├── data/
│   ├── toy_game.tsv
│   └── real_world.tsv
├── images/
│   ├── toy_game/
│   │   ├── fine_grained_perception/
│   │   ├── spatial_reasoning/
│   │   ├── constraint_reasoning/
│   │   └── sequential_planning/
│   └── real_world/
├── evaluators/
│   └── toy_game_open.jsonl
├── manifests/
│   └── sources.tsv
├── LICENSE_DATA.md
└── ATTRIBUTION.md
```

### Data Format and Image References

Question records are stored in TSV files, with image assets maintained separately under `images/`.

Each image filename begins with the corresponding sample `id` and includes a final two-digit suffix indicating its order within the sample. The `image_path` column stores image paths as a JSON-encoded list. All paths use POSIX notation and are relative to the repository root.

For samples with multiple images, preserve the order specified in `image_path` when constructing model inputs.

## Quick Start

Load the question tables using Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("liuqa/Game2Real-Bench")

toy_game = dataset["toy_game"]
real_world = dataset["real_world"]
```

The `image_path` values reference image files stored separately in the repository. Download the corresponding files under `images/` before resolving these paths locally, using the repository root as the base directory.

## Evaluation

For benchmark evaluation, use the companion code repository, which provides `scripts/download_data.py` for downloading the data and `scripts/run_evaluation.py` for running evaluations. Refer to that repository for setup instructions and script usage.

Evaluation resources for the Toy Game open-answer questions are included in `evaluators/toy_game_open.jsonl`.

## Licensing and Attribution

**Game2Real-Bench is distributed as a mixed-license dataset.** The repository-level `license: other` metadata reflects the presence of different upstream licensing terms. It does not replace those terms or relicense the original source material.

Before using or redistributing the dataset, consult `LICENSE_DATA.md`, `ATTRIBUTION.md`, and `manifests/sources.tsv` for the applicable licensing terms, attribution requirements, and source records.

The Real World partition includes material subject to non-commercial or other source-specific restrictions. Permissions should therefore be reviewed at the source level; the availability of this repository should not be interpreted as permission for unrestricted use or redistribution.

## Citation

If you use Game2Real-Bench in your research, please cite the accompanying paper. Full bibliographic details and a BibTeX entry will be added once the manuscript is publicly available.
"""


LICENSE_DATA = """# Dataset licensing

Game2Real-Bench is a mixed-license collection. No single permissive license covers every file.

The project-authored metadata, benchmark organization, and documentation are provided for research use. Third-party images and derived records remain subject to their upstream licenses and terms. Users must review `ATTRIBUTION.md` and `manifests/sources.tsv` and comply with the strictest terms applicable to the files they use.

The Real World partition includes non-commercial and source-specific material. Consequently, redistribution of the complete partition must not be represented as unrestricted Apache-2.0 or MIT data.

The companion software repository is separately licensed under MIT; that software license does not apply to dataset images.
"""


ATTRIBUTION = """# Third-party attribution and terms

The Real World partition selects examples from public research datasets. Public availability does not erase attribution, non-commercial, or source-image obligations.

| Source | Samples | Upstream terms recorded for this release |
|---|---:|---|
| All-Angles-Bench | 20 | MIT dataset page; source EgoHumans terms also apply |
| CAPTURe-real | 30 | MIT dataset page; underlying FSC-147 image terms also apply |
| GQA | 12 | Visual Genome source terms and attribution apply |
| IKEA Manuals at Work | 5 | CC BY 4.0 |
| OmniSpatial | 61 | CC BY-NC 4.0; non-commercial research only; individual image constraints may apply |
| ReasonMap | 14 | Apache-2.0 dataset; retain individual map attribution |
| SPair-71k | 27 | Respect PASCAL VOC and Flickr source terms |
| SpatiaLQA | 50 | MIT dataset page; review underlying images |
| SSI-Bench | 50 | CC BY 4.0 dataset page |
| Visual-Jigsaw | 40 | Retain upstream attribution and source-specific terms |
| VisualTrans | 91 | MIT dataset page |

Exact upstream URLs, revisions, local source statements, and counts are recorded in `manifests/sources.tsv`. Cite both Game2Real-Bench and the corresponding upstream datasets in publications.
"""


UPLOAD_SCRIPT = r'''param(
    [string]$RepoId = "liuqa/Game2Real-Bench",
    [string]$PythonCommand = "python",
    [string]$HfCommand = "hf",
    [switch]$Publish
)

$ErrorActionPreference = "Stop"
$DatasetRoot = (Resolve-Path $PSScriptRoot).Path
$Validator = Join-Path $DatasetRoot "..\code\scripts\validate_release.py"
$Arguments = @($Validator, "--dataset-root", $DatasetRoot)
if ($Publish) { $Arguments += "--for-publish" }

& $PythonCommand @Arguments
if (-not $Publish) {
    Write-Host "Validation passed. Re-run with -Publish to create/upload $RepoId."
    exit 0
}

Get-Command $HfCommand -ErrorAction Stop | Out-Null
& $HfCommand auth whoami
& $HfCommand repos create $RepoId --type dataset --private --exist-ok
& $HfCommand upload-large-folder $RepoId $DatasetRoot --type dataset --private
Write-Host "Published: https://huggingface.co/datasets/$RepoId"
'''


def write_release_metadata(output_root: Path, toy_stats: dict[str, int], real_stats: dict[str, int]) -> None:
    (output_root / "README.md").write_text(DATASET_CARD, encoding="utf-8", newline="\n")
    (output_root / "LICENSE_DATA.md").write_text(LICENSE_DATA, encoding="utf-8", newline="\n")
    (output_root / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8", newline="\n")
    (output_root / "upload_dataset.ps1").write_text(UPLOAD_SCRIPT, encoding="utf-8", newline="\n")
    stats = {"toy_game": toy_stats, "real_world": real_stats}
    (output_root / "manifests" / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_root: Path) -> None:
    manifest = output_root / "manifests" / "files.sha256"
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path != manifest
        and ".cache" not in path.relative_to(output_root).parts
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-root", type=Path, required=True)
    parser.add_argument("--toy-tsv", type=Path, required=True)
    parser.add_argument("--toy-open-evaluator", type=Path, required=True)
    parser.add_argument("--toy-marker", required=True, help="directory marker used to trim absolute source paths")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-tsv", type=Path, required=True)
    parser.add_argument("--real-marker", required=True, help="directory marker used to trim absolute source paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing dataset staging directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists; use --force to rebuild: {output}")
        required_generated = (
            output / "data" / "toy_game.tsv",
            output / "data" / "real_world.tsv",
            output / "manifests" / "files.sha256",
        )
        if not all(path.is_file() for path in required_generated):
            raise ValueError(f"refusing to replace a directory that is not a prior generated release: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    toy_stats = build_toy(
        output,
        source_root=args.toy_root.resolve(),
        source_tsv=args.toy_tsv.resolve(),
        open_evaluator=args.toy_open_evaluator.resolve(),
        source_marker=args.toy_marker,
    )
    real_stats, real_rows = build_real(
        output,
        source_root=args.real_root.resolve(),
        source_tsv=args.real_tsv.resolve(),
        source_marker=args.real_marker,
    )
    build_source_manifest(output, real_rows)
    write_release_metadata(output, toy_stats, real_stats)
    write_checksums(output)
    print(json.dumps({"output": str(output), "toy_game": toy_stats, "real_world": real_stats}, indent=2))


if __name__ == "__main__":
    main()
