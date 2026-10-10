#!/usr/bin/env python3
"""Prepare Game2Real-Bench TSVs under $LMUData for the bundled VLMEvalKit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


TARGETS = {
    "toy_game": "Game2Real_ToyGame.tsv",
    "real_world": "Game2Real_RealWorld.tsv",
}


def convert_partition(dataset_root: Path, lmu_data: Path, partition: str) -> Path:
    source = dataset_root / "data" / f"{partition}.tsv"
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"empty dataset table: {source}")

    for row in rows:
        relative_paths = json.loads(row["image_path"])
        absolute_paths = []
        for value in relative_paths:
            image = (dataset_root / Path(value)).resolve()
            if dataset_root not in image.parents:
                raise ValueError(f"image path escapes the dataset root: {value}")
            if not image.is_file():
                raise FileNotFoundError(image)
            absolute_paths.append(image.as_posix())
        row["image_path"] = json.dumps(absolute_paths, ensure_ascii=False, separators=(",", ":"))
        # The custom loader recognizes this as path metadata and removes it
        # before ImageBaseDataset attempts base64 decoding.
        row["image"] = absolute_paths[0]

    lmu_data.mkdir(parents=True, exist_ok=True)
    output = lmu_data / TARGETS[partition]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--lmu-data",
        type=Path,
        default=Path(os.environ.get("LMUData", Path.home() / "LMUData")),
        help="VLMEvalKit data directory (default: $LMUData or ~/LMUData)",
    )
    parser.add_argument(
        "--partition",
        choices=("all", "toy_game", "real_world"),
        default="all",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    partitions = TARGETS if args.partition == "all" else (args.partition,)
    for partition in partitions:
        output = convert_partition(dataset_root, args.lmu_data.resolve(), partition)
        print(f"{partition}: {output}")


if __name__ == "__main__":
    main()
