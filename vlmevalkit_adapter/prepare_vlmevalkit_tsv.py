#!/usr/bin/env python3
"""Create a VLMEvalKit-ready TSV with absolute local image paths."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("toy_game", "real_world"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    source = root / "data" / f"{args.partition}.tsv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        columns = list(rows[0])
    for row in rows:
        paths = json.loads(row["image_path"])
        absolute = []
        for path in paths:
            candidate = (root / Path(path)).resolve()
            if root not in candidate.parents:
                raise ValueError(f"path escapes dataset root: {path}")
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            absolute.append(candidate.as_posix())
        row["image_path"] = json.dumps(absolute, ensure_ascii=False, separators=(",", ":"))
        row["image"] = absolute[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
