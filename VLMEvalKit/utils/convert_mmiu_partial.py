#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_options(option_text):
    options = {}
    for line in str(option_text).splitlines():
        line = line.strip()
        if len(line) >= 2 and line[0].isalpha() and line[1] in {":", ".", ")"}:
            options[line[0].upper()] = line[2:].strip()
    return options


def local_image_path(raw_path, root):
    parts = Path(str(raw_path).replace("\\", "/")).parts
    try:
        rel_parts = parts[parts.index("MMIU") + 1:]
    except ValueError:
        rel_parts = parts[-3:]
    rel_path = Path("resource") / "MMIU-partial" / Path(*rel_parts)
    disk_path = root.parent / Path(*rel_path.parts[1:])
    if not disk_path.exists():
        raise FileNotFoundError(rel_path.as_posix())
    return rel_path.as_posix()


def convert(root, out_path):
    rows = []
    aggregate_files = [
        root / "Continuous-temporal" / "Continuous-temporal.json",
        root / "Discrete-temporal" / "Discrete-temporal.json",
    ]
    for json_path in aggregate_files:
        split = json_path.parent.name
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for original_index, item in enumerate(data):
            inputs = item.get("input", {})
            options = parse_options(item.get("options", ""))
            image_paths = [
                local_image_path(path, root)
                for path in inputs.get("input_image_path", [])
            ]
            row = {
                "index": len(rows),
                "question": inputs.get("question", ""),
                "context": inputs.get("context", ""),
                "answer": item.get("output", {}).get("output_text", "").strip().upper(),
                "image_path": json.dumps(image_paths, ensure_ascii=False),
                "category": item.get("task", ""),
                "l2-category": split,
                "source": item.get("source", ""),
                "visual_input_component": item.get("visual_input_component", ""),
                "split": "partial",
                "original_index": original_index,
            }
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                if letter in options:
                    row[letter] = options[letter]
            rows.append(row)

    df = pd.DataFrame(rows)
    option_cols = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in df.columns]
    df = df[
        [
            "index", "question", "context", *option_cols, "answer", "image_path",
            "category", "l2-category", "source", "visual_input_component",
            "split", "original_index",
        ]
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="resource/MMIU-partial")
    parser.add_argument("--out", default="resource/MMIU-partial/MMIU_partial.tsv")
    args = parser.parse_args()
    df = convert(Path(args.root), Path(args.out))
    print(f"Wrote {args.out}: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
