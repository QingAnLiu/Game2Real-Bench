"""Prediction-file loading and evaluation orchestration."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .dataset import load_partition
from .evaluator import evaluate_record
from .scoring import summarize_results


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON prediction file must contain a list")
        return value
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def run_evaluation(
    dataset_root: str | Path,
    partition: str,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    records = load_partition(dataset_root, partition)
    by_id = {record["id"]: record for record in records}
    by_index = {str(record["index"]): record for record in records}
    predictions = _load_predictions(Path(predictions_path))
    results = []
    seen: set[str] = set()
    for item in predictions:
        key = str(item.get("id") or item.get("index") or "")
        record = by_id.get(key) or by_index.get(key)
        if record is None:
            raise ValueError(f"prediction references unknown sample: {key!r}")
        if record["id"] in seen:
            raise ValueError(f"duplicate prediction for {record['id']}")
        if "prediction" not in item:
            raise ValueError(f"prediction field is missing for {record['id']}")
        seen.add(record["id"])
        outcome = evaluate_record(item["prediction"], record)
        results.append(
            {
                "id": record["id"],
                "index": record["index"],
                "partition": partition,
                "category": record.get("category", ""),
                "subfield": record.get("subfield", ""),
                "prediction": item["prediction"],
                "extracted": outcome.extracted,
                "evaluator": outcome.evaluator,
                "correct": outcome.correct,
                "detail": outcome.detail,
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "detailed_results.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    scores = summarize_results(results)
    (output / "scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return scores
