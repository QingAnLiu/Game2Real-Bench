"""Score aggregation helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def summarize_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty result set")
    correct = sum(bool(row["correct"]) for row in materialized)
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in materialized:
        group = str(row.get("category") or row.get("subfield") or "unspecified")
        groups[group].append(bool(row["correct"]))
    return {
        "num_samples": len(materialized),
        "num_correct": correct,
        "accuracy": correct / len(materialized),
        "by_category": {
            group: {
                "num_samples": len(values),
                "num_correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for group, values in sorted(groups.items())
        },
    }
