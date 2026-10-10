"""Public Game2Real-Bench dataset registrations for VLMEvalKit."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .image_mixedqa import ImageMixedQADataset, split_mixedqa_paths
from .real_world_final import RealWorldFinalDataset
from ..smp import dump, get_intermediate_file_path, load


TOY_DATASET = "Game2Real_ToyGame"
REAL_WORLD_DATASET = "Game2Real_RealWorld"


def _find_dataset_root(data: pd.DataFrame, partition: str) -> Path:
    if "image_path" not in data or data.empty:
        raise ValueError("Game2Real-Bench metadata does not contain image_path")
    paths = split_mixedqa_paths(data.iloc[0]["image_path"])
    if not paths:
        raise ValueError("Game2Real-Bench sample has no image path")
    image = Path(str(paths[0])).resolve()
    for candidate in (image.parent, *image.parents):
        if (candidate / "data" / f"{partition}.tsv").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the Game2Real-Bench dataset root from the prepared image paths. "
        "Run VLMEvalKit/utils/prepare_game2real_data.py again."
    )


def _official_toy_evaluation(dataset: "Game2RealToyDataset", eval_file: str) -> pd.DataFrame:
    try:
        from game2real.runner import run_evaluation
    except ImportError as exc:
        raise ImportError(
            "Install the companion Game2Real-Bench package first: python -m pip install -e ."
        ) from exc

    predictions = load(eval_file).copy()
    if "prediction" not in predictions:
        raise ValueError(f"prediction column is missing from {eval_file}")

    dataset_root = _find_dataset_root(dataset.data, "toy_game")
    with tempfile.TemporaryDirectory(prefix="game2real-toy-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        prediction_file = temp_root / "predictions.jsonl"
        with prediction_file.open("w", encoding="utf-8", newline="\n") as handle:
            for _, row in predictions.iterrows():
                payload: dict[str, Any] = {
                    "index": str(row["index"]),
                    "prediction": row["prediction"],
                }
                if "id" in predictions and not pd.isna(row.get("id")):
                    payload["id"] = str(row["id"])
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        output_dir = temp_root / "results"
        scores = run_evaluation(dataset_root, "toy_game", prediction_file, output_dir)
        details = pd.read_json(output_dir / "detailed_results.jsonl", lines=True)

    hit_by_index = {str(row["index"]): int(bool(row["correct"])) for _, row in details.iterrows()}
    predictions["hit"] = predictions["index"].map(lambda value: hit_by_index[str(value)])
    result_file = get_intermediate_file_path(eval_file, "_game2real_result")
    dump(predictions, result_file)

    rows = [
        {
            "scope": "overall",
            "group": "all",
            "num_samples": int(scores["num_samples"]),
            "accuracy": float(scores["accuracy"]),
        }
    ]
    for category, values in scores["by_category"].items():
        rows.append(
            {
                "scope": "capability",
                "group": category,
                "num_samples": int(values["num_samples"]),
                "accuracy": float(values["accuracy"]),
            }
        )
    score_table = pd.DataFrame(rows)
    score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
    dump(score_table, score_file)
    return score_table


class Game2RealToyDataset(ImageMixedQADataset):
    """Toy Game split with the released deterministic Game2Real scorer."""

    DATASET_URL = {TOY_DATASET: None}
    DATASET_MD5 = {TOY_DATASET: None}

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        return _official_toy_evaluation(self, eval_file)


class Game2RealRealWorldDataset(RealWorldFinalDataset):
    """Real World split with the released deterministic evaluator suite."""

    DATASET_URL = {REAL_WORLD_DATASET: None}
    DATASET_MD5 = {REAL_WORLD_DATASET: None}
