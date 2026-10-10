"""Official Game2Real-Bench dataset registrations for VLMEvalKit."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .image_mixedqa import ImageMixedQADataset
from .real_world_final import RealWorldFinalDataset
from .utils.game2real import real as real_evaluator
from .utils.game2real import toy as toy_evaluator
from ..smp import dump, get_intermediate_file_path, load


TOY_DATASET = "Game2Real_ToyGame"
REAL_WORLD_DATASET = "Game2Real_RealWorld"
CATEGORY_ORDER = [
    "Fine-grained Perception",
    "Spatial Reasoning",
    "Constraint Reasoning",
    "Sequential Planning",
]


def _key(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _prepare_predictions(dataset: ImageMixedQADataset, eval_file: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    predictions = load(eval_file).copy()
    if "prediction" not in predictions:
        raise ValueError(f"prediction column is missing from {eval_file}")
    if "index" not in predictions and "id" not in predictions:
        raise ValueError(f"{eval_file} must contain index or id")

    metadata = dataset.data.copy()
    by_index = {_key(row["index"]): row.to_dict() for _, row in metadata.iterrows()}
    by_id = {
        _key(row["id"]): row.to_dict()
        for _, row in metadata.iterrows()
        if "id" in metadata and _key(row.get("id"))
    }
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for _, prediction_row in predictions.iterrows():
        output_row = prediction_row.to_dict()
        record = None
        if _key(output_row.get("id")):
            record = by_id.get(_key(output_row.get("id")))
        if record is None and _key(output_row.get("index")):
            record = by_index.get(_key(output_row.get("index")))
        if record is None:
            unknown.append(_key(output_row.get("id")) or _key(output_row.get("index")))
            continue
        record_id = _key(record.get("id")) or _key(record["index"])
        if record_id in seen:
            raise ValueError(f"duplicate prediction for {record_id}")
        seen.add(record_id)
        paired.append((output_row, record))
    if unknown:
        raise ValueError(f"prediction file contains unknown sample identifiers: {unknown[:10]}")
    return paired


def _attach_metadata(output: dict[str, Any], record: dict[str, Any]) -> None:
    for column in (
        "id",
        "index",
        "answer",
        "question_type",
        "evaluator_type",
        "category",
        "subfield",
        "evaluation_role",
        "primary_toy_task",
        "mechanism_alignment_level",
        "input_format",
        "contamination_risk",
        "source",
    ):
        if column in record:
            output[column] = record[column]


def _score_table(data: pd.DataFrame, expected_total: int) -> pd.DataFrame:
    if data.empty:
        raise ValueError("cannot score an empty prediction file")
    official = len(data) == expected_total and data["index"].map(_key).nunique() == expected_total
    rows: list[dict[str, Any]] = []

    def add(scope: str, group: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        rows.append(
            {
                "scope": scope,
                "group": group,
                "num_samples": int(len(subset)),
                "score": float(subset["hit"].mean()),
                "official_partition": bool(official),
            }
        )

    add("overall", "all", data)
    for category in CATEGORY_ORDER:
        add("capability", category, data[data["category"] == category])
    if "evaluator_type" in data:
        for evaluator_type in sorted(data["evaluator_type"].dropna().astype(str).unique()):
            add("evaluator", evaluator_type, data[data["evaluator_type"].astype(str) == evaluator_type])
    if "source" in data:
        for source in sorted(data["source"].dropna().astype(str).unique()):
            add("source", source, data[data["source"].astype(str) == source])
    return pd.DataFrame(rows)


def _save_evaluation(eval_file: str, rows: list[dict[str, Any]], expected_total: int) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    result_file = get_intermediate_file_path(eval_file, "_deterministic_result")
    dump(result, result_file)
    scores = _score_table(result, expected_total)
    score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
    dump(scores, score_file)
    return scores


def _evaluate_toy(dataset: "Game2RealToyDataset", eval_file: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prediction_row, record in _prepare_predictions(dataset, eval_file):
        scoring_row = pd.Series({**record, "prediction": prediction_row["prediction"]})
        outcome = toy_evaluator.evaluate_row(scoring_row)
        output = dict(prediction_row)
        _attach_metadata(output, record)
        output.update(
            {
                "hit": int(outcome["hit"]),
                "extracted_answer": outcome["prediction"],
                "extraction_candidate": outcome["candidate"],
                "extraction_source": outcome["source"],
                "extraction_status": outcome["status"],
                "evaluator_mode": outcome["mode"],
                "log": json.dumps(outcome, ensure_ascii=False),
            }
        )
        rows.append(output)
    return _save_evaluation(eval_file, rows, expected_total=1000)


def _evaluate_real_world(dataset: "Game2RealRealWorldDataset", eval_file: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    override_rules = real_evaluator.load_content_overrides()
    for prediction_row, record in _prepare_predictions(dataset, eval_file):
        raw_prediction = prediction_row["prediction"]
        output = dict(prediction_row)
        _attach_metadata(output, record)
        if str(record.get("source")) == "SpatiaLQA":
            gt_steps, gt_warnings = real_evaluator.parse_spatialqa_steps(record["answer"])
            if gt_warnings or not gt_steps:
                raise ValueError(
                    f"invalid released SpatiaLQA ground truth for {record.get('id')}: {gt_warnings}"
                )
            pred_steps, pred_warnings = real_evaluator.parse_spatialqa_steps(raw_prediction)
            matrix, uncertain, applied = real_evaluator.build_public_match_matrix(
                str(record.get("id") or record["index"]),
                gt_steps,
                pred_steps,
                rules=override_rules,
            )
            metrics = real_evaluator.spatialqa_metrics(gt_steps, pred_steps, matrix)
            output.update(
                {
                    "hit": float(metrics["score"]),
                    "extracted_answer": json.dumps(pred_steps, ensure_ascii=False),
                    "extraction_status": "parsed" if pred_steps else "unparseable",
                    "score_type": "spatialqa_semantic_graph",
                    "content_f1": float(metrics["content_f1"]),
                    "precondition_f1": float(metrics["precondition_f1"]),
                    "semantic_matrix": json.dumps(matrix.tolist()),
                    "log": json.dumps(
                        {
                            "parse_warnings": pred_warnings,
                            "matches": metrics["matches"],
                            "reviewed_overrides": applied,
                            "uncertain_pairs": uncertain,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        else:
            safe_record = {
                key: record.get(key)
                for key in ("evaluator_type", "question", "A", "B", "C", "D", "E", "F")
            }
            extracted, reason = real_evaluator.blind_extract(raw_prediction, safe_record)
            error = ""
            try:
                score = float(real_evaluator.score_non_spatialqa(extracted, record))
            except Exception as exc:
                score = 0.0
                error = f"{type(exc).__name__}: {exc}"
            output.update(
                {
                    "hit": score,
                    "extracted_answer": extracted,
                    "extraction_status": "error" if error else "parsed",
                    "extraction_source": reason,
                    "score_type": "binary_deterministic",
                    "log": error,
                }
            )
        rows.append(output)
    return _save_evaluation(eval_file, rows, expected_total=400)


class Game2RealToyDataset(ImageMixedQADataset):
    """Toy Game split with the official local structured-answer extractor."""

    DATASET_URL = {TOY_DATASET: None}
    DATASET_MD5 = {TOY_DATASET: None}

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        return _evaluate_toy(self, eval_file)


class Game2RealRealWorldDataset(RealWorldFinalDataset):
    """Real World split with task-aware and SpatiaLQA graph evaluation."""

    DATASET_URL = {REAL_WORLD_DATASET: None}
    DATASET_MD5 = {REAL_WORLD_DATASET: None}

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        return _evaluate_real_world(self, eval_file)
