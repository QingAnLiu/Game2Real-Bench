from __future__ import annotations

import json
import os.path as osp
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .image_mixedqa import ImageMixedQADataset
from ..smp import dump, get_intermediate_file_path, load


CATEGORY_ORDER = [
    "Fine-grained Perception",
    "Spatial Reasoning",
    "Constraint Reasoning",
    "Sequential Planning",
]


class RealWorldFinalDataset(ImageMixedQADataset):
    """Portable RealWorld Final v1 loader and deterministic evaluator."""

    TYPE = "MixQA"
    # Game2Real-Bench release aliases. The corresponding TSV files are
    # prepared under $LMUData by utils/prepare_game2real_data.py.
    DATASET_URL = {}
    DATASET_MD5 = {}

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs  # All RealWorld Final v1 evaluators are deterministic.
        predictions = load(eval_file).copy()
        if "prediction" not in predictions:
            raise ValueError(f"prediction column is missing from {eval_file}")

        metadata = self.data.copy()
        metadata["_index_key"] = metadata["index"].map(str)
        predictions["_index_key"] = predictions["index"].map(str)
        meta_by_index = metadata.set_index("_index_key", drop=False)
        unknown = sorted(set(predictions["_index_key"]) - set(meta_by_index.index))
        if unknown:
            raise ValueError(f"prediction file contains unknown indices: {unknown[:10]}")

        hits: list[int] = []
        logs: list[str] = []
        for _, prediction_row in predictions.iterrows():
            record = meta_by_index.loc[prediction_row["_index_key"]].to_dict()
            record["_bundle_root"] = self.data_dir
            raw_prediction = prediction_row["prediction"]
            evaluator_type = str(record["evaluator_type"])
            candidate = extract_prediction(raw_prediction, evaluator_type)
            try:
                correct = score_prediction(candidate, record)
                hits.append(int(correct))
                logs.append(
                    f"evaluator={evaluator_type}; extracted={candidate!r}; correct={bool(correct)}"
                )
            except Exception as exc:
                hits.append(0)
                logs.append(
                    f"evaluator={evaluator_type}; extracted={candidate!r}; "
                    f"error={type(exc).__name__}: {exc}"
                )

        predictions["hit"] = hits
        predictions["log"] = logs
        for column in (
            "answer",
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
            values = meta_by_index[column].to_dict()
            predictions[column] = predictions["_index_key"].map(values)
        predictions.drop(columns=["_index_key"], inplace=True)

        result_file = get_intermediate_file_path(eval_file, "_deterministic_result")
        dump(predictions, result_file)
        scores = build_score_table(predictions)
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        dump(scores, score_file)
        return scores


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_cell(value: Any, default: Any) -> Any:
    if value is None or _is_missing(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    return json.loads(text)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text).rstrip(".")


def normalize_numeric(value: Any) -> str:
    text = normalize_text(value).replace(",", "")
    number = float(text)
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def normalize_answer(value: Any, evaluator_type: str) -> Any:
    if evaluator_type == "mcq_single":
        matches = re.findall(r"\b([A-F])\b", str(value).upper())
        return matches[-1] if matches else str(value).strip().upper()
    if evaluator_type == "numeric":
        return normalize_numeric(value)
    if evaluator_type == "exact_short_text":
        return normalize_text(value)
    if evaluator_type in {"json_ordered_list", "json_unordered_set", "ranking"}:
        parsed = _json_cell(value, value)
        if not isinstance(parsed, list):
            raise ValueError(f"{evaluator_type} answer must be a list")
        normalized = [normalize_text(item).upper() for item in parsed]
        return sorted(normalized) if evaluator_type == "json_unordered_set" else normalized
    if evaluator_type == "json_canonical":
        return canonical_json(_json_cell(value, value))
    if evaluator_type == "route_plan":
        return normalize_route_plan(value)
    if evaluator_type == "bbox_iou":
        return normalize_bbox(_json_cell(value, value))
    if evaluator_type == "bbox_set_iou":
        parsed = _json_cell(value, value)
        if not isinstance(parsed, list):
            raise ValueError("bbox_set_iou answer must be a list")
        return [normalize_bbox(item) for item in parsed]
    if evaluator_type == "point_set_distance":
        parsed = _json_cell(value, value)
        if not isinstance(parsed, list):
            raise ValueError("point_set_distance answer must be a list")
        return [[float(point[0]), float(point[1])] for point in parsed]
    if evaluator_type == "normalized_point_in_mask":
        parsed = _json_cell(value, value)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("normalized_point_in_mask answer must be a non-empty list")
        points = []
        for point in parsed:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("normalized point must be [x,y]")
            x, y = float(point[0]), float(point[1])
            if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("normalized point coordinates must be in [0,1]")
            points.append([x, y])
        return points
    raise ValueError(f"unsupported evaluator_type: {evaluator_type}")


def canonical_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonical_json(item) for item in value]
    if isinstance(value, str):
        return normalize_text(value)
    return value


def normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("bounding box must be [x1,y1,x2,y2]")
    box = [float(item) for item in value]
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("invalid bounding box coordinates")
    return box


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def greedy_bbox_match(predicted, expected, threshold: float) -> bool:
    if len(predicted) != len(expected):
        return False
    remaining = list(expected)
    for box in predicted:
        scores = [bbox_iou(box, candidate) for candidate in remaining]
        if not scores:
            return False
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < threshold:
            return False
        remaining.pop(best)
    return True


def greedy_point_match(predicted, expected, tolerance: float) -> bool:
    if len(predicted) != len(expected):
        return False
    remaining = list(expected)
    for point in predicted:
        distances = [
            ((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2) ** 0.5
            for other in remaining
        ]
        if not distances:
            return False
        best = min(range(len(distances)), key=distances.__getitem__)
        if distances[best] > tolerance:
            return False
        remaining.pop(best)
    return True


def normalize_station(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"\bstation\b", "", text)
    text = re.sub(r"站$", "", text)
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def normalize_route_plan(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            value = json.loads(text)
        else:
            sections = [part.strip() for part in re.split(r"\n?\s*--\s*\n?", text) if part.strip()]
            parsed = []
            for section in sections:
                fields: dict[str, Any] = {}
                for line in section.splitlines():
                    if ":" not in line:
                        continue
                    key, raw = line.split(":", 1)
                    key, raw = normalize_text(key), raw.strip()
                    if key == "route name":
                        fields["route_name"] = normalize_text(raw)
                    elif key == "departure stop":
                        fields["departure_stop"] = normalize_station(raw)
                    elif key == "arrival stop":
                        fields["arrival_stop"] = normalize_station(raw)
                    elif key == "number of via stops":
                        fields["num_via_stops"] = int(raw)
                    elif key == "via stops":
                        fields["via_stops"] = (
                            [] if normalize_text(raw) in {"none", "n/a", ""}
                            else [normalize_station(item) for item in raw.split(",") if item.strip()]
                        )
                if fields:
                    parsed.append(fields)
            value = parsed
    if not isinstance(value, list) or not value:
        raise ValueError("route_plan answer must contain at least one route section")
    normalized = []
    for section in value:
        if not isinstance(section, dict):
            raise ValueError("route_plan sections must be objects")
        item = {}
        for key in ("route_name", "departure_stop", "arrival_stop", "num_via_stops", "via_stops"):
            if key not in section:
                continue
            if key == "route_name":
                item[key] = normalize_text(section[key])
            elif key in {"departure_stop", "arrival_stop"}:
                item[key] = normalize_station(section[key])
            elif key == "num_via_stops":
                item[key] = int(section[key])
            else:
                stops = section[key]
                if isinstance(stops, str):
                    stops = [] if normalize_text(stops) in {"none", "n/a", ""} else stops.split(",")
                item[key] = [normalize_station(stop) for stop in stops]
        normalized.append(item)
    return normalized


def resolve_mask(mask_path: str, bundle_root: str) -> Path:
    path = Path(mask_path)
    if path.is_absolute() and path.is_file():
        return path
    candidate = Path(bundle_root) / Path(*str(mask_path).replace("\\", "/").split("/"))
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"verification mask not found: {mask_path}")


def points_hit_mask(points, payload, bundle_root: str) -> bool:
    mask_path = str(payload.get("mask_path") or "")
    if not mask_path:
        raise ValueError("normalized_point_in_mask evaluator requires mask_path")
    with Image.open(resolve_mask(mask_path, bundle_root)) as image:
        mask = np.asarray(image.convert("L")) > 0
    height, width = mask.shape
    if width != int(payload.get("image_width", width)) or height != int(payload.get("image_height", height)):
        raise ValueError("verification mask dimensions do not match evaluator payload")
    return any(mask[int(round(y * (height - 1))), int(round(x * (width - 1)))] for x, y in points)


def score_prediction(prediction: Any, record: dict[str, Any]) -> bool:
    evaluator_type = str(record["evaluator_type"])
    expected_values = _json_cell(record.get("accepted_answers"), []) or [record.get("answer")]
    payload = _json_cell(record.get("evaluator_payload"), {})
    normalized_prediction = normalize_answer(prediction, evaluator_type)
    if evaluator_type == "bbox_iou":
        threshold = float(payload.get("iou_threshold", 0.5))
        return any(
            bbox_iou(normalized_prediction, normalize_answer(expected, evaluator_type)) >= threshold
            for expected in expected_values
        )
    if evaluator_type == "bbox_set_iou":
        threshold = float(payload.get("iou_threshold", 0.5))
        return any(
            greedy_bbox_match(normalized_prediction, normalize_answer(expected, evaluator_type), threshold)
            for expected in expected_values
        )
    if evaluator_type == "point_set_distance":
        tolerance = float(payload.get("pixel_tolerance", 20.0))
        return any(
            greedy_point_match(normalized_prediction, normalize_answer(expected, evaluator_type), tolerance)
            for expected in expected_values
        )
    if evaluator_type == "normalized_point_in_mask":
        return points_hit_mask(normalized_prediction, payload, str(record["_bundle_root"]))
    return any(
        normalized_prediction == normalize_answer(expected, evaluator_type)
        for expected in expected_values
    )


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    answer_tags = re.findall(r"(?is)<answer>\s*(.*?)\s*</answer>", text)
    if answer_tags:
        text = answer_tags[-1].strip()
    markers = list(re.finditer(r"(?im)^\s*(?:final\s+answer|answer)\s*[:：]\s*", text))
    if markers:
        text = text[markers[-1].end():].strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _best_json_value(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    values = []
    for position, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(text[position:])
            values.append((end, position, value))
        except json.JSONDecodeError:
            continue
    if not values:
        return None
    # Prefer the largest complete JSON span. This preserves an outer list such
    # as [[x1,y1,x2,y2], ...] instead of accidentally selecting its last child.
    return max(values, key=lambda item: (item[0], item[1]))[2]


def extract_prediction(prediction: Any, evaluator_type: str) -> Any:
    text = _strip_wrappers(str(prediction))
    structured = {
        "json_ordered_list",
        "json_unordered_set",
        "ranking",
        "json_canonical",
        "bbox_iou",
        "bbox_set_iou",
        "point_set_distance",
        "normalized_point_in_mask",
    }
    if evaluator_type in structured:
        value = _best_json_value(text)
        return json.dumps(value, ensure_ascii=False) if value is not None else text
    if evaluator_type == "route_plan":
        value = _best_json_value(text)
        return json.dumps(value, ensure_ascii=False) if isinstance(value, list) else text
    if evaluator_type == "mcq_single":
        matches = re.findall(r"\b([A-F])\b", text.upper())
        return matches[-1] if matches else text
    if evaluator_type == "numeric":
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        return matches[-1] if matches else text
    return text


def build_score_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, group: str, subset: pd.DataFrame, accuracy: float | None = None) -> None:
        if subset.empty:
            return
        if accuracy is None:
            accuracy = float(subset["hit"].mean())
        rows.append(
            {
                "scope": scope,
                "group": group,
                "num_samples": int(len(subset)),
                "accuracy": accuracy,
            }
        )

    core = data[data["evaluation_role"] == "core"]
    capability_scores = {
        category: float(core[core["category"] == category]["hit"].mean())
        for category in CATEGORY_ORDER
        if not core[core["category"] == category].empty
    }
    overall = (
        float(np.mean([capability_scores[category] for category in CATEGORY_ORDER]))
        if len(capability_scores) == len(CATEGORY_ORDER)
        else np.nan
    )
    add("overall", "macro_four_capabilities", core, overall)
    add("overall", "micro_core", core)
    for category in CATEGORY_ORDER:
        add("capability", category, core[core["category"] == category])

    clean = core[core["contamination_risk"].isin(["low", "medium"])]
    clean_scores = {
        category: float(clean[clean["category"] == category]["hit"].mean())
        for category in CATEGORY_ORDER
        if not clean[clean["category"] == category].empty
    }
    clean_overall = (
        float(np.mean([clean_scores[category] for category in CATEGORY_ORDER]))
        if len(clean_scores) == len(CATEGORY_ORDER)
        else np.nan
    )
    add("clean_subset", "macro_four_capabilities", clean, clean_overall)
    for group in sorted(data["evaluator_type"].dropna().astype(str).unique()):
        add("evaluator", group, data[data["evaluator_type"].astype(str) == group])
    for column, scope in (
        ("source", "source"),
        ("primary_toy_task", "toy_task"),
        ("mechanism_alignment_level", "alignment"),
        ("input_format", "input_format"),
    ):
        for group in sorted(core[column].dropna().astype(str).unique()):
            add(scope, group, core[core[column].astype(str) == group])
    diagnostic = data[data["evaluation_role"] == "diagnostic"]
    for group in sorted(diagnostic["primary_toy_task"].dropna().astype(str).unique()):
        add("diagnostic_toy_task", group, diagnostic[diagnostic["primary_toy_task"].astype(str) == group])
    return pd.DataFrame(rows, columns=["scope", "group", "num_samples", "accuracy"])
