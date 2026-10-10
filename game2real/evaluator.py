"""Deterministic scoring for Game2Real-Bench predictions."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationResult:
    correct: bool
    extracted: Any
    evaluator: str
    detail: str = ""


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _json_cell(value: Any, default: Any) -> Any:
    if _missing(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    return default if not text else json.loads(text)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text).rstrip(".")


def normalize_numeric(value: Any) -> str:
    number = float(normalize_text(value).replace(",", ""))
    return str(int(number)) if number.is_integer() else format(number, ".12g")


def canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key).casefold(): canonical_json(value[key]) for key in sorted(value, key=lambda x: str(x).casefold())}
    if isinstance(value, list):
        return [canonical_json(item) for item in value]
    if isinstance(value, str):
        text = normalize_text(value)
        coordinate_items = [item.strip() for item in text.split(",")]
        if len(coordinate_items) > 1 and all(re.fullmatch(r"r\d+c\d+", item) for item in coordinate_items):
            return coordinate_items
        direction_words = {
            "u": "u",
            "up": "u",
            "d": "d",
            "down": "d",
            "l": "l",
            "left": "l",
            "r": "r",
            "right": "r",
        }
        direction_items = [item for item in re.split(r"[\s,;]+", text) if item]
        if len(direction_items) > 1 and all(item in direction_words for item in direction_items):
            return [direction_words[item] for item in direction_items]
        try:
            return normalize_numeric(text)
        except ValueError:
            return text
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return normalize_numeric(value)
    return value


def _strip_wrappers(value: Any) -> str:
    text = str(value).strip()
    tags = re.findall(r"(?is)<answer>\s*(.*?)\s*</answer>", text)
    if tags:
        text = tags[-1].strip()
    markers = list(re.finditer(r"(?im)^\s*(?:final\s+answer|answer)\s*[:：]\s*", text))
    if markers:
        text = text[markers[-1].end() :].strip()
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
    values: list[tuple[int, int, Any]] = []
    for position, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(text[position:])
            values.append((end, position, value))
        except json.JSONDecodeError:
            continue
    return max(values, key=lambda item: (item[0], item[1]))[2] if values else None


def normalize_station(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\bstation\b", "", text)
    text = re.sub(r"站$", "", text)
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def normalize_route_plan(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        text = _strip_wrappers(value)
        parsed = _best_json_value(text)
        if isinstance(parsed, list):
            value = parsed
        else:
            sections = [part.strip() for part in re.split(r"\n?\s*--\s*\n?", text) if part.strip()]
            value = []
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
                        fields["via_stops"] = [] if normalize_text(raw) in {"none", "n/a", ""} else [normalize_station(item) for item in raw.split(",")]
                if fields:
                    value.append(fields)
    if not isinstance(value, list) or not value:
        raise ValueError("route_plan answer must contain at least one route section")
    normalized: list[dict[str, Any]] = []
    for section in value:
        if not isinstance(section, Mapping):
            raise ValueError("route_plan sections must be objects")
        item: dict[str, Any] = {}
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


def _extract_real(prediction: Any, evaluator_type: str) -> Any:
    text = _strip_wrappers(prediction)
    if evaluator_type == "mcq_single":
        matches = re.findall(r"\b([A-F])\b", text.upper())
        return matches[-1] if matches else text.upper()
    if evaluator_type == "numeric":
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        return matches[-1] if matches else text
    if evaluator_type == "route_plan":
        return normalize_route_plan(text)
    if evaluator_type in {"json_ordered_list", "json_unordered_set", "ranking", "json_canonical"}:
        parsed = _best_json_value(text)
        return parsed if parsed is not None else text
    return text


def _normalize_real(value: Any, evaluator_type: str) -> Any:
    if evaluator_type == "mcq_single":
        matches = re.findall(r"\b([A-F])\b", str(value).upper())
        return matches[-1] if matches else str(value).strip().upper()
    if evaluator_type == "numeric":
        return normalize_numeric(value)
    if evaluator_type == "exact_short_text":
        return normalize_text(value)
    if evaluator_type in {"json_ordered_list", "ranking"}:
        parsed = _json_cell(value, value)
        if not isinstance(parsed, list):
            raise ValueError(f"{evaluator_type} answer must be a list")
        return [normalize_text(item).upper() for item in parsed]
    if evaluator_type == "json_unordered_set":
        parsed = _json_cell(value, value)
        if not isinstance(parsed, list):
            raise ValueError("json_unordered_set answer must be a list")
        return sorted(normalize_text(item).upper() for item in parsed)
    if evaluator_type == "json_canonical":
        return canonical_json(_json_cell(value, value))
    if evaluator_type == "route_plan":
        return normalize_route_plan(value)
    raise ValueError(f"unsupported Real World evaluator_type: {evaluator_type}")


def evaluate_real(prediction: Any, record: Mapping[str, Any]) -> EvaluationResult:
    evaluator_type = str(record["evaluator_type"])
    extracted = _extract_real(prediction, evaluator_type)
    normalized_prediction = _normalize_real(extracted, evaluator_type)
    expected_values = _json_cell(record.get("accepted_answers"), []) or [record.get("answer")]
    correct = any(normalized_prediction == _normalize_real(expected, evaluator_type) for expected in expected_values)
    return EvaluationResult(correct, extracted, evaluator_type)


def _parse_key_value_surface(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    text = _strip_wrappers(value)
    parsed = _best_json_value(text)
    if isinstance(parsed, Mapping):
        return dict(parsed)
    parts = [part.strip() for part in text.split(";") if part.strip()]
    result: dict[str, Any] = {}
    for part in parts:
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_]*)\s*=\s*([^;=]+)", part)
        if not match:
            return None
        result[match.group(1)] = match.group(2).strip()
    return result or None


def _normalize_toy_surface(value: Any) -> Any:
    if isinstance(value, Mapping):
        return canonical_json(value)
    if isinstance(value, list):
        return canonical_json(value)
    mapping = _parse_key_value_surface(value)
    if mapping is not None:
        return canonical_json(mapping)
    text = normalize_text(value)
    text = re.sub(r"\s*([=;,])\s*", r"\1", text)
    return canonical_json(text)


def evaluate_toy(prediction: Any, record: Mapping[str, Any]) -> EvaluationResult:
    question_type = str(record.get("question_type") or "open")
    if question_type != "open":
        extracted = _extract_real(prediction, "mcq_single")
        correct = _normalize_real(extracted, "mcq_single") == _normalize_real(record["answer"], "mcq_single")
        return EvaluationResult(correct, extracted, "toy_single_choice")
    accepted = _json_cell(record.get("accepted_answers"), []) or [record.get("answer")]
    extracted = _strip_wrappers(prediction)
    normalized = _normalize_toy_surface(extracted)
    correct = any(normalized == _normalize_toy_surface(expected) for expected in accepted)
    schema = "toy_open"
    try:
        payload = _json_cell(record.get("evaluator_payload"), {})
        schema = str(payload.get("open_schema_id") or schema)
    except (ValueError, TypeError):
        pass
    return EvaluationResult(correct, extracted, schema)


def evaluate_record(prediction: Any, record: Mapping[str, Any]) -> EvaluationResult:
    partition = str(record.get("partition") or "")
    if partition == "real_world" or record.get("evaluator_type"):
        return evaluate_real(prediction, record)
    if partition == "toy_game" or record.get("question_type"):
        return evaluate_toy(prediction, record)
    raise ValueError("record does not identify a Game2Real-Bench partition")
