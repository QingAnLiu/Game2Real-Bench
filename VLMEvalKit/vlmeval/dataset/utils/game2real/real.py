"""Official deterministic evaluator helpers for Game2Real Real World."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "Game2Real_RealWorld.tsv"
SWEEP_ROOT = ROOT / "outputs_game2real"
CATEGORY_TO_CODE = {
    "Fine-grained Perception": "C1",
    "Spatial Reasoning": "C2",
    "Constraint Reasoning": "C3",
    "Sequential Planning": "C4",
}


def is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalized_text(value: Any) -> str:
    if value is None or is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("joy-con", "controller").replace("joy con", "controller")
    text = text.replace("on/off", "on off").replace("on-off", "on off")
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_response_wrappers(value: Any) -> str:
    text = "" if value is None or is_missing(value) else str(value).strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()
    answer_tags = re.findall(r"(?is)<(?:answer|ans)>\s*(.*?)\s*</(?:answer|ans)>", text)
    if answer_tags:
        text = answer_tags[-1].strip()
    boxed = re.findall(r"(?is)\\boxed\s*\{([^{}]*)\}", text)
    if boxed:
        text = boxed[-1].strip()
    markers = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+(?:answer|response)|answer|choice|result)\s*[:：]\s*",
            text,
        )
    )
    if markers:
        text = text[markers[-1].end() :].strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json|javascript|python|text)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def json_values(text: str) -> list[tuple[int, int, Any]]:
    values: list[tuple[int, int, Any]] = []
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, length = decoder.raw_decode(text[position:])
            values.append((length, position, value))
        except json.JSONDecodeError:
            continue
    return values


def best_structured_value(text: str) -> Optional[Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        value = ast.literal_eval(stripped)
        if isinstance(value, (dict, list, tuple, set, str, int, float)):
            return value
    except (ValueError, SyntaxError):
        pass
    values = json_values(stripped)
    if values:
        return max(values, key=lambda item: (item[0], -item[1]))[2]
    return None


def option_labels_from_row(row: dict[str, Any]) -> set[str]:
    return {
        label
        for label in "ABCDEF"
        if label in row and not is_missing(row[label]) and str(row[label]).strip()
    }


def extract_mcq(text: str, row: dict[str, Any]) -> str:
    labels = option_labels_from_row(row) or set("ABCDEF")
    structured = best_structured_value(text)
    if isinstance(structured, dict):
        for key in ("answer", "final_answer", "choice", "option", "label"):
            if key in structured:
                candidate = str(structured[key]).strip().upper()
                match = re.search(r"\b([A-F])\b", candidate)
                if match and match.group(1) in labels:
                    return match.group(1)
    patterns = [
        r"(?i)(?:final\s+answer|answer|choice|option)\s*(?:is|=|:|：)?\s*\(?([A-F])\)?\b",
        r"(?i)\bselect(?:ed|ing)?\s+(?:option\s*)?\(?([A-F])\)?\b",
        r"(?i)\btherefore\s*,?\s*(?:the\s+answer\s+is\s*)?\(?([A-F])\)?\b",
    ]
    for pattern in patterns:
        matches = []
        for match in re.finditer(pattern, text):
            lookback = text[max(0, match.start() - 40) : match.start()].casefold()
            if re.search(r"\b(?:assume|guess|maybe|perhaps|possibly)\b", lookback):
                continue
            label = match.group(1).upper()
            if label in labels:
                matches.append(label)
        if matches:
            return matches[-1]
    compact = re.sub(r"[`*_\s.()\[\]{}]", "", text).upper()
    if compact in labels:
        return compact
    matches = [m for m in re.findall(r"\b([A-F])\b", text.upper()) if m in labels]
    return matches[-1] if matches else text.strip()


def extract_numeric(text: str) -> str:
    structured = best_structured_value(text)
    if isinstance(structured, dict):
        for key in ("answer", "final_answer", "result", "count", "total"):
            value = structured.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if isinstance(value, str) and re.fullmatch(
                r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", value.strip()
            ):
                return value.strip()
    trailing = re.search(
        r"(?:^|\n)\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\Z", text.rstrip()
    )
    if trailing:
        return trailing.group(1)
    patterns = [
        r"(?i)(?:final\s+answer|answer|result)\s*(?:is|=|:|：)?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?im)^\s*total\b[^\n]*=\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\w+)?\s*[.]?\s*$",
        r"(?i)\b(?:sum|total)\b[^\n=]{0,300}=\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?i)\b(?:has|have|contains?|count(?:s|ed)?)\s+([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:keys?|items?|objects?|oranges?|blocks?|pieces?)\b",
        r"(?i)\b(?:therefore|thus|so)\b[^\d+-]*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return matches[-1] if matches else text.strip()


def extract_short_text(text: str) -> str:
    yes_no = re.findall(r"(?i)\b(yes|no)\b", text)
    if yes_no:
        return yes_no[-1]
    quoted = re.findall(r"[\"']([^\"']{1,100})[\"']", text)
    if quoted:
        return quoted[-1].strip()
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def extract_list_like(text: str) -> str:
    value = best_structured_value(text)
    if isinstance(value, (tuple, set)):
        value = list(value)
    if isinstance(value, dict):
        for key in ("answer", "ranking", "order", "items", "labels", "objects"):
            if isinstance(value.get(key), (list, tuple)):
                value = list(value[key])
                break
            if isinstance(value.get(key), str):
                candidate = value[key].strip()
                if "," in candidate:
                    value = [part.strip(" \t\r\n\"'") for part in candidate.split(",") if part.strip()]
                    break
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "[{" and stripped[-1] in "]}":
        inner = stripped[1:-1].strip()
        parts = [part.strip(" \t\r\n\"'") for part in inner.split(",") if part.strip()]
        if len(parts) >= 2 and all("\n" not in part for part in parts):
            return json.dumps(parts, ensure_ascii=False)
    frame_labels = re.findall(r"(?i)\bF\d+\b", stripped)
    if len(frame_labels) >= 2 and re.fullmatch(r"(?is)[\s,;|\[\]{}\"'F\d]+", stripped):
        return json.dumps([label.upper() for label in frame_labels], ensure_ascii=False)
    if "," in stripped and "\n" not in stripped and len(stripped) <= 500:
        parts = [part.strip(" \t\r\n\"'") for part in stripped.split(",") if part.strip()]
        if len(parts) >= 2 and all(len(part) <= 100 for part in parts):
            return json.dumps(parts, ensure_ascii=False)
    rank_parts = [part.strip() for part in re.split(r"\s*(?:>|→|->)\s*", text) if part.strip()]
    if len(rank_parts) >= 2:
        cleaned = [re.sub(r"(?i)^(?:rank\s*\d+|\d+)[.):\s-]+", "", part).strip() for part in rank_parts]
        return json.dumps(cleaned, ensure_ascii=False)
    bullet_parts = []
    for line in text.splitlines():
        match = re.match(r"\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$", line)
        if match:
            bullet_parts.append(match.group(1))
    if bullet_parts:
        return json.dumps(bullet_parts, ensure_ascii=False)
    labels = re.findall(r"\b([A-F])\b", text.upper())
    if len(labels) >= 2:
        return json.dumps(labels, ensure_ascii=False)
    return text.strip()


def extract_route(text: str) -> str:
    value = best_structured_value(text)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return text.strip()


def blind_extract(prediction: Any, row: dict[str, Any]) -> tuple[str, str]:
    """Extract without reading answer, accepted_answers, or evaluator_payload."""
    evaluator_type = str(row["evaluator_type"])
    text = strip_response_wrappers(prediction)
    if evaluator_type == "mcq_single":
        extracted = extract_mcq(text, row)
    elif evaluator_type == "numeric":
        extracted = extract_numeric(text)
    elif evaluator_type == "exact_short_text":
        extracted = extract_short_text(text)
    elif evaluator_type in {"ranking", "json_unordered_set", "json_ordered_list"}:
        extracted = extract_list_like(text)
    elif evaluator_type == "route_plan":
        extracted = extract_route(text)
    else:
        extracted = text
    reason = "unchanged_wrapped_response" if extracted == str(prediction).strip() else "blind_task_specific_reextract"
    return extracted, reason


def parse_json_cell(value: Any, default: Any) -> Any:
    if value is None or is_missing(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def normalize_scalar_answer(value: Any, evaluator_type: str) -> Any:
    if evaluator_type == "mcq_single":
        matches = re.findall(r"\b([A-F])\b", str(value).upper())
        return matches[-1] if matches else str(value).strip().upper()
    if evaluator_type == "numeric":
        text = unicodedata.normalize("NFKC", str(value)).strip().replace(",", "")
        if re.fullmatch(r"[-+]?\d{1,3}(?:\s\d{3})+(?:\.\d+)?", text):
            text = re.sub(r"\s+", "", text)
        number = float(text)
        return str(int(number)) if number.is_integer() else format(number, ".12g")
    if evaluator_type == "exact_short_text":
        return normalized_text(value).rstrip(".")
    if evaluator_type in {"json_ordered_list", "json_unordered_set", "ranking"}:
        parsed = parse_json_cell(value, value)
        if isinstance(parsed, tuple):
            parsed = list(parsed)
        if not isinstance(parsed, list):
            raise ValueError(f"{evaluator_type} answer must be a list")
        normalized = [normalized_text(item).upper() for item in parsed]
        return sorted(normalized) if evaluator_type == "json_unordered_set" else normalized
    if evaluator_type == "route_plan":
        return normalize_route_plan(value)
    raise ValueError(f"unsupported non-SpatiaLQA evaluator_type: {evaluator_type}")


def normalize_station(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"\bstation\b", "", text)
    text = re.sub(r"站$", "", text)
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def normalize_route_plan(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        parsed_value = best_structured_value(value)
        if isinstance(parsed_value, list):
            value = parsed_value
        else:
            sections = [part.strip() for part in re.split(r"\n?\s*--\s*\n?", value) if part.strip()]
            parsed_sections = []
            for section in sections:
                fields: dict[str, Any] = {}
                for line in section.splitlines():
                    if ":" not in line:
                        continue
                    key, raw = line.split(":", 1)
                    key, raw = normalized_text(key), raw.strip()
                    if key == "route name":
                        fields["route_name"] = normalized_text(raw)
                    elif key == "departure stop":
                        fields["departure_stop"] = normalize_station(raw)
                    elif key == "arrival stop":
                        fields["arrival_stop"] = normalize_station(raw)
                    elif key == "number of via stops":
                        count_match = re.search(r"\d+", raw)
                        if count_match:
                            fields["num_via_stops"] = int(count_match.group())
                        if "(" in raw and ")" in raw and "via_stops" not in fields:
                            named = raw[raw.find("(") + 1 : raw.rfind(")")]
                            named = re.sub(r"(?i)^\s*stops?\s*:\s*", "", named)
                            fields["via_stops"] = [
                                normalize_station(item) for item in named.split(",") if item.strip()
                            ]
                    elif key == "via stops":
                        fields["via_stops"] = (
                            [] if normalized_text(raw) in {"none", "n/a", ""}
                            else [normalize_station(item) for item in raw.split(",") if item.strip()]
                        )
                if fields:
                    parsed_sections.append(fields)
            value = parsed_sections
    if not isinstance(value, list) or not value:
        raise ValueError("route_plan answer must contain at least one route section")
    normalized = []
    for section in value:
        if not isinstance(section, dict):
            raise ValueError("route_plan sections must be objects")
        item: dict[str, Any] = {}
        for key in ("route_name", "departure_stop", "arrival_stop", "num_via_stops", "via_stops"):
            if key not in section:
                continue
            if key == "route_name":
                item[key] = normalized_text(section[key])
            elif key in {"departure_stop", "arrival_stop"}:
                item[key] = normalize_station(section[key])
            elif key == "num_via_stops":
                item[key] = int(section[key])
            else:
                stops = section[key]
                if isinstance(stops, str):
                    stops = [] if normalized_text(stops) in {"none", "n/a", ""} else stops.split(",")
                item[key] = [normalize_station(stop) for stop in stops]
        normalized.append(item)
    return normalized


def score_non_spatialqa(extracted: str, row: dict[str, Any]) -> bool:
    evaluator_type = str(row["evaluator_type"])
    expected_values = parse_json_cell(row.get("accepted_answers"), []) or [row.get("answer")]
    candidate = normalize_scalar_answer(extracted, evaluator_type)
    return any(candidate == normalize_scalar_answer(expected, evaluator_type) for expected in expected_values)


def content_from_mapping(value: dict[str, Any]) -> str:
    if "content" in value:
        content = value["content"]
        if isinstance(content, dict):
            inner = " ".join(f"{key} {item}" for key, item in content.items() if key != "precondition")
            if inner.strip():
                return inner
            outer = " ".join(
                f"{key} {value[key]}"
                for key in ("action", "object", "source", "destination", "into", "onto", "target")
                if key in value and str(value[key]).strip()
            )
            return outer
        return str(content)
    step_id = str(value.get("step_id", "")).strip()
    if len(step_id.split()) >= 3:
        return step_id
    keys = ("action", "object", "source", "destination", "into", "onto", "target")
    parts = [f"{key} {value[key]}" for key in keys if key in value]
    return " ".join(parts)


def parse_spatialqa_steps(value: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if value is None or is_missing(value):
        return {}, ["empty_prediction"]
    if isinstance(value, str):
        text = strip_response_wrappers(value)
        parsed = best_structured_value(text)
        if parsed is None:
            wrapped = text.strip().strip(",")
            if re.search(r"(?:^|,)\s*(?:step_?\s*)?\d+\s*:\s*\{", wrapped, flags=re.I):
                parsed = best_structured_value("{" + wrapped + "}")
        if parsed is None:
            objects = [item[2] for item in json_values(text) if isinstance(item[2], dict) and "content" in item[2]]
            if objects:
                parsed = objects
                warnings.append("recovered_individual_step_objects")
        if parsed is None:
            repaired = text.replace('\\"', '"')
            content_match = re.search(r'(?is)"content"\s*:\s*"([^"]+)"', repaired)
            pre_match = re.search(r'(?is)"precondition"\s*:\s*(\[[^\]]*\])', repaired)
            if content_match:
                preconditions: list[Any] = []
                if pre_match:
                    try:
                        preconditions = json.loads(pre_match.group(1))
                    except json.JSONDecodeError:
                        pass
                parsed = [{"content": content_match.group(1), "precondition": preconditions}]
                warnings.append("repaired_malformed_single_step")
        value = parsed
    if isinstance(value, dict) and "step_ids" in value and isinstance(value["step_ids"], list):
        value = value["step_ids"]
    if isinstance(value, dict) and "content" in value:
        value = [value]
    if isinstance(value, list):
        items = [(f"step{i + 1}", item) for i, item in enumerate(value)]
    elif isinstance(value, dict):
        items = list(value.items())
    else:
        return {}, warnings + ["unparseable_prediction"]

    steps: dict[str, dict[str, Any]] = {}
    raw_to_canonical: dict[str, str] = {}
    for raw_key, item in items:
        if not isinstance(item, dict):
            if isinstance(item, (str, list)):
                item = {"content": " ".join(map(str, item)) if isinstance(item, list) else item}
            else:
                warnings.append(f"ignored_non_step:{raw_key}")
                continue
        content = content_from_mapping(item).strip()
        if not content:
            warnings.append(f"ignored_contentless_step:{raw_key}")
            continue
        canonical = f"step{len(steps) + 1}"
        raw_to_canonical[str(raw_key)] = canonical
        raw_to_canonical[normalized_text(raw_key)] = canonical
        raw_to_canonical[str(len(steps) + 1)] = canonical
        raw_to_canonical[f"step{len(steps) + 1}"] = canonical
        pre = item.get("precondition", [])
        if pre is None:
            pre = []
        if not isinstance(pre, list):
            pre = [pre]
        steps[canonical] = {"content": content, "precondition_raw": pre}

    content_to_key = {normalized_text(step["content"]): key for key, step in steps.items()}
    for step in steps.values():
        refs: list[str] = []
        for ref in step.pop("precondition_raw"):
            ref_text = str(ref)
            key = raw_to_canonical.get(ref_text) or raw_to_canonical.get(normalized_text(ref_text))
            if key is None:
                key = content_to_key.get(normalized_text(ref_text))
            refs.append(key or f"unmatched:{normalized_text(ref_text)}")
        step["precondition"] = refs
    return steps, warnings


ACTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("clean", ("clean", "wipe", "mop", "scrub")),
    ("sprinkle", ("sprinkle", "spread .* detergent", "apply .* detergent")),
    ("fill", ("catch water", "fill .* water", "collect water")),
    ("light", ("light", "ignite")),
    ("plug_in", ("plug .* into", "connect .* socket", "insert .* plug")),
    ("pour", ("pour", "transfer", "water the flowers")),
    ("install", ("install", "attach", "assemble", "connect", "mount")),
    ("uninstall", ("detach", "disassemble", "uninstall")),
    ("open", ("open", "unzip", "remove cap", "remove lid", "take off cap", "take off lid")),
    ("close", ("close", "cover", "zip up", "zip", "replace cap", "replace lid", "put .* cap", "put .* lid")),
    ("power_on", ("power on", "turn on", "switch on", "press the on off", "press on off", "activate", "start .* microwave")),
    ("power_off", ("power off", "turn off", "switch off", "deactivate")),
    ("put_down", ("put down", "set down", "return .* counter", "place .* back")),
    ("place", ("put .* into", "put .* in ", "place .* into", "place .* in ", "insert", "release .* into", "move .* into")),
    ("remove", ("remove", "take .* out", "take out", "take .* from", "extract")),
    ("pick_up", ("pick up", "pickup", "lift", "grasp", "grab", "take hold", "hold", "raise", "upward force")),
    ("press", ("press", "push .* button")),
    ("move", ("move", "approach", "navigate", "bring", "position")),
    ("wait", ("wait", "heat", "boil")),
    ("identify", ("identify", "locate", "find", "ensure", "check")),
]

STOPWORDS = {
    "the", "a", "an", "of", "from", "to", "into", "in", "on", "onto", "at", "with", "and", "or",
    "its", "it", "some", "one", "any", "current", "designated", "nearby", "up", "off", "out", "back",
    "robot", "arm", "gripper", "surface", "position", "location", "side", "top", "bottom", "inside",
}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "gray", "grey", "brown"}
POSITIONS = {"left", "right", "front", "rear", "middle", "center", "upper", "lower", "first", "second", "third"}
MATERIALS = {"glass", "plastic", "transparent", "clear", "stainless", "steel", "silver", "wooden", "metal", "rubber"}
CONTAINER_TOKENS = {"box", "bag", "bottle", "holder", "case", "drawer", "cabinet", "basin", "bucket", "cup", "stapler", "microwave", "helmet", "bowl", "pad"}
GENERIC_PART_TOKENS = {"lid", "cap", "door", "edge", "handle", "tip"}
GENERIC_CONTEXT_TOKENS = {"table", "counter", "countertop", "desk", "surface", "ground", "floor", "scene", "location", "position"}


def canonical_action(text: str) -> str:
    norm = normalized_text(text)
    if re.search(r"\b(?:tap|faucet)\b", norm):
        if re.search(r"\b(?:turn|switch)\s+on\b", norm):
            return "open"
        if re.search(r"\b(?:turn|switch)\s+off\b", norm):
            return "close"
    for action, patterns in ACTION_PATTERNS:
        if any(re.search(rf"\b{pattern}\b", norm) for pattern in patterns):
            return action
    return "unknown"


def semantic_tokens(text: str) -> set[str]:
    norm = semantic_normal_form(text)

    tokens = {token for token in norm.split() if token not in STOPWORDS and len(token) > 1}
    stemming = {"bottles": "bottle", "tablets": "tablet", "cups": "cup", "controllers": "controller",
                "boxes": "box", "bags": "bag", "lighters": "lighter", "books": "book", "shoes": "shoe",
                "keys": "key", "trays": "tray", "lids": "lid", "caps": "cap", "mats": "mat",
                "papers": "paper", "wires": "wire", "hangers": "hanger", "flowers": "flower"}
    return {stemming.get(token, token) for token in tokens}


def semantic_normal_form(text: str) -> str:
    norm = normalized_text(text)
    replacements = {
        "joycon": "controller", "controller controller": "controller", "joystick": "controller",
        "trash bag": "garbage bag", "backpack": "bag", "carrying case": "bag",
        "earphone case": "headphone case", "earbuds": "headphone", "headphones": "headphone",
        "cell phone": "phone", "mobile phone": "phone", "smartphone": "phone",
        "television": "tv", "coffee pot": "teapot", "pitcher": "teapot",
        "effervescent tablet bottle": "green bottle", "tablet bottle": "green bottle",
        "nintendo switch": "game console", "handheld game console": "game console",
        "console screen": "screen", "display unit": "screen",
        "cabinet door": "cabinet", "bag zipper": "bag", "zipper": "bag",
        "wall outlet": "socket", "power outlet": "socket", "power cord": "plug",
        "plant pot": "flower", "potted plant": "flower", "watering can": "bowl",
        "cleaning bottle": "cleaner", "detergent bottle": "cleaner",
        "cigarette pack": "cigarette box", "cardboard box": "paper box", "trash sack": "garbage bag",
        "tea pot": "teapot", "bottle of laundry detergent": "laundry detergent bottle",
        "handbag": "bag", "stationery pouch": "stationery bag", "spectacles": "glasses",
        "trousers": "pants", "hot dog": "ham sausage", "jello": "jelly",
        "chopping board": "cutting board", "table cloth": "tablecloth", "coke bottle": "cola bottle",
        "paper sheet": "paper", "test sheet": "test paper", "marker": "pen", "shears": "scissors",
        "cleaning rag": "cloth", "ipad": "tablet",
        "toaster oven": "microwave", "electric oven": "microwave",
    }
    for old, new in replacements.items():
        norm = norm.replace(old, new)
    return norm


ACTION_WORDS = {
    "pick", "pickup", "remove", "take", "grab", "grasp", "lift", "open", "close", "pour", "put",
    "place", "insert", "install", "attach", "assemble", "connect", "mount", "unzip", "zip", "power",
    "turn", "switch", "press", "push", "activate", "start", "wipe", "clean", "mop", "scrub", "sprinkle",
    "spread", "apply", "cover", "replace", "return", "move", "release", "fill", "collect", "catch", "water",
    "ignite", "light", "set", "down", "up", "out", "off", "on", "into", "back", "use", "force", "edge",
    "handle", "toward", "towards", "vertically", "upward", "upwards", "directly",
}


def semantic_parts(text: str) -> tuple[set[str], set[str]]:
    norm = semantic_normal_form(text)
    relation = re.search(r"\b(?:from|into|onto|inside|under|above|over|to|of|with|near|in|on)\b", norm)
    if relation:
        primary_text = norm[: relation.start()]
        context_text = norm[relation.end() :]
    else:
        primary_text, context_text = norm, ""
    primary = semantic_tokens(primary_text) - ACTION_WORDS
    context = semantic_tokens(context_text) - ACTION_WORDS
    if not primary:
        primary = semantic_tokens(norm) - ACTION_WORDS
    return primary, context


def action_compatible(left: str, right: str) -> tuple[bool, float]:
    if left == right and left != "unknown":
        return True, 1.0
    if {left, right} <= {"remove", "pick_up"}:
        return True, 0.82
    if {left, right} <= {"place", "put_down"}:
        return True, 0.78
    if {left, right} <= {"power_on", "press"}:
        return True, 0.78
    if {left, right} <= {"pour", "place"}:
        return True, 0.72
    if {left, right} <= {"pour", "sprinkle"}:
        return True, 0.76
    if {left, right} <= {"clean", "pour"}:
        return False, 0.0
    if {left, right} <= {"fill", "pour"}:
        return True, 0.72
    return False, 0.0


def pair_similarity(gt_text: str, pred_text: str) -> tuple[float, list[str]]:
    if semantic_normal_form(gt_text) == semantic_normal_form(pred_text):
        return 1.0, ["exact_normalized_content"]
    gt_action, pred_action = canonical_action(gt_text), canonical_action(pred_text)
    compatible, action_score = action_compatible(gt_action, pred_action)
    reasons = [f"actions:{gt_action}/{pred_action}"]
    if not compatible:
        return 0.0, reasons + ["action_mismatch"]
    if "water the flowers" in semantic_normal_form(gt_text) and pred_action == "place":
        return 0.0, reasons + ["watering_is_not_placing_flowers"]
    gt_tokens, pred_tokens = semantic_tokens(gt_text), semantic_tokens(pred_text)
    gt_primary, gt_context = semantic_parts(gt_text)
    pred_primary, pred_context = semantic_parts(pred_text)
    gt_identity_primary = gt_primary
    pred_identity_primary = pred_primary
    if gt_action in {"open", "close"} and gt_primary <= GENERIC_PART_TOKENS and gt_context:
        gt_identity_primary = gt_context
    if pred_action in {"open", "close"} and pred_primary <= GENERIC_PART_TOKENS and pred_context:
        pred_identity_primary = pred_context
    descriptors = COLORS | POSITIONS | MATERIALS
    gt_core = gt_identity_primary - descriptors - GENERIC_PART_TOKENS - {"game"}
    pred_core = pred_identity_primary - descriptors - GENERIC_PART_TOKENS - {"game"}
    overlap = gt_core & pred_core
    if not overlap:
        return 0.0, reasons + ["no_primary_object_overlap"]
    gt_containers, pred_containers = gt_core & CONTAINER_TOKENS, pred_core & CONTAINER_TOKENS
    if bool(gt_containers) != bool(pred_containers) and not (gt_containers & pred_containers):
        return 0.0, reasons + ["container_identity_conflict"]
    if gt_action in {"place", "pour", "install", "plug_in"} and pred_action in {"place", "pour", "install", "plug_in"}:
        gt_destination = gt_context - COLORS - POSITIONS
        pred_destination = pred_context - COLORS - POSITIONS
        if gt_destination and pred_destination and not (gt_destination & pred_destination):
            return 0.0, reasons + ["destination_conflict"]
    if gt_action == pred_action == "remove":
        gt_source = gt_context - descriptors - GENERIC_PART_TOKENS
        pred_source = pred_context - descriptors - GENERIC_PART_TOKENS
        gt_specific = gt_source - GENERIC_CONTEXT_TOKENS
        pred_specific = pred_source - GENERIC_CONTEXT_TOKENS
        if gt_specific and pred_specific and not (gt_specific & pred_specific):
            return 0.0, reasons + ["source_conflict"]
    gt_modifiers = gt_identity_primary & descriptors
    pred_modifiers = pred_identity_primary & descriptors
    if gt_modifiers and pred_modifiers and not gt_modifiers & pred_modifiers:
        return 0.0, reasons + ["identity_modifier_conflict"]
    gt_colors, pred_colors = gt_identity_primary & COLORS, pred_identity_primary & COLORS
    gt_pos, pred_pos = gt_identity_primary & POSITIONS, pred_identity_primary & POSITIONS
    positional_conflicts = ({"left", "right"}, {"front", "rear"}, {"upper", "lower"})
    for pair in positional_conflicts:
        if gt_pos & pair and pred_pos & pair and not (gt_pos & pred_pos & pair):
            return 0.0, reasons + ["position_conflict"]
    union = gt_core | pred_core
    lexical = len(overlap) / len(union) if union else 0.0
    specificity = len(overlap) / min(len(gt_core), len(pred_core)) if gt_core and pred_core else 0.0
    score = 0.55 * action_score + 0.30 * specificity + 0.15 * lexical
    if gt_pos and pred_pos and gt_pos & pred_pos:
        score += 0.06
    if gt_colors and pred_colors and gt_colors & pred_colors:
        score += 0.04
    reasons += [f"overlap:{','.join(sorted(overlap))}", f"score:{score:.3f}"]
    return min(score, 1.0), reasons


def load_overrides(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


CONTENT_OVERRIDE_PATH = Path(__file__).with_name("spatialqa_overrides.json")


def load_content_overrides(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load reviewed content-pair decisions without tying them to model names."""
    override_path = path or CONTENT_OVERRIDE_PATH
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("rules"), list):
        raise ValueError(f"invalid SpatiaLQA override resource: {override_path}")
    rules: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str], int] = {}
    for position, raw_rule in enumerate(payload["rules"], start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"SpatiaLQA override rule {position} must be an object")
        sample_id = str(raw_rule.get("sample_id", "")).strip()
        prediction_sha256 = str(raw_rule.get("prediction_sha256", "")).strip().lower()
        gt_text = str(raw_rule.get("ground_truth", "")).strip()
        pred_text = str(raw_rule.get("prediction", "")).strip()
        decision = raw_rule.get("match")
        if (
            not sample_id
            or not re.fullmatch(r"[0-9a-f]{64}", prediction_sha256)
            or not gt_text
            or not pred_text
            or decision not in (0, 1)
        ):
            raise ValueError(f"SpatiaLQA override rule {position} is incomplete")
        key = (
            sample_id,
            prediction_sha256,
            semantic_normal_form(gt_text),
            semantic_normal_form(pred_text),
        )
        if key in seen and seen[key] != int(decision):
            raise ValueError(f"conflicting SpatiaLQA override for {key}")
        seen[key] = int(decision)
        rules.append({**raw_rule, "match": int(decision)})
    return rules


def spatialqa_prediction_sha256(steps: dict[str, dict[str, Any]]) -> str:
    payload = [
        {
            "content": semantic_normal_form(step["content"]),
            "precondition": [str(value) for value in step.get("precondition", [])],
        }
        for step in steps.values()
    ]
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_public_match_matrix(
    sample_id: str,
    gt_steps: dict[str, dict[str, Any]],
    pred_steps: dict[str, dict[str, Any]],
    threshold: float = 0.72,
    rules: Optional[list[dict[str, Any]]] = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the public semantic matrix and apply reviewed pair-level decisions."""
    gt_items, pred_items = list(gt_steps.items()), list(pred_steps.items())
    matrix = np.zeros((len(gt_items), len(pred_items)), dtype=int)
    uncertain: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    override_rules = load_content_overrides() if rules is None else rules
    prediction_sha256 = spatialqa_prediction_sha256(pred_steps)
    indexed_rules: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for rule in override_rules:
        key = (
            str(rule["sample_id"]),
            str(rule["prediction_sha256"]),
            semantic_normal_form(str(rule["ground_truth"])),
            semantic_normal_form(str(rule["prediction"])),
        )
        indexed_rules[key] = rule

    for gt_index, (_, gt_step) in enumerate(gt_items):
        for pred_index, (_, pred_step) in enumerate(pred_items):
            similarity, reasons = pair_similarity(gt_step["content"], pred_step["content"])
            decision = int(similarity >= threshold)
            rule = indexed_rules.get(
                (
                    str(sample_id),
                    prediction_sha256,
                    semantic_normal_form(gt_step["content"]),
                    semantic_normal_form(pred_step["content"]),
                )
            )
            if rule is not None:
                previous = decision
                decision = int(rule["match"])
                applied.append(
                    {
                        "sample_id": str(sample_id),
                        "gt_index": gt_index,
                        "pred_index": pred_index,
                        "automatic_decision": previous,
                        "reviewed_decision": decision,
                        "reason": str(rule.get("reason", "")),
                    }
                )
            matrix[gt_index, pred_index] = decision
            if 0.55 <= similarity <= 0.85:
                uncertain.append(
                    {
                        "sample_id": str(sample_id),
                        "gt_index": gt_index,
                        "pred_index": pred_index,
                        "gt_content": gt_step["content"],
                        "pred_content": pred_step["content"],
                        "similarity": round(similarity, 6),
                        "decision": decision,
                        "reasons": reasons,
                    }
                )

    # Lock unique exact-content pairs before maximum matching.  A binary
    # semantic matrix can otherwise select an equally valid cross-match and
    # corrupt precondition edges even when prediction and ground truth are
    # identical.
    gt_exact: dict[str, list[int]] = {}
    pred_exact: dict[str, list[int]] = {}
    for index, (_, step) in enumerate(gt_items):
        gt_exact.setdefault(semantic_normal_form(step["content"]), []).append(index)
    for index, (_, step) in enumerate(pred_items):
        pred_exact.setdefault(semantic_normal_form(step["content"]), []).append(index)
    for content in sorted(set(gt_exact) & set(pred_exact)):
        if len(gt_exact[content]) != 1 or len(pred_exact[content]) != 1:
            continue
        gt_index = gt_exact[content][0]
        pred_index = pred_exact[content][0]
        matrix[gt_index, :] = 0
        matrix[:, pred_index] = 0
        matrix[gt_index, pred_index] = 1
    return matrix, uncertain, applied


def build_match_matrix(
    stable_id: str,
    model: str,
    gt_steps: dict[str, dict[str, Any]],
    pred_steps: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gt_items, pred_items = list(gt_steps.items()), list(pred_steps.items())
    matrix = np.zeros((len(gt_items), len(pred_items)), dtype=int)
    audit: list[dict[str, Any]] = []
    key = f"{stable_id}::{model}"
    forced = overrides.get(key, {}).get("matrix")
    for i, (_, gt_step) in enumerate(gt_items):
        for j, (_, pred_step) in enumerate(pred_items):
            similarity, reasons = pair_similarity(gt_step["content"], pred_step["content"])
            decision = int(similarity >= threshold)
            matrix[i, j] = decision
            if 0.55 <= similarity <= 0.85:
                audit.append({
                    "stable_id": stable_id,
                    "model": model,
                    "gt_index": i,
                    "pred_index": j,
                    "gt_content": gt_step["content"],
                    "pred_content": pred_step["content"],
                    "similarity": round(similarity, 6),
                    "initial_decision": decision,
                    "reasons": reasons,
                })
    if forced is not None:
        forced_array = np.asarray(forced, dtype=int)
        if forced_array.shape != matrix.shape:
            raise ValueError(f"override shape mismatch for {key}: {forced_array.shape} != {matrix.shape}")
        matrix = forced_array
    return matrix, audit


def maximum_matches(matrix: np.ndarray) -> list[tuple[int, int]]:
    if matrix.size == 0:
        return []
    # The official matrix is binary, so maximum-cardinality bipartite matching
    # is exactly the Hungarian objective used by SpatiaLQA.
    pred_to_gt: dict[int, int] = {}

    def augment(gt_index: int, seen: set[int]) -> bool:
        candidates = [int(index) for index in np.flatnonzero(matrix[gt_index])]
        for pred_index in candidates:
            if pred_index in seen:
                continue
            seen.add(pred_index)
            if pred_index not in pred_to_gt or augment(pred_to_gt[pred_index], seen):
                pred_to_gt[pred_index] = gt_index
                return True
        return False

    for gt_index in range(matrix.shape[0]):
        augment(gt_index, set())
    return sorted((gt_index, pred_index) for pred_index, gt_index in pred_to_gt.items())


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def spatialqa_metrics(
    gt_steps: dict[str, dict[str, Any]], pred_steps: dict[str, dict[str, Any]], matrix: np.ndarray
) -> dict[str, Any]:
    matches = maximum_matches(matrix)
    content_tp = len(matches)
    content_fp = len(pred_steps) - content_tp
    content_fn = len(gt_steps) - content_tp
    content_precision, content_recall, content_f1 = f1_from_counts(content_tp, content_fp, content_fn)

    gt_keys, pred_keys = list(gt_steps), list(pred_steps)
    pred_to_gt = {pred_keys[pred_i]: gt_keys[gt_i] for gt_i, pred_i in matches}
    gt_edges = {
        (step_key, pre_key)
        for step_key, step in gt_steps.items()
        for pre_key in step.get("precondition", [])
    }
    pred_edges = set()
    for step_key, step in pred_steps.items():
        # Match the official implementation: every unmatched endpoint is
        # represented by the same sentinel and the resulting edge set is
        # de-duplicated.  The only deliberate divergence is that zero-match
        # samples are retained in the aggregate instead of being skipped.
        mapped_step = pred_to_gt.get(step_key, "x")
        for pre_key in step.get("precondition", []):
            mapped_pre = pred_to_gt.get(pre_key, "x")
            pred_edges.add((mapped_step, mapped_pre))
    precondition_tp = len(gt_edges & pred_edges)
    precondition_fp = len(pred_edges - gt_edges)
    precondition_fn = len(gt_edges - pred_edges)
    precondition_precision, precondition_recall, precondition_f1 = f1_from_counts(
        precondition_tp, precondition_fp, precondition_fn
    )
    return {
        "matches": matches,
        "content_tp": content_tp,
        "content_fp": content_fp,
        "content_fn": content_fn,
        "content_precision": content_precision,
        "content_recall": content_recall,
        "content_f1": content_f1,
        "precondition_tp": precondition_tp,
        "precondition_fp": precondition_fp,
        "precondition_fn": precondition_fn,
        "precondition_precision": precondition_precision,
        "precondition_recall": precondition_recall,
        "precondition_f1": precondition_f1,
        "score": (content_f1 + precondition_f1) / 2,
    }


def discover_model_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(model_dir.glob("*_deterministic_result.xlsx"))
        if not files:
            continue
        direct = [path for path in files if path.parent == model_dir]
        if len(direct) != 1:
            raise ValueError(f"expected one direct result file for {model_dir.name}, found {direct}")
        result[model_dir.name] = direct[0]
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or is_missing(value):
        return None
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({key: json_safe(value) for key, value in row.items()}, ensure_ascii=False) + "\n")


def evaluate_model(
    model: str,
    result_path: Path,
    metadata: pd.DataFrame,
    output_dir: Path,
    overrides: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if result_path.suffix.lower() == ".tsv":
        result = pd.read_csv(result_path, sep="\t")
    elif result_path.suffix.lower() == ".csv":
        result = pd.read_csv(result_path)
    else:
        result = pd.read_excel(result_path)
    if result["stable_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate stable_id in {result_path}")
    indexed = result.set_index(result["stable_id"].astype(str), drop=False)
    target_ids = metadata["stable_id"].astype(str).tolist()
    missing = sorted(set(target_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"{model} missing {len(missing)} target IDs: {missing[:5]}")

    detail_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    uncertain_rows: list[dict[str, Any]] = []
    total_counts = Counter()
    for _, meta_series in metadata.sort_values("benchmark_index").iterrows():
        meta = meta_series.to_dict()
        stable_id = str(meta["stable_id"])
        prediction_row = indexed.loc[stable_id]
        prediction = prediction_row["prediction"]
        base = {
            "model": model,
            "benchmark_index": int(meta["benchmark_index"]),
            "stable_id": stable_id,
            "category": meta["category"],
            "category_code": CATEGORY_TO_CODE[meta["category"]],
            "source": meta["source"],
            "evaluator_type": meta["evaluator_type"],
            "prediction": prediction,
            "old_hit": int(prediction_row.get("hit", 0)),
        }
        if meta["source"] == "SpatiaLQA":
            gt_steps, gt_warnings = parse_spatialqa_steps(meta["answer"])
            pred_steps, pred_warnings = parse_spatialqa_steps(prediction)
            override_key = f"{stable_id}::{model}"
            override_reason = str(overrides.get(override_key, {}).get("reason", ""))
            matrix, uncertain = build_match_matrix(
                stable_id, model, gt_steps, pred_steps, overrides, threshold
            )
            metrics = spatialqa_metrics(gt_steps, pred_steps, matrix)
            for key in ("content_tp", "content_fp", "content_fn", "precondition_tp", "precondition_fp", "precondition_fn"):
                total_counts[key] += int(metrics[key])
            row = {
                **base,
                "extracted_answer": json.dumps(pred_steps, ensure_ascii=False),
                "new_score": metrics["score"],
                "score_type": "spatialqa_semantic_graph",
                "content_precision": metrics["content_precision"],
                "content_recall": metrics["content_recall"],
                "content_f1": metrics["content_f1"],
                "precondition_precision": metrics["precondition_precision"],
                "precondition_recall": metrics["precondition_recall"],
                "precondition_f1": metrics["precondition_f1"],
                "content_tp": metrics["content_tp"],
                "content_fp": metrics["content_fp"],
                "content_fn": metrics["content_fn"],
                "precondition_tp": metrics["precondition_tp"],
                "precondition_fp": metrics["precondition_fp"],
                "precondition_fn": metrics["precondition_fn"],
                "semantic_matrix": json.dumps(matrix.tolist()),
                "matches": json.dumps(metrics["matches"]),
                "parse_warnings": json.dumps(gt_warnings + pred_warnings, ensure_ascii=False),
                "semantic_override_reason": override_reason,
            }
            for item in uncertain:
                item["image_path"] = meta["image_path"]
                uncertain_rows.append(item)
            if gt_warnings or pred_warnings:
                audit_rows.append({
                    "model": model,
                    "stable_id": stable_id,
                    "audit_type": "spatialqa_parse_warning",
                    "details": json.dumps(gt_warnings + pred_warnings, ensure_ascii=False),
                })
            if override_reason:
                audit_rows.append({
                    "model": model,
                    "stable_id": stable_id,
                    "audit_type": "spatialqa_image_review_override",
                    "details": override_reason,
                })
        else:
            safe_meta = {
                key: meta.get(key)
                for key in ("evaluator_type", "question", "A", "B", "C", "D", "E", "F")
            }
            extracted, reason = blind_extract(prediction, safe_meta)
            try:
                new_hit = int(score_non_spatialqa(extracted, meta))
                error = ""
            except Exception as exc:
                new_hit = 0
                error = f"{type(exc).__name__}: {exc}"
            row = {
                **base,
                "extracted_answer": extracted,
                "new_score": float(new_hit),
                "score_type": "binary_deterministic",
                "extraction_reason": reason,
                "evaluation_error": error,
            }
            old_hit = int(prediction_row.get("hit", 0))
            if old_hit != new_hit or reason != "unchanged_wrapped_response" or error:
                audit_rows.append({
                    "model": model,
                    "stable_id": stable_id,
                    "audit_type": "answer_reextract",
                    "evaluator_type": meta["evaluator_type"],
                    "old_hit": old_hit,
                    "new_hit": new_hit,
                    "extracted_answer": extracted,
                    "reason": reason,
                    "error": error,
                })
        detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    if len(detail) != 400 or detail["stable_id"].nunique() != 400:
        raise AssertionError(f"{model}: expected 400 unique rows")
    category_counts = detail.groupby("category_code").size().to_dict()
    if category_counts != {"C1": 100, "C2": 100, "C3": 100, "C4": 100}:
        raise AssertionError(f"{model}: category counts {category_counts}")
    if not detail["new_score"].between(0, 1).all():
        raise AssertionError(f"{model}: out-of-range scores")

    category_scores = detail.groupby("category_code")["new_score"].mean().to_dict()
    overall = float(detail["new_score"].mean())
    if not np.isclose(overall, np.mean([category_scores[f"C{i}"] for i in range(1, 5)])):
        raise AssertionError(f"{model}: overall/category reconciliation failed")
    cp, cr, cf = f1_from_counts(total_counts["content_tp"], total_counts["content_fp"], total_counts["content_fn"])
    pp, pr, pf = f1_from_counts(
        total_counts["precondition_tp"], total_counts["precondition_fp"], total_counts["precondition_fn"]
    )
    summary = {
        "model": model,
        **{f"C{i}": category_scores[f"C{i}"] for i in range(1, 5)},
        "Overall_400": overall,
        "SpatiaLQA_mean_composite": float(
            detail.loc[detail["source"] == "SpatiaLQA", "new_score"].mean()
        ),
        "SpatiaLQA_zero_match_samples": int(
            ((detail["source"] == "SpatiaLQA") & (detail["content_tp"] == 0)).sum()
        ),
        "SpatiaLQA_Rc": cr,
        "SpatiaLQA_Pc": cp,
        "SpatiaLQA_Fc": cf,
        "SpatiaLQA_Rp": pr,
        "SpatiaLQA_Pp": pp,
        "SpatiaLQA_Fp": pf,
        **{f"SpatiaLQA_{key}": int(value) for key, value in total_counts.items()},
    }
    write_jsonl(output_dir / f"{model}_real_world_detail.jsonl", detail_rows)
    return summary, audit_rows, uncertain_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=META_PATH)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_game2real_rejudge")
    parser.add_argument("--overrides", type=Path, default=ROOT / "tools" / "spatialqa_match_overrides.json")
    parser.add_argument("--semantic-threshold", type=float, default=0.72)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, sep="\t")
    if len(metadata) != 400 or metadata["stable_id"].astype(str).nunique() != 400:
        raise ValueError("metadata must contain exactly 400 unique stable IDs")
    expected_sources = {
        "VisualTrans": 91, "OmniSpatial": 61, "SpatiaLQA": 50, "SSI-Bench": 50,
        "Visual-Jigsaw": 40, "CAPTURe-real": 30, "SPair-71k": 27,
        "All-Angles-Bench": 20, "ReasonMap": 14, "GQA": 12, "IKEA Manuals at Work": 5,
    }
    if metadata["source"].value_counts().to_dict() != expected_sources:
        raise ValueError("Final-400 source distribution does not match the approved design")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(args.overrides)
    model_files = discover_model_files(args.sweep_root)
    if len(model_files) != 15:
        raise ValueError(f"expected 15 models, found {len(model_files)}")

    summaries: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for model, path in model_files.items():
        print(f"Evaluating {model} from {path.name}", flush=True)
        summary, model_audits, model_uncertain = evaluate_model(
            model, path, metadata, args.output_dir, overrides, args.semantic_threshold
        )
        summaries.append(summary)
        audits.extend(model_audits)
        uncertain.extend(model_uncertain)

    summary_path = args.output_dir / "real_world_model_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.output_dir / "real_world_audit.jsonl", audits)
    write_jsonl(args.output_dir / "spatialqa_uncertain_pairs.jsonl", uncertain)
    print(json.dumps({"models": len(summaries), "audit_rows": len(audits), "uncertain_pairs": len(uncertain)}, indent=2))


