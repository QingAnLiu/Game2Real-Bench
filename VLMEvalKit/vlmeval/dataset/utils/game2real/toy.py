"""Deterministic Toy Game answer extraction for Game2Real-Bench.

The scorer recovers an explicit final commitment from model output, then
matches it against the canonical answer and the released accepted-answer
structures. It never calls a model judge or a network service.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Mapping, Optional


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _accepted_answers(value: Any) -> list[Any]:
    if _is_missing(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else [parsed]


def normalize_atom(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).casefold(): normalize_atom(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize_atom(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("／", "/").replace("＼", "\\")
    number = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if number:
        parsed = float(text.replace(",", ""))
        return int(parsed) if parsed.is_integer() else parsed
    return re.sub(r"\s+", " ", text).rstrip(".")


def strip_reasoning_and_wrappers(prediction: Any) -> tuple[str, str]:
    text = "" if _is_missing(prediction) else str(prediction).strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()
    tags = re.findall(r"(?is)<(?:answer|ans)>\s*(.*?)\s*</(?:answer|ans)>", text)
    if tags:
        return tags[-1].strip(), "answer_tag"
    boxed = re.findall(r"(?is)\\boxed\s*\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1].strip(), "boxed_answer"
    markers = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+(?:answer|response)|answer|result)\s*[:：]\s*",
            text,
        )
    )
    if markers:
        return text[markers[-1].end() :].strip(), "explicit_final_marker"
    fenced = list(re.finditer(r"(?is)```(?:json|text|python)?\s*(.*?)\s*```", text))
    if fenced and not text[fenced[-1].end() :].strip():
        return fenced[-1].group(1).strip(), "terminal_code_fence"
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json|text|python)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        return text.strip(), "code_fence"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(text) > 500 and lines:
        tail = lines[-1].strip("`*_ ")
        if len(tail) <= 1000 and (
            re.fullmatch(r"[A-F]", tail, flags=re.I)
            or "=" in tail
            or (tail.startswith(("[", "{")) and tail.endswith(("]", "}")))
        ):
            return tail, "terminal_line_after_reasoning"
    return text, "complete_response"


def best_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    decoder = json.JSONDecoder()
    values: list[tuple[int, int, Any]] = []
    for position, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, length = decoder.raw_decode(text[position:])
            values.append((length, position, value))
        except json.JSONDecodeError:
            continue
    return max(values, key=lambda item: (item[0], item[1]))[2] if values else None


def parse_value(raw: str) -> Any:
    value = raw.strip().strip("`*_ ").rstrip(".")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.replace(",", "")):
        number = float(value.replace(",", ""))
        return int(number) if number.is_integer() else number
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    return value.strip("\"'")


def parse_expected_shape(text: str, expected: Any) -> Optional[Any]:
    structured = best_json(text)
    if structured is not None:
        return structured
    if isinstance(expected, dict):
        parsed: dict[str, Any] = {}
        for key in expected:
            matches = re.findall(
                rf"(?i)(?<![\w]){re.escape(str(key))}\s*(?:=|:)\s*([^;\n,]+)",
                text,
            )
            if matches:
                parsed[str(key)] = parse_value(matches[-1])
        return parsed if set(parsed) == {str(key) for key in expected} else None
    if isinstance(expected, list):
        compact = text.strip().strip("[](){}")
        if not compact:
            return []
        if all(isinstance(item, str) and len(item) == 1 for item in expected):
            tokens = re.findall(r"(?i)(?<![A-Z])([UDLRABCDEF])(?![A-Z])", compact)
            return [token.upper() for token in tokens] if tokens else None
        parts = [part.strip(" \t\r\n\"'") for part in re.split(r"[,;]", compact)]
        return [part for part in parts if part]
    return None


def normalize_string_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("／", "/").replace("＼", "\\")
    text = re.sub(r"\s*([=;,:])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".")


def extract_mcq_commitment(text: str) -> str:
    text = re.sub(r"[`*_]", "", text)
    patterns = [
        r"(?i)(?:final\s+answer|answer|choice|option)\s*(?:is|=|:|：)?\s*\(?([A-F])\)?\b",
        r"(?i)\btherefore\s*,?\s*(?:the\s+answer\s+is\s*)?\(?([A-F])\)?\b",
    ]
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            return found[-1].upper()
    lines = [line.strip("`*_ .()[]{}") for line in text.splitlines() if line.strip()]
    if lines and re.fullmatch(r"[A-F]", lines[-1], flags=re.I):
        return lines[-1].upper()
    if lines:
        labelled = re.match(r"(?i)^([A-F])[.):]\s+\S", lines[-1])
        if labelled:
            return labelled.group(1).upper()
        terminal_phrase = re.search(
            r"(?i)\b(?:answer|choice|option|operation|location)\b[^\n]{0,40}\bis\s+([A-F])\b",
            lines[-1],
        )
        if terminal_phrase:
            return terminal_phrase.group(1).upper()
    compact = re.sub(r"[`*_\s.()\[\]{}]", "", text).upper()
    return compact if re.fullmatch(r"[A-F]", compact) else ""


def score_toy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("prediction", "")
    commitment, extraction_reason = strip_reasoning_and_wrappers(raw)
    answer = row.get("answer", row.get("GT", ""))
    accepted = _accepted_answers(row.get("accepted_answers", "[]"))
    expected_values = [answer, *accepted]
    question_type = str(row.get("question_type", "")).strip().casefold()

    if question_type == "single_choice":
        raw_text = str(raw)
        incomplete_think = "<think>" in raw_text.casefold() and "</think>" not in raw_text.casefold()
        if incomplete_think and extraction_reason == "complete_response":
            extracted: Any = ""
        else:
            extracted = extract_mcq_commitment(commitment)
        expected_labels = {
            label
            for value in expected_values
            if (label := extract_mcq_commitment(str(value)))
        }
        correct = bool(extracted) and extracted in expected_labels
        evidence = "option_label"
        mode = "single_choice"
    else:
        extracted = commitment
        correct = False
        evidence = "none"
        mode = "structured_open"
        normalized_candidate = normalize_string_answer(commitment)
        for expected in expected_values:
            if isinstance(expected, str) and normalized_candidate == normalize_string_answer(expected):
                correct, evidence = True, "canonical_surface_answer"
                break
        if not correct:
            for expected in accepted:
                parsed = parse_expected_shape(commitment, expected)
                if parsed is not None and normalize_atom(parsed) == normalize_atom(expected):
                    extracted, correct, evidence = parsed, True, "accepted_answer_structure"
                    break

    return {
        "hit": int(correct),
        "prediction": extracted if isinstance(extracted, str) else json.dumps(extracted, ensure_ascii=False),
        "gt": str(answer),
        "mode": mode,
        "source": extraction_reason,
        "status": "matched" if correct else ("unparseable" if extracted in ("", None) else "wrong"),
        "candidate": commitment,
        "match_evidence": evidence,
    }


def evaluate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one released Toy Game record."""
    return score_toy_row(row)
