from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "VLMEvalKit" / "vlmeval" / "dataset" / "utils" / "game2real" / "toy.py"
SPEC = importlib.util.spec_from_file_location("game2real_toy_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
toy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(toy)


def test_single_choice_uses_explicit_final_answer() -> None:
    row = {
        "question_type": "single_choice",
        "answer": "D",
        "accepted_answers": "[]",
        "prediction": "<think>I considered A and B.</think>\n<answer>D</answer>",
    }
    result = toy.evaluate_row(row)
    assert result["hit"] == 1
    assert result["prediction"] == "D"
    assert result["source"] == "answer_tag"


def test_unclosed_reasoning_trace_is_not_an_answer() -> None:
    row = {
        "question_type": "single_choice",
        "answer": "B",
        "accepted_answers": "[]",
        "prediction": "<think>The candidates are A and B",
    }
    result = toy.evaluate_row(row)
    assert result["hit"] == 0
    assert result["status"] == "unparseable"


def test_open_answer_accepts_reordered_typed_structure() -> None:
    row = {
        "question_type": "open",
        "answer": "R1C1=1; R2C2=2",
        "accepted_answers": '[{"R1C1": 1, "R2C2": 2}]',
        "prediction": 'Final answer: {"R2C2": 2.0, "R1C1": 1}',
    }
    result = toy.evaluate_row(row)
    assert result["hit"] == 1
    assert result["match_evidence"] == "accepted_answer_structure"


def test_reasoning_that_only_mentions_correct_surface_is_wrong() -> None:
    row = {
        "question_type": "open",
        "answer": "Top=1; Bottom=2",
        "accepted_answers": "[]",
        "prediction": "One possibility is Top=1; Bottom=2, but I am not sure.",
    }
    assert toy.evaluate_row(row)["hit"] == 0
