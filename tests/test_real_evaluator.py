from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "VLMEvalKit" / "vlmeval" / "dataset" / "utils" / "game2real" / "real.py"
SPEC = importlib.util.spec_from_file_location("game2real_real_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
real = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real)


def score(prediction: str, evaluator_type: str, answer, accepted=None, **extra) -> bool:
    record = {
        "evaluator_type": evaluator_type,
        "answer": answer,
        "accepted_answers": json.dumps(accepted if accepted is not None else [answer]),
        **extra,
    }
    extracted, _ = real.blind_extract(prediction, record)
    return bool(real.score_non_spatialqa(extracted, record))


def test_mcq_numeric_and_short_text() -> None:
    assert score("The final answer is C.", "mcq_single", "C", A="x", B="y", C="z", D="w")
    assert score("1,200", "numeric", "1200")
    assert score("YES.", "exact_short_text", "yes")


def test_set_ranking_and_route_evaluators() -> None:
    assert score('["b", "a"]', "json_unordered_set", [], [["A", "B"]])
    assert score('["a", "b", "c"]', "ranking", [], [["A", "B", "C"]])
    route = [
        {
            "route_name": "Blue",
            "departure_stop": "East Station",
            "arrival_stop": "West Station",
            "num_via_stops": 2,
            "via_stops": ["A", "B"],
        }
    ]
    assert score(json.dumps(route), "route_plan", route, [route])


def test_spatial_graph_exact_replay_scores_one() -> None:
    answer = {
        "step_1": {"content": "pick up the red block", "precondition": []},
        "step_2": {"content": "place the red block on the tray", "precondition": ["step_1"]},
    }
    gt_steps, gt_warnings = real.parse_spatialqa_steps(answer)
    pred_steps, pred_warnings = real.parse_spatialqa_steps(json.dumps(answer))
    assert not gt_warnings
    assert not pred_warnings
    matrix, _, applied = real.build_public_match_matrix("real-0001", gt_steps, pred_steps, rules=[])
    metrics = real.spatialqa_metrics(gt_steps, pred_steps, matrix)
    assert not applied
    assert metrics["content_f1"] == 1.0
    assert metrics["precondition_f1"] == 1.0
    assert metrics["score"] == 1.0


def test_reviewed_override_resource_is_well_formed() -> None:
    rules = real.load_content_overrides()
    assert rules
    assert all(len(rule["prediction_sha256"]) == 64 for rule in rules)
