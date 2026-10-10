from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "VLMEvalKit" / "vlmeval" / "dataset"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_adapter():
    for name in ("vlmeval", "vlmeval.dataset", "vlmeval.dataset.utils", "vlmeval.dataset.utils.game2real"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    image_mixedqa = types.ModuleType("vlmeval.dataset.image_mixedqa")
    image_mixedqa.ImageMixedQADataset = type("ImageMixedQADataset", (), {})
    sys.modules[image_mixedqa.__name__] = image_mixedqa
    real_world = types.ModuleType("vlmeval.dataset.real_world_final")
    real_world.RealWorldFinalDataset = type("RealWorldFinalDataset", (), {})
    sys.modules[real_world.__name__] = real_world

    smp = types.ModuleType("vlmeval.smp")
    smp.dump = lambda *_args, **_kwargs: None
    smp.load = lambda *_args, **_kwargs: None
    smp.get_intermediate_file_path = lambda path, suffix, extension=None: (
        f"{path}{suffix}.{extension or 'xlsx'}"
    )
    sys.modules[smp.__name__] = smp

    utility_root = DATASET_ROOT / "utils" / "game2real"
    load_module("vlmeval.dataset.utils.game2real.toy", utility_root / "toy.py")
    load_module("vlmeval.dataset.utils.game2real.real", utility_root / "real.py")
    return load_module("vlmeval.dataset.game2real_bench", DATASET_ROOT / "game2real_bench.py")


def test_toy_adapter_scores_predictions_and_emits_capability_summary() -> None:
    adapter = load_adapter()
    dataset = types.SimpleNamespace(
        data=pd.DataFrame(
            [
                {
                    "id": "toy-0001",
                    "index": 1,
                    "answer": "D",
                    "accepted_answers": "[]",
                    "question_type": "single_choice",
                    "category": "Fine-grained Perception",
                    "subfield": "demo",
                },
                {
                    "id": "toy-0002",
                    "index": 2,
                    "answer": "x=1; y=2",
                    "accepted_answers": '[{"x":1,"y":2}]',
                    "question_type": "open",
                    "category": "Constraint Reasoning",
                    "subfield": "demo",
                },
            ]
        )
    )
    adapter.load = lambda _path: pd.DataFrame(
        [
            {"id": "toy-0001", "prediction": "Answer: D"},
            {"id": "toy-0002", "prediction": 'Final answer: {"y":2,"x":1}'},
        ]
    )
    written = {}
    adapter.dump = lambda value, path: written.setdefault(path, value)

    scores = adapter._evaluate_toy(dataset, "predictions.xlsx")

    assert float(scores.loc[scores["scope"] == "overall", "score"].iloc[0]) == 1.0
    assert not bool(scores["official_partition"].iloc[0])
    detail = next(value for path, value in written.items() if "deterministic_result" in path)
    assert detail["hit"].tolist() == [1, 1]


def test_real_world_adapter_uses_task_aware_extraction() -> None:
    adapter = load_adapter()
    dataset = types.SimpleNamespace(
        data=pd.DataFrame(
            [
                {
                    "id": "real-0001",
                    "index": 1,
                    "answer": "1200",
                    "accepted_answers": '["1200"]',
                    "question_type": "open",
                    "evaluator_type": "numeric",
                    "category": "Fine-grained Perception",
                    "subfield": "demo",
                    "source": "demo",
                }
            ]
        )
    )
    adapter.load = lambda _path: pd.DataFrame([{"id": "real-0001", "prediction": "1,200"}])
    adapter.dump = lambda *_args, **_kwargs: None

    scores = adapter._evaluate_real_world(dataset, "predictions.xlsx")

    assert float(scores.loc[scores["scope"] == "overall", "score"].iloc[0]) == 1.0
