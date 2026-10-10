from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_vlmevalkit_registers_game2real() -> None:
    registry = (ROOT / "VLMEvalKit" / "vlmeval" / "dataset" / "__init__.py").read_text(encoding="utf-8")
    integration = (ROOT / "VLMEvalKit" / "vlmeval" / "dataset" / "game2real_bench.py").read_text(
        encoding="utf-8"
    )
    assert "Game2RealToyDataset" in registry
    assert "Game2RealRealWorldDataset" in registry
    assert 'TOY_DATASET = "Game2Real_ToyGame"' in integration
    assert 'REAL_WORLD_DATASET = "Game2Real_RealWorld"' in integration
    assert "from .utils.game2real import real as real_evaluator" in integration
    assert "from .utils.game2real import toy as toy_evaluator" in integration
    assert "game2real.runner" not in integration


def test_vlmevalkit_config_contains_no_literal_credentials() -> None:
    path = ROOT / "VLMEvalKit" / "vlmeval" / "config.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text)
    assert "os.environ.get('O1_API_KEY', '')" in text
    assert not re.search(r"(?i)(?:sk-|hf_)[A-Za-z0-9_-]{16,}", text)


def test_release_has_only_vlmevalkit_evaluation_entrypoint() -> None:
    assert not (ROOT / "scripts" / "run_evaluation.py").exists()
    assert not (ROOT / "game2real").exists()
    assert (ROOT / "VLMEvalKit" / "run.py").is_file()
