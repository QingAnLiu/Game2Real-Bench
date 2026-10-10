#!/usr/bin/env python3
"""Score Game2Real-Bench prediction files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from game2real.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--partition", choices=("toy_game", "real_world"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scores = run_evaluation(args.dataset_root, args.partition, args.predictions, args.output_dir)
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
