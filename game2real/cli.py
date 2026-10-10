"""Command-line entry point for deterministic evaluation."""
from __future__ import annotations

import argparse

from .runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--partition", choices=("toy_game", "real_world"), required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_evaluation(args.dataset_root, args.partition, args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
