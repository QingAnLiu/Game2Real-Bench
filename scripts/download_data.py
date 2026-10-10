#!/usr/bin/env python3
"""Download a complete Game2Real-Bench dataset snapshot from Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="liuqa/Game2Real-Bench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit('Install the Hub dependency with: python -m pip install ".[hub]"') from exc
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
