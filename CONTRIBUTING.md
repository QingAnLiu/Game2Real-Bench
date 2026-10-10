# Contributing

Thank you for helping improve Game2Real-Bench.

## Before Opening a Pull Request

1. Create a focused branch and keep unrelated changes separate.
2. Install development dependencies with `python -m pip install -e ".[test]"`.
3. Run `python -m pytest -q`.
4. Update the root `README.md` when a change affects public usage, data fields, or evaluation behavior.

## Evaluation Changes

Scoring behavior is part of the benchmark contract. Changes must include tests covering accepted answers, rejected answers, normalization behavior, and malformed predictions. Avoid judge-model dependencies unless a future benchmark version explicitly adopts them.

## Data Issues

Do not submit upstream dataset images directly through a pull request. Report sample IDs and the issue type so the maintainers can review licensing, attribution, annotation, or integrity concerns in the companion dataset repository.

