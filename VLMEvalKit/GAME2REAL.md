# Game2Real-Bench in VLMEvalKit

This directory is the VLMEvalKit fork used for Game2Real-Bench inference. The benchmark adds two public dataset registrations:

| Dataset name | Partition | Evaluation |
|---|---|---|
| `Game2Real_ToyGame` | 1,000 Toy Game samples | released Game2Real deterministic scorer |
| `Game2Real_RealWorld` | 400 Real World samples | deterministic evaluator suite in `real_world_final.py` |

## Setup

From the Game2Real-Bench repository root:

```bash
python -m pip install -e .
python -m pip install -e ./VLMEvalKit
python scripts/download_data.py \
  --repo-id liuqa/Game2Real-Bench \
  --output-dir ./Game2Real-Bench-data
python VLMEvalKit/utils/prepare_game2real_data.py \
  --dataset-root ./Game2Real-Bench-data \
  --partition all
```

The preparation utility validates every image reference and writes `Game2Real_ToyGame.tsv` and `Game2Real_RealWorld.tsv` under `$LMUData` (or `~/LMUData` when the environment variable is unset). Images are not duplicated.

## Run

```bash
python VLMEvalKit/run.py \
  --data Game2Real_ToyGame Game2Real_RealWorld \
  --model <VLMEvalKit_MODEL_NAME>
```

Model-provider credentials must be supplied through environment variables. The checked-in `vlmeval/config.py` intentionally contains no API key or token.

For scoring an existing prediction file independently of VLMEvalKit, use `../scripts/run_evaluation.py` from the repository root.
