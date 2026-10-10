# Bundled VLMEvalKit

This directory contains the VLMEvalKit fork used to run Game2Real-Bench. It preserves the standard VLMEvalKit entry point and model registry while adding the benchmark-specific dataset loaders and deterministic evaluators required for reproducible scoring.

For the complete upstream toolkit and documentation, visit [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Game2Real-Bench Additions

| Path | Purpose |
|---|---|
| `vlmeval/dataset/game2real_bench.py` | Toy Game dataset registration and scoring bridge |
| `vlmeval/dataset/real_world_final.py` | Real World dataset and seven evaluator families |
| `vlmeval/dataset/image_mixedqa.py` | Mixed single-choice/open-answer dataset support |
| `utils/prepare_game2real_data.py` | Validates and prepares both partitions under `$LMUData` |

## Usage

From the Game2Real-Bench repository root:

```bash
python -m pip install -e .
python -m pip install -e ./VLMEvalKit

python VLMEvalKit/utils/prepare_game2real_data.py \
  --dataset-root ./Game2Real-Bench-data \
  --partition all

python VLMEvalKit/run.py \
  --data Game2Real_ToyGame Game2Real_RealWorld \
  --model <VLMEvalKit_MODEL_NAME>
```

The registered datasets are `Game2Real_ToyGame` and `Game2Real_RealWorld`. Provider credentials, if required by the selected model, must be supplied through environment variables; no API key is stored in `vlmeval/config.py`.

See the [project README](../README.md#vlmevalkit-evaluation) for the complete Game2Real-Bench workflow.

## License and Attribution

VLMEvalKit retains its upstream Apache-2.0 license in [`LICENSE`](LICENSE). Game2Real-Bench project-authored integration code is distributed under the repository-level MIT license. See the repository's [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) for attribution details.
