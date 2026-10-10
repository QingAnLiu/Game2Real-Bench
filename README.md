<div align="center">

# Game2Real-Bench

### Do Toy Games Reveal and Transfer to Real-World Visual Reasoning?

**A capability-aligned benchmark connecting controlled visual games with real-world visual reasoning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/QingAnLiu/Game2Real-Bench/actions/workflows/ci.yml/badge.svg)](https://github.com/QingAnLiu/Game2Real-Bench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/Code%20License-MIT-2ea44f)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Private%20Preview-f3c742)](https://huggingface.co/datasets/liuqa/Game2Real-Bench)

[Dataset](https://huggingface.co/datasets/liuqa/Game2Real-Bench) · [Quick Start](#quick-start) · [Evaluation](docs/EVALUATION.md) · [Data Format](docs/DATA_FORMAT.md) · [Citation](#citation)

</div>

## Overview

Game2Real-Bench studies whether the visual reasoning capabilities measured by controlled toy games correspond to those required by aligned real-world tasks. Rather than matching scenes by appearance, the benchmark connects the two settings through shared reasoning requirements: visual evidence, spatial operations, constraints, and state transitions.

The benchmark contains **1,400 evaluation samples** organized around four capabilities: **Fine-grained Perception**, **Spatial Reasoning**, **Constraint Reasoning**, and **Sequential Planning**. Toy Game provides controllable generation and programmatically verified answers, while Real World preserves the diversity and ambiguity of natural visual settings.

<p align="center">
  <img src="assets/figures/construction_pipeline.png" width="100%" alt="Game2Real-Bench construction pipeline">
</p>

<p align="center"><em>Game2Real-Bench is constructed through capability-guided task design, procedural generation and verification, and mechanism-aligned real-world curation.</em></p>

## Benchmark at a Glance

| Capability | Toy Game | Real World | Total |
|---|---:|---:|---:|
| Fine-grained Perception | 178 | 100 | 278 |
| Spatial Reasoning | 222 | 100 | 322 |
| Constraint Reasoning | 305 | 100 | 405 |
| Sequential Planning | 295 | 100 | 395 |
| **Total** | **1,000** | **400** | **1,400** |

- **Toy Game:** 18 tasks and 65 QA families, including 400 single-choice and 600 open-answer samples.
- **Real World:** 400 capability-aligned samples using seven deterministic evaluator types.
- **Images:** single-image and multi-image inputs are supported; image order is preserved by the dataset loader.
- **Evaluation:** no external judge model is required for the released scoring pipeline.

## Capability-aligned Design

<p align="center">
  <img src="assets/figures/capability_examples.png" width="100%" alt="Capability-aligned Toy Game and Real World examples">
</p>

The Toy Game and Real World partitions share a capability taxonomy while retaining their distinct visual domains. This enables evaluation at three levels: overall performance, capability-level performance, and task/subfield-level behavior.

## Repository Structure

```text
.
├── assets/figures/              # README figures derived from the paper
├── docs/                        # data-format and evaluation references
├── examples/                    # example prediction file
├── game2real/                   # portable loader and deterministic scorers
├── VLMEvalKit/                  # bundled VLMEvalKit fork used by the benchmark
│   ├── run.py                   # standard VLMEvalKit entry point
│   ├── utils/
│   │   └── prepare_game2real_data.py
│   └── vlmeval/dataset/         # Game2Real dataset loaders and evaluators
├── scripts/
│   ├── download_data.py         # download the Hugging Face snapshot
│   ├── build_dataset_release.py # rebuild a sanitized dataset release
│   ├── run_evaluation.py        # score model predictions
│   ├── validate_release.py      # validate code and data before release
│   └── upload_code.ps1          # validate and publish the code repository
├── tests/                       # unit tests for loading and scoring
└── vlmevalkit_adapter/          # lightweight compatibility utilities
```

This repository includes the VLMEvalKit fork used for benchmark inference and evaluation, together with the public-facing loaders, deterministic scorers, release builder, and validation utilities. Procedural game-generator source is maintained separately while its public-release scope is being decided.

## Quick Start

### 1. Install

```bash
git clone https://github.com/QingAnLiu/Game2Real-Bench.git
cd Game2Real-Bench
python -m pip install -e ".[hub,test]"
python -m pip install -e ./VLMEvalKit
```

### 2. Authenticate and download the dataset

The dataset is currently a private preview. Use a Hugging Face account with repository access:

```bash
hf auth login
python scripts/download_data.py \
  --repo-id liuqa/Game2Real-Bench \
  --output-dir ./Game2Real-Bench-data
```

### 3. Prepare predictions

Prediction files may be JSON, JSONL, CSV, or TSV. Each row must contain `id` (or `index`) and `prediction`:

```json
{"id": "toy-0001", "prediction": "D"}
{"id": "toy-0002", "prediction": "C"}
```

### 4. Run deterministic evaluation

```bash
python scripts/run_evaluation.py \
  --dataset-root ./Game2Real-Bench-data \
  --partition toy_game \
  --predictions ./predictions.jsonl \
  --output-dir ./results/toy_game
```

The evaluator writes:

- `scores.json`: aggregate scores overall and by benchmark taxonomy;
- `detailed_results.jsonl`: normalized answers and per-sample correctness.

See [Evaluation](docs/EVALUATION.md) for supported answer formats and [Data Format](docs/DATA_FORMAT.md) for the released schema.

## VLMEvalKit Integration

Prepare both benchmark partitions under `$LMUData` while preserving questions, answers, and image order:

```bash
python VLMEvalKit/utils/prepare_game2real_data.py \
  --dataset-root ./Game2Real-Bench-data \
  --partition all
```

Run inference through the bundled VLMEvalKit using the registered dataset names:

```bash
python VLMEvalKit/run.py \
  --data Game2Real_ToyGame Game2Real_RealWorld \
  --model <VLMEvalKit_MODEL_NAME>
```

`Game2Real_ToyGame` uses the released Game2Real deterministic scorer, and `Game2Real_RealWorld` uses the seven deterministic evaluator families implemented in `real_world_final.py`. No API credential is stored in `VLMEvalKit/vlmeval/config.py`; configure model-provider credentials through environment variables when needed.

See [the bundled integration guide](VLMEvalKit/GAME2REAL.md) for data preparation, registered names, and evaluation behavior.

## Validation and Tests

```bash
python -m pytest -q

python scripts/validate_release.py \
  --dataset-root ../dataset \
  --code-root .
```

The release validator checks sample counts, image references, sample-ID naming, capability directories, portable paths, evaluator resources, and repository hygiene.

## Data and Licensing

Project-authored software in this repository is released under the [MIT License](LICENSE). Dataset files are distributed separately and contain source material under mixed upstream terms; the software license does not apply to dataset images. Review the dataset repository's `LICENSE_DATA.md`, `ATTRIBUTION.md`, and source manifest before reuse or redistribution.

See [Third-Party Licenses](THIRD_PARTY_LICENSES.md) for software-side notices.

## Citation

The manuscript is currently being prepared for public release. If you use this private preview, please cite:

> Qing'an Liu, Juntong Feng, Yue Zhu, Yujie Cheng, Xinzhe Han, Shuai Hao, Yunzhi Zhuge, Haiwen Diao, and Huchuan Lu. *Game2Real-Bench: Do Toy Games Reveal and Transfer to Real-World Visual Reasoning?* 2026.

A final BibTeX entry will be added when the manuscript is publicly available. Repository citation metadata are also provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

Game2Real-Bench builds on the open-source multimodal evaluation ecosystem. We thank the maintainers of VLMEvalKit and the upstream datasets documented in the companion dataset repository.

