# Data Format

Game2Real-Bench data are hosted separately at [`liuqa/Game2Real-Bench`](https://huggingface.co/datasets/liuqa/Game2Real-Bench). The repository contains two tab-separated tables:

- `data/toy_game.tsv`: 1,000 Toy Game samples;
- `data/real_world.tsv`: 400 Real World samples.

## Common Fields

| Field | Description |
|---|---|
| `id` | Stable sample identifier, such as `toy-0001` or `real-0001`. |
| `index` | One-based numeric index within the partition. |
| `partition` | `toy_game` or `real_world`. |
| `question` | Model-facing question text. |
| `hint` | Optional task-specific instruction. |
| `A`-`F` | Optional answer choices. Empty for open-answer samples. |
| `answer` | Canonical gold answer. |
| `accepted_answers` | JSON-encoded list of accepted gold surfaces. |
| `evaluator_payload` | JSON-encoded task-specific evaluator metadata. |
| `question_type` | `single_choice` or `open`. |
| `difficulty` | Released difficulty label. |
| `category` | One of the four benchmark capability groups. |
| `subfield` | Toy-game task or aligned real-world subfield. |
| `image` | First image path, retained for single-image compatibility. |
| `image_path` | JSON-encoded ordered list of repository-relative image paths. |
| `question_image_roles` | Optional JSON-encoded roles for multi-image inputs. |
| `question_image_note` | Optional note describing image ordering or usage. |

All image paths use POSIX separators and are relative to the dataset repository root. Always preserve the order stored in `image_path` when constructing a multimodal prompt.

## Toy Game

Toy Game contains 18 tasks and 65 QA families. Its 1,000 samples include:

- 400 single-choice questions;
- 600 open-answer questions.

Open-answer normalization is selected through `evaluator_payload` and the released evaluator resource. The public API performs exact, deterministic normalization; it does not call an external language model.

## Real World

Real World adds source, alignment, verification, and licensing fields used to document how each sample was selected or constructed. The most important evaluation fields are:

| Field | Description |
|---|---|
| `evaluator_type` | Deterministic scoring rule for the sample. |
| `answer_normalization_version` | Version of the normalization contract. |
| `primary_toy_task` | Primary Toy Game task aligned to the sample. |
| `mapped_toy_tasks` | JSON-encoded list of aligned Toy Game tasks. |
| `input_format` / `num_images` | Input modality and expected image count. |
| `mechanism_alignment` | Text explanation of the shared reasoning mechanism. |
| `source` / `source_subset` / `source_split` | Upstream source record. |
| `source_license` / `redistribution_status` | Source-specific reuse information. |

The complete source inventory and attribution records are included in the dataset repository under `manifests/`, `LICENSE_DATA.md`, and `ATTRIBUTION.md`.

## Loading Records

```python
from game2real import load_partition

toy_game = load_partition("./Game2Real-Bench-data", "toy_game")
real_world = load_partition("./Game2Real-Bench-data", "real_world")

sample = toy_game[0]
print(sample["id"], sample["question"], sample["image_path"])
```

The loader parses JSON-valued cells and resolves image references without changing the stored question or answer content.

