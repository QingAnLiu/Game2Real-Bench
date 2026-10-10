# Evaluation

Game2Real-Bench uses deterministic, task-aware scoring. No external judge model is required.

## Prediction Files

Prediction files may be `.json`, `.jsonl`, `.csv`, or `.tsv`.

- JSON files must contain a list of objects.
- JSONL files contain one object per line.
- CSV and TSV files must include a header row.

Every prediction object must contain:

- `id` or `index`, identifying a released sample;
- `prediction`, containing the model output.

Duplicate identifiers, unknown samples, and missing `prediction` fields are rejected. Evaluation is performed over the supplied rows, so official full-partition reporting should include all 1,000 Toy Game samples or all 400 Real World samples.

## Running the Evaluator

```bash
python scripts/run_evaluation.py \
  --dataset-root ./Game2Real-Bench-data \
  --partition real_world \
  --predictions ./predictions/real_world.jsonl \
  --output-dir ./results/real_world
```

The installed console entry point provides the same interface:

```bash
game2real-evaluate \
  --dataset-root ./Game2Real-Bench-data \
  --partition toy_game \
  --predictions ./predictions/toy_game.jsonl \
  --output-dir ./results/toy_game
```

## Toy Game Scoring

- `single_choice`: extracts the final option letter from `A`-`F` and compares it with the gold answer.
- `open`: removes common answer wrappers, normalizes the expected task surface, and compares against all released accepted answers.

Open-answer schemas may normalize structured key-value surfaces, JSON objects/lists, coordinates, directions, numeric values, whitespace, punctuation, and case as defined by the released evaluator metadata.

## Real World Scoring

The 400-sample release uses seven evaluator types:

| Evaluator | Samples | Behavior |
|---|---:|---|
| `mcq_single` | 236 | Extract and compare a final option letter. |
| `exact_short_text` | 12 | Unicode, case, whitespace, and terminal-period normalization. |
| `numeric` | 30 | Extract and normalize a numeric answer. |
| `json_canonical` | 50 | Canonicalize nested JSON, coordinates, directions, and numeric values. |
| `json_unordered_set` | 8 | Compare normalized list members without order. |
| `ranking` | 50 | Compare a normalized ordered ranking. |
| `route_plan` | 14 | Normalize and compare structured route sections and stops. |

## Outputs

`scores.json` contains aggregate metrics produced by the scoring summary. `detailed_results.jsonl` contains one record per prediction with:

- sample identity and taxonomy;
- raw prediction;
- extracted/normalized answer surface;
- evaluator name;
- correctness and diagnostic detail.

Keep `detailed_results.jsonl` when investigating prompt-format or answer-format errors.

## Reproducibility Notes

- Preserve image order for multi-image samples.
- Do not rewrite questions or options before evaluation.
- Report the evaluated partition and number of predictions.
- Record the dataset revision used for each run.
- Use the released deterministic evaluator rather than manual or model-judge scoring for comparable results.

