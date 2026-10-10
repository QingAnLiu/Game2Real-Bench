# VLMEvalKit adapter

`prepare_vlmevalkit_tsv.py` converts the public repository-relative image paths into absolute local paths without changing sample order or benchmark content.

Use the generated TSV as a custom local dataset in a compatible VLMEvalKit checkout. Game2Real-Bench's companion evaluator should be used for the final Real World deterministic scoring and for Toy Game open-answer semantic normalization.
