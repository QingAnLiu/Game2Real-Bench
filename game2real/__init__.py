"""Game2Real-Bench public evaluation utilities."""

from .dataset import load_partition, resolve_image_paths
from .evaluator import EvaluationResult, evaluate_record
from .scoring import summarize_results

__all__ = [
    "EvaluationResult",
    "evaluate_record",
    "load_partition",
    "resolve_image_paths",
    "summarize_results",
]

__version__ = "1.0.0"
