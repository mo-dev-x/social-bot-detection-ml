"""Compatibility entry point for training and inference helpers."""

from src.model_training import (
    analyze_feature_importance,
    compute_competition_score,
    find_best_threshold,
    run_final_detection,
    save_submission,
    train_full_pipeline,
    train_lightgbm_model,
    validate_model,
)
from src.rules_engine import rules_engine

__all__ = [
    "analyze_feature_importance",
    "compute_competition_score",
    "find_best_threshold",
    "rules_engine",
    "run_final_detection",
    "save_submission",
    "train_full_pipeline",
    "train_lightgbm_model",
    "validate_model",
]


if __name__ == "__main__":
    train_full_pipeline(
        dataset_path="data/train_en/dataset.posts&users.json",
        ground_truth_path="data/train_en/dataset.bots.txt",
        language="en",
    )
