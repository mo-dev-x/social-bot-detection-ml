"""Lightweight validation entry point for environment and data readiness."""

from pathlib import Path

from src.feature_extraction import create_feature_dataframe
from src.model_training import build_training_dataframe


def main():
    dataset_path = Path("data/training/dataset.posts&users.1.json")
    ground_truth_path = Path("data/training/dataset.bots.1.txt")

    print("Environment check: Python package imports succeeded.")

    missing_paths = [str(path) for path in (dataset_path, ground_truth_path) if not path.exists()]
    if missing_paths:
        print("Dataset check: waiting for training data.")
        print("Missing files:")
        for path in missing_paths:
            print(f"  - {path}")
        print("The codebase is ready; rerun this script after the dataset arrives.")
        return

    df = create_feature_dataframe(str(dataset_path), language="en")
    training_df, dataset_specs = build_training_dataframe()
    print(f"Feature extraction OK: {df.shape[0]} users, {df.shape[1]} columns")
    print(f"Training batches discovered: {len(dataset_specs)}")
    print(f"Combined training table: {training_df.shape[0]} users, {training_df.shape[1]} columns")
    print("Dataset is present and the project is ready for training.")


if __name__ == "__main__":
    main()
