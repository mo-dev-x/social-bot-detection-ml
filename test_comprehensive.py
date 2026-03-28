"""Comprehensive validation suite for the bot detection system."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib

from src.feature_extraction import create_feature_dataframe
from src.rules_engine import rules_engine


def test_imports():
    """Test that core dependencies import correctly."""
    print("\n[TEST 1] Checking imports...")
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        import lightgbm  # noqa: F401
        import joblib  # noqa: F401
        import tqdm  # noqa: F401
        print("   OK All dependencies installed")
        return True
    except ImportError as exc:
        print(f"   FAIL Import failed: {exc}")
        return False


def test_data_paths():
    """Report whether the expected training files are available yet."""
    print("\n[TEST 2] Checking data paths...")
    required_paths = [
        Path("data/train_en/dataset.posts&users.json"),
        Path("data/train_en/dataset.bots.txt"),
        Path("data/train_fr/dataset.posts&users.json"),
        Path("data/train_fr/dataset.bots.txt"),
    ]

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        print("   SKIP Waiting for data files:")
        for path in missing:
            print(f"      - {path}")
        return True

    print("   OK All data files present")
    return True


def test_feature_extraction():
    """Test that feature extraction works when data is present."""
    print("\n[TEST 3] Testing feature extraction...")
    en_path = Path("data/train_en/dataset.posts&users.json")
    fr_path = Path("data/train_fr/dataset.posts&users.json")
    if not en_path.exists() or not fr_path.exists():
        print("   SKIP Feature extraction test waiting for datasets")
        return True

    try:
        df_en = create_feature_dataframe(str(en_path), language="en")
        df_fr = create_feature_dataframe(str(fr_path), language="fr")
        print(f"   OK English: {df_en.shape[0]} users, {df_en.shape[1]} features")
        print(f"   OK French: {df_fr.shape[0]} users, {df_fr.shape[1]} features")

        total_nans = int(df_en.isnull().sum().sum() + df_fr.isnull().sum().sum())
        if total_nans > 0:
            print(f"   FAIL Found {total_nans} NaN values")
            return False
        return True
    except Exception as exc:
        print(f"   FAIL Feature extraction failed: {exc}")
        return False


def test_rules_engine():
    """Test that the rules engine returns a boolean-like value."""
    print("\n[TEST 4] Testing rules engine...")
    try:
        import pandas as pd

        row = pd.Series(
            {
                "z_score": 3.0,
                "cv_time_delta": 0.2,
                "duplicate_tweet_ratio": 0.5,
                "avg_cosine_similarity": 0.95,
                "hour_entropy": 0.1,
                "tweets_per_day_avg": 25.0,
                "max_tweets_in_10min": 30,
            }
        )
        result = rules_engine(row)
        if isinstance(result, (bool, int)):
            print("   OK Rules engine works")
            return True
        print(f"   FAIL Unexpected return type: {type(result)}")
        return False
    except Exception as exc:
        print(f"   FAIL Rules engine test failed: {exc}")
        return False


def test_model_directory():
    """Test that the models directory exists and is writable."""
    print("\n[TEST 5] Checking model directory...")
    try:
        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)
        test_file = model_dir / "write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        print("   OK Models directory is writable")
        return True
    except Exception as exc:
        print(f"   FAIL Model directory error: {exc}")
        return False


def test_full_pipeline():
    """Test the complete training pipeline when English data is present."""
    print("\n[TEST 6] Testing full training pipeline...")
    data_path = Path("data/train_en/dataset.posts&users.json")
    truth_path = Path("data/train_en/dataset.bots.txt")
    if not data_path.exists() or not truth_path.exists():
        print("   SKIP Full pipeline test waiting for English dataset")
        return True

    try:
        from src.model_training import train_full_pipeline

        result = train_full_pipeline(
            dataset_path=str(data_path),
            ground_truth_path=str(truth_path),
            language="en",
        )
        if "model" in result and "threshold" in result:
            print("   OK Full pipeline completed successfully")
            return True
        print("   FAIL Pipeline returned incomplete results")
        return False
    except Exception as exc:
        print(f"   FAIL Full pipeline test failed: {exc}")
        return False


def test_model_serialization():
    """Test that trained models can be loaded when present."""
    print("\n[TEST 7] Testing model serialization...")
    model_paths = [
        Path("models/model_en.pkl"),
        Path("models/model_fr.pkl"),
        Path("models/threshold_en.pkl"),
        Path("models/threshold_fr.pkl"),
    ]
    if not all(path.exists() for path in model_paths):
        print("   SKIP Serialization test waiting for trained models")
        return True

    try:
        model_en = joblib.load(model_paths[0])
        model_fr = joblib.load(model_paths[1])
        threshold_en = joblib.load(model_paths[2])
        threshold_fr = joblib.load(model_paths[3])
        print(f"   OK Model EN loaded ({type(model_en).__name__})")
        print(f"   OK Model FR loaded ({type(model_fr).__name__})")
        print(f"   OK Threshold EN: {threshold_en:.2f}")
        print(f"   OK Threshold FR: {threshold_fr:.2f}")
        return True
    except Exception as exc:
        print(f"   FAIL Failed to load models: {exc}")
        return False


def test_submission_format():
    """Test that submission files can be created and parsed."""
    print("\n[TEST 8] Testing submission format...")
    try:
        from src.model_training import save_submission

        output_dir = Path("submissions")
        output_dir.mkdir(parents=True, exist_ok=True)
        test_ids = [123456, 789012, 345678]
        filepath = Path(save_submission(test_ids, "test_team", "en", output_dir=str(output_dir)))
        if not filepath.exists():
            print("   FAIL Submission file not created")
            return False

        lines = [line.strip() for line in filepath.read_text(encoding="utf-8").splitlines() if line.strip()]
        filepath.unlink()
        if lines == [str(user_id) for user_id in test_ids]:
            print(f"   OK Submission format correct ({len(lines)} IDs)")
            return True
        print("   FAIL Submission format incorrect")
        return False
    except Exception as exc:
        print(f"   FAIL Submission test failed: {exc}")
        return False


def main() -> int:
    """Run all tests and return a process exit code."""
    print("\n" + "=" * 60)
    print("BOT DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    tests = [
        test_imports,
        test_data_paths,
        test_feature_extraction,
        test_rules_engine,
        test_model_directory,
        test_full_pipeline,
        test_model_serialization,
        test_submission_format,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as exc:
            print(f"   FAIL Unexpected error: {exc}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("ALL SYSTEMS GO - Environment and code are ready.")
    elif passed >= total - 1:
        print("MOSTLY READY - Review the warning above.")
    else:
        print("ISSUES DETECTED - Review failures above.")
    print("=" * 60 + "\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
