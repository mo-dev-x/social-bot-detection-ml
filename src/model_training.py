"""Model training and inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .feature_extraction import create_feature_dataframe
from .rules_engine import rules_engine
from .utils import ensure_directory

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_competition_score(y_true, y_pred):
    """Official asymmetric competition score."""
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    score = 2 * tp - 2 * fn - 6 * fp
    return {
        "score": score,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": sum((y_true == 0) & (y_pred == 0)),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


def train_lightgbm_model(X_train, y_train, language: str = "en"):
    """Train the tabular classifier."""
    if language == "fr":
        n_estimators = 300
        max_depth = 8
    else:
        n_estimators = 250
        max_depth = 7

    logger.info("Training LightGBM model for %s", language.upper())
    model = lgb.LGBMClassifier(
        num_leaves=31,
        max_depth=max_depth,
        learning_rate=0.05,
        n_estimators=n_estimators,
        subsample=0.9,
        colsample_bytree=0.9,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(y_true, y_probs):
    """Search thresholds for the best competition score."""
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    results = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        score_dict = compute_competition_score(y_true, y_pred)
        results.append({"threshold": threshold, **score_dict})

    results = sorted(results, key=lambda item: item["score"], reverse=True)
    print("\n=== THRESHOLD TUNING RESULTS ===")
    print(f"{'Threshold':<10} {'Score':<10} {'Precision':<10} {'Recall':<10} {'TP':<6} {'FN':<6} {'FP':<6}")
    print("-" * 60)
    for result in results:
        print(
            f"{result['threshold']:<10.2f} {result['score']:<10.0f} "
            f"{result['precision']:<10.2f} {result['recall']:<10.2f} "
            f"{result['tp']:<6} {result['fn']:<6} {result['fp']:<6}"
        )

    best = results[0]
    print(f"\nBest threshold: {best['threshold']:.2f} (Score: {best['score']:.0f})")
    return best["threshold"]


def validate_model(model, X_val, y_val, threshold: float = 0.85):
    """Score the model on validation data."""
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    score_dict = compute_competition_score(y_val, preds)
    print("\n=== VALIDATION RESULTS ===")
    print(f"Decision Threshold: {threshold:.2f}")
    print(f"Competition Score: {score_dict['score']:.0f}")
    print(f"True Positives: {score_dict['tp']}")
    print(f"False Negatives: {score_dict['fn']}")
    print(f"False Positives: {score_dict['fp']}")
    print(f"True Negatives: {score_dict['tn']}")
    print(f"Precision: {score_dict['precision']:.2%}")
    print(f"Recall: {score_dict['recall']:.2%}")
    return score_dict


def analyze_feature_importance(model, X_train):
    """Print and return ranked feature importance."""
    feature_importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\n=== TOP 20 FEATURES ===")
    print(feature_importance_df.head(20))
    return feature_importance_df


def train_full_pipeline(dataset_path, ground_truth_path, language: str = "en", models_dir: str = "models"):
    """Train, tune, validate, and persist a language-specific model."""
    logger.info("Starting training pipeline for %s", language.upper())
    print(f"\n{'=' * 60}")
    print(f"TRAINING PIPELINE: {language.upper()}")
    print(f"{'=' * 60}")

    print(f"\n1. Loading dataset: {dataset_path}")
    features_df = create_feature_dataframe(dataset_path, language=language)
    logger.info("Feature dataframe shape for %s: %s", language.upper(), features_df.shape)
    print(f"   -> Extracted features for {len(features_df)} users")
    print(f"   -> {len(features_df.columns)} columns")

    print(f"\n2. Loading ground truth: {ground_truth_path}")
    with Path(ground_truth_path).open("r", encoding="utf-8") as handle:
        bot_ids = {int(line.strip()) for line in handle if line.strip()}

    features_df["label"] = features_df["user_id"].isin(bot_ids).astype(int)
    print(f"   -> {features_df['label'].sum()} positive labels")

    print(f"\n3. Splitting train/val (70/30)")
    X = features_df.drop(["user_id", "label"], axis=1)
    y = features_df["label"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   -> Train: {len(X_train)} ({y_train.sum()} bots)")
    print(f"   -> Val: {len(X_val)} ({y_val.sum()} bots)")

    print(f"\n4. Training LightGBM model")
    model = train_lightgbm_model(X_train, y_train, language=language)
    print("   -> Model trained")

    print(f"\n5. Feature importance")
    feature_importance = analyze_feature_importance(model, X_train)

    print(f"\n6. Finding optimal threshold")
    probs_val = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, probs_val)

    print(f"\n7. Validating model")
    val_results = validate_model(model, X_val, y_val, threshold=best_threshold)

    target_dir = ensure_directory(models_dir)
    model_path = target_dir / f"model_{language}.pkl"
    threshold_path = target_dir / f"threshold_{language}.pkl"

    print(f"\n8. Saving model")
    joblib.dump(model, model_path)
    print(f"   -> Model saved to {model_path}")

    print(f"\n9. Saving threshold")
    joblib.dump(best_threshold, threshold_path)
    print(f"   -> Threshold saved to {threshold_path}")

    return {
        "model": model,
        "threshold": best_threshold,
        "val_results": val_results,
        "feature_importance": feature_importance,
        "feature_names": X_train.columns.tolist(),
    }


def run_final_detection(dataset_path, language: str = "en", model_path=None, threshold_path=None):
    """Run inference on an unlabeled evaluation dataset."""
    model_path = model_path or f"models/model_{language}.pkl"
    threshold_path = threshold_path or f"models/threshold_{language}.pkl"
    logger.info("Running final detection for %s", language.upper())

    print(f"\n{'=' * 60}")
    print(f"FINAL DETECTION: {language.upper()}")
    print(f"{'=' * 60}")

    print(f"\n1. Loading dataset: {dataset_path}")
    features_df = create_feature_dataframe(dataset_path, language=language)
    logger.info("Inference feature dataframe shape for %s: %s", language.upper(), features_df.shape)
    print(f"   -> Features extracted for {len(features_df)} users")

    print(f"\n2. Loading model: {model_path}")
    model = joblib.load(model_path)
    print("   -> Model loaded")

    print(f"\n3. Loading threshold: {threshold_path}")
    threshold = joblib.load(threshold_path)
    print(f"   -> Threshold: {threshold:.2f}")

    print(f"\n4. Generating predictions")
    X = features_df.drop("user_id", axis=1)
    probs = model.predict_proba(X)[:, 1]
    ml_preds = (probs >= threshold).astype(int)

    print(f"\n5. Applying rules engine")
    rules_flags = features_df.apply(rules_engine, axis=1).astype(int)
    combined_preds = np.maximum(ml_preds, rules_flags.to_numpy())

    flagged_users = features_df.loc[combined_preds == 1, "user_id"].tolist()
    print(f"   -> ML model flagged: {int(ml_preds.sum())} users")
    print(f"   -> Rules flagged: {int(rules_flags.sum())} users")
    print(f"   -> Combined flagged: {len(flagged_users)} users")
    return flagged_users, probs, features_df


def save_submission(flagged_user_ids, team_name, language, output_dir="."):
    """Write prediction files in competition format."""
    directory = ensure_directory(output_dir)
    filename = directory / f"{team_name}.detections.{language}.txt"
    with filename.open("w", encoding="utf-8") as handle:
        for user_id in flagged_user_ids:
            handle.write(f"{user_id}\n")
    print(f"\nSubmission saved to: {filename}")
    print(f"  -> {len(flagged_user_ids)} user IDs")
    return str(filename)


if __name__ == "__main__":
    train_full_pipeline(
        dataset_path="data/train_en/dataset.posts&users.json",
        ground_truth_path="data/train_en/dataset.bots.txt",
        language="en",
    )
    train_full_pipeline(
        dataset_path="data/train_fr/dataset.posts&users.json",
        ground_truth_path="data/train_fr/dataset.bots.txt",
        language="fr",
    )
