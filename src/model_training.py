"""Model training and inference pipeline."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .feature_extraction import create_feature_dataframe
from .rules_engine import rules_engine
from .utils import ensure_directory, load_json_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATTERN = re.compile(r"dataset\.posts&users\.(\d+)\.json$")
DEFAULT_DATASET_DIR = Path("data/training")


def compute_competition_score(y_true, y_pred):
    """Official asymmetric competition score."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    score = 2 * tp - 2 * fn - 6 * fp
    return {
        "score": int(score),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }


def _build_lgbm_params(language: str = "multi", seed: int = 42, feature_fraction: float | None = None):
    if language == "fr":
        params = {
            "n_estimators": 160,
            "learning_rate": 0.05,
            "num_leaves": 24,
            "min_child_samples": 15,
            "min_split_gain": 0.05,
            "reg_lambda": 2.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "class_weight": "balanced",
        }
    else:
        params = {
            "n_estimators": 180,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 25,
            "min_split_gain": 0.05,
            "reg_lambda": 1.5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
    if feature_fraction is not None:
        params["colsample_bytree"] = feature_fraction
    params.update({"random_state": seed, "verbose": -1, "n_jobs": -1})
    return params


def train_lightgbm_model(X_train, y_train, language: str = "multi", seed: int = 42, feature_fraction: float | None = None):
    """Train one LightGBM classifier with conservative defaults."""
    logger.info("Training LightGBM model for %s (seed=%s)", language.upper(), seed)
    model = lgb.LGBMClassifier(**_build_lgbm_params(language=language, seed=seed, feature_fraction=feature_fraction))
    model.fit(X_train, y_train)
    return model


def train_lightgbm_ensemble(X_train, y_train, language: str = "multi", seeds=None):
    """Train a small seed ensemble for more stable probabilities."""
    seeds = seeds or [42, 777, 2026]
    models = []
    for seed in seeds:
        models.append(train_lightgbm_model(X_train, y_train, language=language, seed=seed, feature_fraction=0.8))
    return models


def predict_probabilities(model_or_models, X):
    """Predict probabilities from either a single model or a list of models."""
    if isinstance(model_or_models, list):
        probabilities = [model.predict_proba(X)[:, 1] for model in model_or_models]
        return np.mean(probabilities, axis=0)
    return model_or_models.predict_proba(X)[:, 1]


def apply_safety_layer(features_df, probabilities, threshold, language: str | None = None):
    """Combine model confidence with strict high-precision rules."""
    rule_frame = features_df.copy()
    if language:
        rule_frame["_language"] = language
    rule_flags = rule_frame.apply(rules_engine, axis=1).to_numpy(dtype=bool)
    preds = ((np.asarray(probabilities) >= threshold) | rule_flags).astype(int)
    return preds, rule_flags.astype(int)


def find_best_threshold(y_true, y_probs, feature_frame, language: str = "en"):
    """Search thresholds using the competition score and safety rules."""
    low = 0.30 if language == "fr" else 0.50
    thresholds = np.linspace(low, 0.99, 140)
    results = []
    for threshold in thresholds:
        y_pred, rule_flags = apply_safety_layer(feature_frame, y_probs, threshold, language=language)
        score_dict = compute_competition_score(y_true, y_pred)
        results.append(
            {
                "threshold": float(threshold),
                "rule_flags": int(rule_flags.sum()),
                **score_dict,
            }
        )

    results.sort(key=lambda item: (item["score"], item["precision"], -item["fp"]), reverse=True)
    best = results[0]
    print("\n=== THRESHOLD TUNING RESULTS (TOP 10) ===")
    print(f"{'Threshold':<10} {'Score':<8} {'Precision':<10} {'Recall':<8} {'TP':<4} {'FN':<4} {'FP':<4}")
    print("-" * 62)
    for result in results[:10]:
        print(
            f"{result['threshold']:<10.2f} {result['score']:<8} "
            f"{result['precision']:<10.2f} {result['recall']:<8.2f} "
            f"{result['tp']:<4} {result['fn']:<4} {result['fp']:<4}"
        )
    print(f"\nBest threshold: {best['threshold']:.2f} (Score: {best['score']})")
    return best["threshold"], pd.DataFrame(results)


def validate_model(model, X_val, y_val, threshold: float = 0.85, feature_frame=None):
    """Score the model using the conservative safety layer."""
    probabilities = predict_probabilities(model, X_val)
    if feature_frame is None:
        raise ValueError("feature_frame is required for safety-layer validation.")

    preds, rule_flags = apply_safety_layer(feature_frame, probabilities, threshold)
    score_dict = compute_competition_score(y_val, preds)
    print("\n=== VALIDATION RESULTS ===")
    print(f"Decision Threshold: {threshold:.2f}")
    print(f"Rules approved: {int(rule_flags.sum())}")
    print(f"Competition Score: {score_dict['score']}")
    print(f"True Positives: {score_dict['tp']}")
    print(f"False Negatives: {score_dict['fn']}")
    print(f"False Positives: {score_dict['fp']}")
    print(f"True Negatives: {score_dict['tn']}")
    print(f"Precision: {score_dict['precision']:.2%}")
    print(f"Recall: {score_dict['recall']:.2%}")
    return score_dict


def analyze_feature_importance(model, X_train):
    """Return ranked feature importance."""
    if isinstance(model, list):
        importances = np.mean([member.feature_importances_ for member in model], axis=0)
    else:
        importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": importances}
    ).sort_values("importance", ascending=False)
    print("\n=== TOP FEATURES ===")
    print(feature_importance_df.head(15))
    return feature_importance_df


def summarize_rules(feature_frame, labels, language: str | None = None):
    """Summarize standalone rule behavior on labeled data."""
    rule_frame = feature_frame.copy()
    if language:
        rule_frame["_language"] = language
    flags = rule_frame.apply(rules_engine, axis=1).astype(int).to_numpy()
    score = compute_competition_score(labels, flags)
    return {
        "flags": int(flags.sum()),
        **score,
    }


def select_feature_columns(training_df, excluded_columns):
    """Drop dead or near-constant features before training."""
    candidate_columns = [column for column in training_df.columns if column not in excluded_columns]
    variances = training_df[candidate_columns].var(numeric_only=True)
    kept_columns = variances[variances > 1e-8].index.tolist()
    dropped_columns = variances[variances <= 1e-8].index.tolist()
    return kept_columns, dropped_columns


def _discover_dataset_specs(dataset_dir: str | Path = DEFAULT_DATASET_DIR):
    dataset_dir = Path(dataset_dir)
    specs = []
    for json_path in sorted(dataset_dir.glob("dataset.posts&users.*.json")):
        match = DATASET_PATTERN.search(json_path.name)
        if not match:
            continue
        dataset_id = match.group(1)
        labels_path = dataset_dir / f"dataset.bots.{dataset_id}.txt"
        if not labels_path.exists():
            logger.warning("Skipping %s because %s is missing", json_path.name, labels_path.name)
            continue

        payload = load_json_dataset(json_path)
        specs.append(
            {
                "dataset_id": dataset_id,
                "batch_id": payload.get("id", dataset_id),
                "language": payload.get("lang", "en"),
                "dataset_path": json_path,
                "ground_truth_path": labels_path,
            }
        )
    return specs


def _load_bot_ids(ground_truth_path: str | Path):
    return {line.strip() for line in Path(ground_truth_path).read_text(encoding="utf-8").splitlines() if line.strip()}


def build_training_dataframe(dataset_specs=None, dataset_dir: str | Path = DEFAULT_DATASET_DIR):
    """Aggregate all labeled datasets into one user-level table."""
    dataset_specs = dataset_specs or _discover_dataset_specs(dataset_dir)
    if not dataset_specs:
        raise FileNotFoundError("No labeled datasets were found.")

    frames = []
    for spec in dataset_specs:
        features_df = create_feature_dataframe(spec["dataset_path"], language=spec["language"])
        bot_ids = _load_bot_ids(spec["ground_truth_path"])
        features_df["label"] = features_df["user_id"].isin(bot_ids).astype(int)
        features_df["batch_id"] = str(spec["batch_id"])
        features_df["language"] = spec["language"]
        features_df["dataset_id"] = str(spec["dataset_id"])
        frames.append(features_df)

    training_df = pd.concat(frames, ignore_index=True)
    training_df = training_df.fillna(0)
    return training_df, dataset_specs


def _align_feature_columns(features_df, feature_names):
    aligned = features_df.reindex(columns=feature_names, fill_value=0)
    return aligned.replace([np.inf, -np.inf], 0).fillna(0)


def _cross_batch_validate(X, y, groups, feature_frame, language):
    unique_groups = pd.Series(groups).nunique()
    if unique_groups < 2:
        raise ValueError("GroupKFold requires at least two distinct batches.")

    splitter = GroupKFold(n_splits=min(5, unique_groups))
    oof_probs = np.zeros(len(X), dtype=float)
    fold_reports = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        train_groups = sorted(set(groups.iloc[train_idx]))
        val_groups = sorted(set(groups.iloc[val_idx]))
        print(f"\n=== FOLD {fold_idx} ===")
        print(f"Train batches: {train_groups}")
        print(f"Validate batches: {val_groups}")

        model = train_lightgbm_ensemble(X.iloc[train_idx], y.iloc[train_idx], language=language)
        probs = predict_probabilities(model, X.iloc[val_idx])
        oof_probs[val_idx] = probs

        threshold, _ = find_best_threshold(
            y.iloc[val_idx],
            probs,
            feature_frame.iloc[val_idx].reset_index(drop=True),
            language=language,
        )
        preds, _ = apply_safety_layer(
            feature_frame.iloc[val_idx].reset_index(drop=True),
            probs,
            threshold,
            language=language,
        )
        report = compute_competition_score(y.iloc[val_idx], preds)
        report["fold"] = fold_idx
        report["threshold"] = threshold
        report["train_batches"] = train_groups
        report["val_batches"] = val_groups
        fold_reports.append(report)
        print(
            f"Fold {fold_idx} score={report['score']} precision={report['precision']:.2%} "
            f"recall={report['recall']:.2%} fp={report['fp']}"
        )

    return oof_probs, pd.DataFrame(fold_reports)


def train_full_pipeline(dataset_path=None, ground_truth_path=None, language: str | None = None, models_dir: str = "models"):
    """Train one language-specific model and persist the resulting artifacts."""
    del dataset_path, ground_truth_path

    if not language:
        raise ValueError("train_full_pipeline now requires language='en' or language='fr'.")

    print(f"\n{'=' * 60}")
    print(f"TRAINING PIPELINE: {language.upper()}")
    print(f"{'=' * 60}")

    training_df, dataset_specs = build_training_dataframe()
    training_df = training_df.loc[training_df["language"] == language].reset_index(drop=True)
    dataset_specs = [spec for spec in dataset_specs if spec["language"] == language]
    if training_df.empty:
        raise ValueError(f"No datasets found for language={language!r}")

    print("\n1. Labeled datasets")
    for spec in dataset_specs:
        print(f"   -> batch={spec['batch_id']} lang={spec['language']} file={Path(spec['dataset_path']).name}")

    print("\n2. User-level training table")
    print(f"   -> Rows: {len(training_df)}")
    print(f"   -> Bots: {int(training_df['label'].sum())}")
    print(f"   -> Batches: {sorted(training_df['batch_id'].unique().tolist())}")

    feature_columns, dropped_features = select_feature_columns(
        training_df,
        excluded_columns={"user_id", "label", "batch_id", "language", "dataset_id"},
    )
    X = training_df[feature_columns]
    y = training_df["label"]
    groups = training_df["batch_id"]
    feature_frame = training_df[feature_columns]

    if dropped_features:
        print(f"   -> Dropped near-constant features: {', '.join(dropped_features)}")

    print("\n3. Cross-batch validation with GroupKFold")
    oof_probs, fold_report_df = _cross_batch_validate(X, y, groups, feature_frame, language=language)

    print("\n4. Threshold search on out-of-fold predictions")
    best_threshold, threshold_report = find_best_threshold(y, oof_probs, feature_frame, language=language)
    oof_preds, rule_flags = apply_safety_layer(feature_frame, oof_probs, best_threshold, language=language)
    oof_results = compute_competition_score(y, oof_preds)
    print(
        f"   -> OOF score={oof_results['score']} precision={oof_results['precision']:.2%} "
        f"recall={oof_results['recall']:.2%} fp={oof_results['fp']} rules={int(rule_flags.sum())}"
    )
    rules_summary = summarize_rules(feature_frame, y, language=language)
    print(
        f"   -> Rules alone score={rules_summary['score']} precision={rules_summary['precision']:.2%} "
        f"recall={rules_summary['recall']:.2%} fp={rules_summary['fp']}"
    )

    print("\n5. Training final model on all labeled users")
    model = train_lightgbm_ensemble(X, y, language=language)
    feature_importance = analyze_feature_importance(model, X)

    target_dir = ensure_directory(models_dir)
    bundle = {
        "model": model,
        "feature_names": feature_columns,
        "language": language,
        "batches": sorted(training_df["batch_id"].unique().tolist()),
        "ensemble_seeds": [42, 777, 2026],
    }

    model_path = target_dir / f"model_{language}.pkl"
    threshold_path = target_dir / f"threshold_{language}.pkl"
    report_path = target_dir / f"training_report_{language}.json"

    print("\n6. Saving model artifacts")
    joblib.dump(bundle, model_path)
    joblib.dump(best_threshold, threshold_path)
    report_path.write_text(
        json.dumps(
            {
                "language": language,
                "datasets": [
                    {
                        "batch_id": spec["batch_id"],
                        "language": spec["language"],
                        "dataset_path": str(spec["dataset_path"]),
                        "ground_truth_path": str(spec["ground_truth_path"]),
                    }
                    for spec in dataset_specs
                ],
                "oof_results": oof_results,
                "rules_summary": rules_summary,
                "best_threshold": best_threshold,
                "dropped_features": dropped_features,
                "fold_reports": fold_report_df.to_dict(orient="records"),
                "threshold_candidates_top10": threshold_report.head(10).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"   -> Model: {model_path}")
    print(f"   -> Threshold: {threshold_path}")
    print(f"   -> Report: {report_path}")

    return {
        "model": model,
        "threshold": best_threshold,
        "val_results": oof_results,
        "feature_importance": feature_importance,
        "feature_names": feature_columns,
        "fold_reports": fold_report_df,
        "rule_flags": rule_flags,
    }


def train_all_languages(models_dir: str = "models"):
    """Train both language-specific models."""
    return {
        "en": train_full_pipeline(language="en", models_dir=models_dir),
        "fr": train_full_pipeline(language="fr", models_dir=models_dir),
    }


def run_final_detection(dataset_path, language: str = "en", model_path=None, threshold_path=None):
    """Run inference on an unlabeled evaluation dataset."""
    model_path = Path(model_path or f"models/model_{language}.pkl")
    threshold_path = Path(threshold_path or f"models/threshold_{language}.pkl")

    print(f"\n{'=' * 60}")
    print(f"FINAL DETECTION: {language.upper()}")
    print(f"{'=' * 60}")

    dataset = load_json_dataset(dataset_path)
    dataset_language = language or dataset.get("lang", "en")

    print(f"\n1. Loading dataset: {dataset_path}")
    features_df = create_feature_dataframe(dataset_path, language=dataset_language)
    print(f"   -> Features extracted for {len(features_df)} users")

    print(f"\n2. Loading model: {model_path}")
    bundle = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    feature_names = bundle["feature_names"]
    model = bundle["model"]
    print(f"   -> Threshold: {threshold:.2f}")

    extracted_feature_names = [column for column in features_df.columns if column != "user_id"]
    missing_features = [name for name in feature_names if name not in extracted_feature_names]
    extra_features = [name for name in extracted_feature_names if name not in feature_names]
    print(
        f"   -> Feature schema: extracted={len(extracted_feature_names)} "
        f"expected={len(feature_names)} missing={len(missing_features)} extra={len(extra_features)}"
    )
    if missing_features:
        preview = ", ".join(missing_features[:5])
        print(f"   -> Missing features filled with 0: {preview}")
    if extra_features:
        preview = ", ".join(extra_features[:5])
        print(f"   -> Extra features ignored: {preview}")

    X = _align_feature_columns(features_df.drop(columns=["user_id"]), feature_names)
    probabilities = predict_probabilities(model, X)
    preds, rule_flags = apply_safety_layer(X, probabilities, threshold, language=dataset_language)
    flagged_users = features_df.loc[preds == 1, "user_id"].tolist()

    print(f"\n3. Applying safety layer")
    print(f"   -> Above threshold: {int((probabilities >= threshold).sum())}")
    print(f"   -> Rule-approved: {int(rule_flags.sum())}")
    print(f"   -> Final flagged: {len(flagged_users)}")
    return flagged_users, probabilities, features_df


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
    train_all_languages()
