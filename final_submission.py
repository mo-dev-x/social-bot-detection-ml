"""One-shot script for the final competition submission window."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from src.model_training import run_final_detection, save_submission

# ============================================================================
# CONFIGURATION: Update this before the final hour.
# ============================================================================
TEAM_NAME = "SignalGuard"
# ============================================================================


def _count_submission_ids(filepath: Path) -> int:
    """Count non-empty IDs in a submission file."""
    lines = [line.strip() for line in filepath.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines)


def parse_args() -> argparse.Namespace:
    """Parse optional dataset overrides for dry runs."""
    parser = argparse.ArgumentParser(description="Run final bot-detection submission generation.")
    parser.add_argument(
        "--dataset-dir",
        default="data/final_eval",
        help="Directory containing final_eval_en.json and final_eval_fr.json by default.",
    )
    parser.add_argument(
        "--en-dataset",
        help="Optional explicit path for the English dataset. Useful for dry runs on training data.",
    )
    parser.add_argument(
        "--fr-dataset",
        help="Optional explicit path for the French dataset. Useful for dry runs on training data.",
    )
    parser.add_argument(
        "--team-name",
        default=TEAM_NAME,
        help="Override the team name without editing this file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run final detection and save submissions."""
    args = parse_args()
    start_time = datetime.now()
    print(f"\n{'=' * 60}")
    print(f"FINAL SUBMISSION: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Team: {args.team_name}")
    print(f"{'=' * 60}")

    Path("submissions").mkdir(parents=True, exist_ok=True)

    statuses: list[tuple[str, str]] = []
    created_files: list[Path] = []
    dataset_dir = Path(args.dataset_dir)
    dataset_overrides = {
        "en": Path(args.en_dataset) if args.en_dataset else dataset_dir / "final_eval_en.json",
        "fr": Path(args.fr_dataset) if args.fr_dataset else dataset_dir / "final_eval_fr.json",
    }

    for language, label, dataset_name in [
        ("en", "ENGLISH", "final_eval_en.json"),
        ("fr", "FRENCH", "final_eval_fr.json"),
    ]:
        dataset_path = dataset_overrides[language]
        print(f"\n{label} DETECTION")
        print(f"  Dataset: {dataset_path}")

        if not dataset_path.exists():
            print(f"  Dataset not found: {dataset_path}")
            print(f"  Skipping {language} run.")
            statuses.append((language, "skipped"))
            continue

        try:
            flagged_ids, _, _ = run_final_detection(str(dataset_path), language=language)
            output_path = Path(save_submission(flagged_ids, args.team_name, language, output_dir="submissions"))
            created_files.append(output_path)
            print(f"  Flagged {len(flagged_ids)} users")
            statuses.append((language, "success"))
        except Exception as exc:
            print(f"  Error during {language} detection: {exc}")
            statuses.append((language, "failed"))

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print(f"{'=' * 60}")

    print("\nSUBMISSION FILES:")
    for language in ["en", "fr"]:
        filepath = Path("submissions") / f"{args.team_name}.detections.{language}.txt"
        if filepath.exists():
            count = _count_submission_ids(filepath)
            print(f"  OK  {filepath.name} ({count} IDs)")
        else:
            print(f"  --  {filepath.name} (not created)")

    failed = [language for language, status in statuses if status == "failed"]
    succeeded = [language for language, status in statuses if status == "success"]

    if failed:
        print("\nSubmission run finished with errors.")
        print(f"Failed languages: {', '.join(failed)}")
        return 1

    if not succeeded:
        print("\nNo submission files were created.")
        print("Make sure the final evaluation datasets are present in data/final_eval/.")
        return 1

    print("\nSubmission run completed successfully.")
    print("Next step: email the generated submission file(s) with your GitHub repo link.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
