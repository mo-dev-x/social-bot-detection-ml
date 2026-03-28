"""CLI entry point for final detection runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model_training import run_final_detection, save_submission


def build_parser():
    parser = argparse.ArgumentParser(description="Run bot detection on a dataset.")
    parser.add_argument("dataset_path", help="Path to the JSON dataset")
    parser.add_argument("output_path", help="Output submission file path")
    parser.add_argument("--language", default="en", choices=["en", "fr"], help="Dataset language")
    parser.add_argument("--team-name", default="team_name", help="Team name for output naming")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    flagged_ids, _, _ = run_final_detection(args.dataset_path, language=args.language)
    output_path = Path(args.output_path)
    filename = save_submission(
        flagged_ids,
        args.team_name,
        args.language,
        output_dir=output_path.parent if str(output_path.parent) != "." else ".",
    )
    print(f"Detection complete. Output written to {filename}")


if __name__ == "__main__":
    main()
