# Social Bot Detection ML

Behavior-based bot detection pipeline for social media account classification.

This project classifies accounts using behavioral and profile signals rather than topic or sentiment modeling. The system aggregates post activity into user-level features, trains a LightGBM classifier, applies threshold tuning against an asymmetric scoring rule, and combines the model with a conservative rules layer to reduce false positives.

## Why this approach

- Bots can imitate language more easily than they can imitate human timing and activity patterns.
- Precision matters more than raw recall when false positives are expensive.
- Interpretable features make debugging, validation, and threshold tuning much easier.

## Repository layout

```text
.
|-- src/
|   |-- feature_extraction.py
|   |-- model_training.py
|   |-- detector.py
|   |-- rules_engine.py
|   `-- utils.py
|-- data/
|-- models/
|-- submissions/
|-- feature_extraction.py
|-- model_training.py
|-- final_submission.py
|-- test_everything.py
`-- requirements.txt
```

The `src/` package is the source of truth. The root-level Python files are lightweight entry points for convenience.

## Feature groups

The current extractor builds user-level features across several groups:

- Temporal behavior: posting interval statistics, hour/day entropy, weekend ratio, sleep/work activity patterns, and burstiness windows.
- Text repetition and style: duplicate ratio, cosine similarity, near-duplicate ratio, vocabulary diversity, vocabulary repetition, punctuation, emoji, uppercase, digit, hashtag, mention, reply, and URL usage.
- Language consistency: lightweight per-post language hints and language-switch counts.
- Profile features: username entropy, digit ratio, description presence and length, location presence, and name-to-username similarity.
- Activity statistics: tweet count, z-score, tweets per hour/day, and activity-day coverage.

## Pipeline

1. Load raw JSON datasets containing users and posts.
2. Aggregate posts at the user level and build engineered features.
3. Train a LightGBM classifier on labeled users.
4. Search decision thresholds using the asymmetric competition score.
5. Apply a conservative rules layer for obvious high-risk cases.
6. Export flagged user IDs in submission format.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a model:

```bash
python model_training.py
```

Run the final submission flow:

```bash
python final_submission.py
```

Smoke-test feature extraction:

```bash
python test_everything.py
```

## Data expectations

Training datasets are expected under paths like:

```text
data/
|-- train_en/
|   |-- dataset.posts&users.json
|   `-- dataset.bots.txt
|-- train_fr/
|   |-- dataset.posts&users.json
|   `-- dataset.bots.txt
`-- final_eval/
    |-- final_eval_en.json
    `-- final_eval_fr.json
```

The JSON loader expects top-level `users` and `posts` keys.

## Output

Predictions are saved in competition-friendly text files:

```text
submissions/TEAM_NAME.detections.en.txt
submissions/TEAM_NAME.detections.fr.txt
```

Each file contains one user ID per line.

## Notes

- Models and submission outputs are ignored by git.
- The repository includes empty `data/`, `models/`, and `submissions/` folders so the expected layout is visible without committing generated files.
- Full runtime verification still requires installing the Python dependencies in your environment.
