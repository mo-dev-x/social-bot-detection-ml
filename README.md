# Social Bot Detection ML

SignalGuard's bot-detection pipeline for the Bot or Not competition.

The project treats bot detection as a user-level classification problem. It aggregates each user's posts into behavioral, repetition, profile, and activity features, trains separate English and French LightGBM models, tunes decision thresholds against the competition metric, and exports submission-ready user ID lists.

## What the project optimizes for

- User-level detection, not tweet-level classification
- Behavioral signal over semantic NLP
- Very low false positives because the scoring rule is asymmetric
- Separate English and French models
- Cross-batch validation instead of random within-batch splits

The competition score is:

```text
score = 2 * TP - 2 * FN - 6 * FP
```

That makes false positives much more expensive than missed bots, so the pipeline is intentionally conservative.

## Repository structure

```text
.
|-- src/
|   |-- feature_extraction.py
|   |-- model_training.py
|   |-- detector.py
|   |-- rules_engine.py
|   `-- utils.py
|-- documentation/
|   |-- ARCHITECTURE_DECISIONS.md
|   `-- PROJECT_TRACKER.md
|-- data/
|   |-- training/
|   |-- evaluation_samples/
|   `-- final_eval/
|-- models/
|-- submissions/
|-- feature_extraction.py
|-- model_training.py
|-- final_submission.py
|-- test_everything.py
|-- test_comprehensive.py
`-- requirements.txt
```

The `src/` package is the source of truth. Root-level Python files are convenience entry points.

## Documentation

The project docs live in `documentation/`:

- `documentation/ARCHITECTURE_DECISIONS.md`
  - records the architectural choices behind the approach
- `documentation/PROJECT_TRACKER.md`
  - tracks progress, validation phases, and remaining competition work

If you want the repo narrative first, start there before reading the code.

## Data layout

The repository now uses `data/` as the single home for datasets:

```text
data/
|-- training/
|   |-- dataset.posts&users.1.json
|   |-- dataset.bots.1.txt
|   |-- ...
|   |-- dataset.posts&users.6.json
|   `-- dataset.bots.6.txt
|-- evaluation_samples/
|   |-- dataset.posts&users.30.json
|   |-- dataset.bots.30.txt
|   |-- dataset.posts&users.31.json
|   `-- dataset.bots.31.txt
`-- final_eval/
    |-- final_eval_en.json
    `-- final_eval_fr.json
```

Notes:

- `data/training/` contains the labeled competition training batches used for model training.
- `data/evaluation_samples/` contains additional labeled sample evaluation-style datasets you can use for experiments or sanity checks.
- `data/final_eval/` is reserved for the real final drop during the submission window.

The JSON files are expected to contain top-level `users` and `posts` keys.

## Current model pipeline

1. Load a labeled training batch from `data/training/`.
2. Group posts by `author_id`.
3. Build one feature row per user.
4. Train a LightGBM model per language.
5. Run leave-one-batch-out validation with `GroupKFold`.
6. Search thresholds using the competition score.
7. Apply the rules engine as a conservative safety layer.
8. Save trained models and thresholds to `models/`.

## Feature groups

The current feature extractor focuses on compact user-level signals:

- Temporal:
  - `mean_time_delta`
  - `std_time_delta`
  - `min_time_delta`
  - `max_time_delta`
  - `cv_time_delta`
  - `hour_entropy`
  - `night_activity_ratio`
  - `tweets_per_hour`
- Text behavior:
  - `avg_tweet_length`
  - `std_tweet_length`
  - `type_token_ratio`
  - `unique_words_ratio`
- Repetition / similarity:
  - `duplicate_tweet_ratio`
  - `avg_similarity_between_tweets`
  - `max_similarity`
- Profile:
  - `username_length`
  - `digit_ratio`
  - `username_entropy`
  - `bio_length`
  - `has_description`
- Activity:
  - `tweet_count`
  - `z_score`
  - `tweet_time_span`

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lightweight smoke test:

```bash
python test_everything.py
```

Run the broader validation suite:

```bash
python test_comprehensive.py
```

Train both language-specific models:

```bash
python model_training.py
```

This writes artifacts such as:

```text
models/model_en.pkl
models/threshold_en.pkl
models/training_report_en.json
models/model_fr.pkl
models/threshold_fr.pkl
models/training_report_fr.json
```

## Final submission flow

When the final evaluation files arrive, place them in:

```text
data/final_eval/final_eval_en.json
data/final_eval/final_eval_fr.json
```

Then run:

```bash
python final_submission.py
```

Submission files are written to:

```text
submissions/TEAM_NAME.detections.en.txt
submissions/TEAM_NAME.detections.fr.txt
```

Each file contains one user ID per line.

## Important implementation notes

- Models are trained separately for English and French.
- Thresholds are also chosen separately for English and French.
- `src/model_training.py` expects labeled training data under `data/training/`.
- `final_submission.py` expects the actual final drop under `data/final_eval/`.
- The rules engine is intentionally conservative because of the FP penalty.

## Current status

- The codebase is aligned with the real dataset schema.
- The training and validation scripts run successfully against the current repo layout.
- The documentation folder now reflects the project plan and architecture.
- Model quality still needs improvement before final submission, so `documentation/PROJECT_TRACKER.md` remains the source of truth for next steps.
