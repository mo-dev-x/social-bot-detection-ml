# Pre-Competition Checklist

## Immediate setup

- Update `TEAM_NAME` in `final_submission.py`.
- Keep using `.gitignore` as-is; no rename is needed.
- Make sure the project folders exist: `data/`, `models/`, and `submissions/`.
- Install dependencies with `python -m pip install -r requirements.txt` inside the project virtual environment.
- Run `python test_comprehensive.py` to verify the environment and code readiness.

## Once practice data arrives

1. Put the files in the expected structure:

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

2. Run the quick readiness check:

```bash
python test_everything.py
```

3. Run the broader validation suite:

```bash
python test_comprehensive.py
```

4. Train the models:

```bash
python model_training.py
```

5. Verify the expected outputs exist in `models/`:

- `model_en.pkl`
- `model_fr.pkl`
- `threshold_en.pkl`
- `threshold_fr.pkl`

## Before the competition window

- Make sure the GitHub repo is public and up to date.
- Confirm `final_submission.py` is the submission script you intend to use.
- Verify `TEAM_NAME` is correct in `final_submission.py`.
- Keep a backup copy of the trained models.
- Run `python test_comprehensive.py` one more time.

## During the competition hour

1. Place the final evaluation files in `data/final_eval/`.
2. Run:

```bash
python final_submission.py
```

3. Confirm the expected submission files were created in `submissions/`.
4. Verify each file contains one user ID per line with no header.
5. Email the submission file(s) and the GitHub repo link before the deadline.

## Troubleshooting notes

- If `data/final_eval/final_eval_en.json` or `data/final_eval/final_eval_fr.json` is missing, `final_submission.py` will skip that language.
- If model files are missing, retrain with `python model_training.py` before the competition window.
- If imports fail, make sure you are running commands from the project root with the project virtual environment active.

## Scoring reminder

- `+2` for each true positive
- `-2` for each false negative
- `-6` for each false positive

The pipeline should stay precision-first. Avoid lowering thresholds aggressively unless validation clearly supports it.
