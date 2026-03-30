# French model improvement plan

**Problem:** OOF score 30, recall 66.3% — missing 28 out of 83 bots.  
**Root cause:** High fold variance (-10, +36, +10) means the model hasn't learned patterns that generalize across all 3 FR batches.  
**Goal:** Push OOF score above 60, recall above 78%, while keeping precision above 90%.

> Do these phases in order. Each one is a concrete code change + a retrain + a check of the new report.

---

## Phase A — Widen the threshold search

**The single fastest win. Do this first.**

**What's wrong:**  
`find_best_threshold` currently searches `np.linspace(0.50, 0.99, 100)` — it never considers thresholds below 0.5. For FR, the model is under-flagging: it has 28 FNs and only 4 FPs. The math says you can afford to recover roughly 7 more TPs before a single additional FP breaks even. The optimal threshold for FR is almost certainly below 0.5.

**Code change — `src/model_training.py`, `find_best_threshold`:**

```python
# Before
thresholds = np.linspace(0.50, 0.99, 100)

# After — add a language parameter and widen for FR
def find_best_threshold(y_true, y_probs, feature_frame, language: str = "en"):
    low = 0.30 if language == "fr" else 0.50
    thresholds = np.linspace(low, 0.99, 140)
```

Then pass `language=language` wherever `find_best_threshold` is called inside `_cross_batch_validate` and `train_full_pipeline`.

**Check after retraining:**  
- Does `best_threshold` for FR drop below 0.5?
- Does OOF recall go above 75%?
- Does OOF precision stay above 88%? (At the break-even point, each extra TP is worth it as long as you don't add more than 0.33 FPs per TP recovered.)

---

## Phase B — Diagnose what the 28 missed bots look like

**Before changing any features, understand what you're missing.**

Add this diagnostic script and run it once on the FR training data. It tells you which feature ranges the FN bots fall into so you know exactly what signal is missing.

**New file — `diagnose_fr.py` (run once, don't commit to main pipeline):**

```python
"""Diagnostic: profile the 28 missed FR bots to find what feature they share."""
from pathlib import Path
import pandas as pd
from src.feature_extraction import create_feature_dataframe
from src.model_training import build_training_dataframe
import joblib

training_df, _ = build_training_dataframe()
fr_df = training_df[training_df["language"] == "fr"].copy()

bundle = joblib.load("models/model_fr.pkl")
threshold = joblib.load("models/threshold_fr.pkl")
feature_names = bundle["feature_names"]
model = bundle["model"]

X = fr_df[feature_names]
probs = model.predict_proba(X)[:, 1]
fr_df["prob"] = probs
fr_df["pred"] = (probs >= threshold).astype(int)

# The 28 bots the model missed
fn_bots = fr_df[(fr_df["label"] == 1) & (fr_df["pred"] == 0)]
tp_bots = fr_df[(fr_df["label"] == 1) & (fr_df["pred"] == 1)]
humans  = fr_df[fr_df["label"] == 0]

print(f"FN bots:  {len(fn_bots)}")
print(f"TP bots:  {len(tp_bots)}")
print(f"\n=== FN bots — feature means ===")
print(fn_bots[feature_names].mean().sort_values().to_string())
print(f"\n=== TP bots — feature means (for comparison) ===")
print(tp_bots[feature_names].mean().sort_values().to_string())
print(f"\n=== FN bots — probability distribution ===")
print(fn_bots["prob"].describe())
```

**What to look for in the output:**

- If FN bots have **high `cv_time_delta`** (>0.5) — they post irregularly, which looks human. You need text-based features to catch them.
- If FN bots have **low `duplicate_tweet_ratio`** but high `avg_similarity_between_tweets` — they vary their exact wording but recycle themes. The `near_duplicate_ratio` threshold of 0.85 may be too strict; try 0.70.
- If FN bots have **low `tweet_count`** (<15) — they're low-volume bots that your burst/regularity features can't see. Profile features become more important.
- If FN bots have **probabilities between 0.35–0.48** — Phase A (threshold widening) will catch them directly, no feature work needed.
- If FN bots have **probabilities below 0.35** — the model genuinely doesn't recognize them. You need new features (Phase D).

**Do not proceed to Phase C or D until you've run this and understand which case you're in.**

---

## Phase C — Fix the rules engine for French

**The rules engine fires 0 times on all FR training data. It's completely inactive.**

This isn't because FR bots are subtle — it's because the rule thresholds were hand-written for EN patterns and never calibrated to FR. Fix this using the actual FR bot distribution.

**Step 1 — Check what FR bot features actually look like:**

```python
# Add to diagnose_fr.py or run interactively
all_bots = fr_df[fr_df["label"] == 1]
print(all_bots[[
    "near_duplicate_ratio", "tweet_count", "burst_ratio_1h",
    "cross_user_repost_ratio", "template_duplicate_ratio",
    "hour_uniform_chi2", "duplicate_tweet_ratio", "cv_time_delta",
    "hour_entropy", "z_score", "tweets_per_hour"
]].describe())
```

**Step 2 — Recalibrate `src/rules_engine.py` thresholds based on what you see:**

The current rules require very high thresholds simultaneously (e.g., `near_duplicate_ratio >= 0.50 AND burst_ratio_1h >= 0.35`). French bots may hit each condition moderately but not hit both at the same time. Loosen the conjunctions:

```python
# Current — too strict
duplicate_burst_rule = (
    row["near_duplicate_ratio"] >= 0.50
    and row["tweet_count"] >= 20
    and row["burst_ratio_1h"] >= 0.35
)

# Loosened — adjust numbers after checking the actual FR distribution
duplicate_burst_rule = (
    row["near_duplicate_ratio"] >= 0.40   # lower threshold
    and row["tweet_count"] >= 15           # lower count
    and row["burst_ratio_1h"] >= 0.25      # lower burst
)
```

**Calibration rule:** after each change, run `summarize_rules(feature_frame, y)` on the FR training data and check:
- Flags should increase from 0 to at least 5–10
- Precision of rules alone should stay above 85% — if it drops below that, tighten back

**Step 3 — Add one FR-specific rule targeting template repetition:**

French bot campaigns often recycle sentence templates more than EN ones. Add this rule:

```python
template_spam_rule = (
    row["template_top_ratio"] >= 0.40      # >40% of tweets share the same template
    and row["avg_similarity_between_tweets"] >= 0.55
    and row["tweet_count"] >= 12
)
```

---

## Phase D — Add FR-targeted features (only if Phase B shows probabilities below 0.35)

**Skip this phase if Phase A (threshold widening) already recovers most FNs.**

If the diagnostic shows that many FN bots have probabilities well below 0.35, the model is blind to them — meaning the current feature set doesn't contain the right signal. These are the two highest-impact additions for French specifically.

### D1 — Posting interval periodicity score

FR bots that post irregularly in timing but on a fixed schedule (e.g., every day at slightly different times) won't be caught by `cv_time_delta`. Add a feature that detects day-level periodicity:

```python
# In FeatureExtractor._extract_temporal_features
# After computing deltas:

# Periodicity: what fraction of inter-post gaps are near a common divisor?
if len(deltas) >= 4:
    median_delta = float(np.median(deltas))
    if median_delta > 0:
        residuals = deltas % median_delta
        near_periodic = np.sum(residuals < 0.15 * median_delta) / len(deltas)
        base["periodic_interval_ratio"] = float(near_periodic)
    else:
        base["periodic_interval_ratio"] = 0.0
else:
    base["periodic_interval_ratio"] = 0.0
```

### D2 — Vocabulary concentration (FR-specific)

FR bots that don't duplicate tweets exactly but recycle a small core vocabulary will have a low type-token ratio but may slip past `duplicate_tweet_ratio`. Add:

```python
# In FeatureExtractor._extract_text_features
# After computing all_tokens:

# Top-10 word concentration: what fraction of all tokens are the 10 most common words?
if len(all_tokens) >= 20:
    top10_count = sum(count for _, count in Counter(all_tokens).most_common(10))
    base["top10_word_concentration"] = safe_divide(top10_count, len(all_tokens))
else:
    base["top10_word_concentration"] = 0.0
```

### D3 — Tune LightGBM specifically for FR class imbalance

FR has 83 bots out of ~626 total users (~13% bot rate). The model may be underweighting bots. Add `class_weight` to the FR model:

```python
# In train_lightgbm_model, change FR config:
model = lgb.LGBMClassifier(
    n_estimators=160,
    learning_rate=0.05,
    num_leaves=24,          # reduce from 31 — FR has less data, easier to overfit
    min_child_samples=15,   # reduce from 25 — FR has fewer bots per leaf
    min_split_gain=0.05,
    reg_lambda=2.0,         # increase regularization slightly
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight="balanced", # ADD THIS — upweights the minority bot class
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
```

---

## Phase E — Final FR retrain and validation check

Once you've applied whichever phases improved the diagnostic scores, do a clean final retrain and check all three numbers before committing:

```bash
python model_training.py  # retrains both EN and FR
```

**Accept the new FR model only if all three conditions hold:**

| Metric | Minimum to accept |
|--------|-------------------|
| OOF score | ≥ 50 (currently 30) |
| Precision | ≥ 88% (currently 93% — can afford to drop slightly) |
| Recall | ≥ 75% (currently 66%) |
| Fold 1 score | ≥ 0 (currently -10 — must not be negative) |

If fold 1 is still negative after all changes, the model will likely hurt you on the final eval batch if it resembles batch 6. In that case, **raise the FR threshold manually** to eliminate FPs even at the cost of more FNs — a score of 0 is better than -10.

**Do not touch the EN model.** It's solid — 93% precision, 88% recall, consistent folds. Any change risks breaking it.

---

## Execution order summary

```
Phase A  →  retrain FR  →  check new report
    If recall > 75%: go straight to Phase E
    If recall < 75%: continue to Phase B

Phase B  →  run diagnose_fr.py  →  read the output
    If FN probs are 0.35–0.48: Phase A was enough, go to Phase E
    If rules never fire on bots: go to Phase C
    If FN probs are below 0.35: go to Phase D

Phase C  →  recalibrate rules  →  retrain FR  →  check new report
Phase D  →  add features       →  retrain FR  →  check new report

Phase E  →  final clean retrain of both models  →  verify report  →  commit
```

---

*Last updated: March 30, 2026*
