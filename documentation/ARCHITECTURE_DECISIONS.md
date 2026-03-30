# Architecture Decision Record

**Project:** Bot or Not — SignalGuard detector  
**Competition:** April 2026  
**Authors:** SignalGuard team

This document records the key architectural decisions made for this project, the reasoning behind each choice, and the trade-offs accepted. It is meant to be updated as decisions are revisited.

---

## ADR-001 — User-level classification over tweet-level

**Status:** Accepted

**Context:**  
The dataset contains tweets from individual users. Bot detection could be framed either at the tweet level (is this tweet from a bot?) or the user level (is this account a bot?).

**Decision:**  
Classify at the user level. Aggregate all of a user's tweets into a single feature row, then classify the user.

**Rationale:**
- The competition task is explicitly user-level — submissions are lists of user IDs, not tweet IDs.
- A single tweet contains almost no behavioral signal. Bots can write any individual tweet convincingly. The tell is the *pattern* across dozens of tweets.
- User-level features (temporal variance, vocabulary diversity, burst behavior) are far more discriminative than any per-tweet signal.

**Trade-offs:**
- Users with very few tweets (edge case: 10 tweets minimum per the competition rules) have noisier features.
- We lose any tweet-level signal that doesn't aggregate cleanly (e.g., reaction to specific events).

---

## ADR-002 — LightGBM as the primary classifier

**Status:** Accepted

**Context:**  
We needed a classifier for a tabular feature set (~50 engineered features) on a small dataset (hundreds of users per batch). Options considered: logistic regression, random forest, XGBoost, LightGBM, neural network.

**Decision:**  
Use LightGBM (`LGBMClassifier`) as the primary classifier.

**Rationale:**
- Handles tabular data natively and efficiently — our feature set is entirely tabular.
- Outperforms random forest on most tabular benchmarks at this scale.
- Faster to train than XGBoost with comparable accuracy.
- `is_unbalance=True` handles class imbalance (bots are a minority) without manual resampling.
- Feature importance is built-in and interpretable — critical for debugging a competition system.
- Resistant to feature scaling differences (tree-based model, no normalization needed).

**Trade-offs:**
- More hyperparameters to tune than logistic regression.
- Can overfit on small datasets — mitigated by conservative `n_estimators` and cross-batch validation.
- Neural approaches might find non-linear patterns we miss, but require far more data to generalize.

---

## ADR-003 — Behavioral and profile features only (no NLP)

**Status:** Accepted

**Context:**  
Tweet text could be used for sentiment analysis, topic modeling, or fine-tuned language model embeddings (e.g., via a pre-trained BERT model).

**Decision:**  
Use only behavioral signals and surface-level text statistics. Do not use semantic NLP (no embeddings, no topic models, no sentiment).

**Rationale:**
- The competition dataset description explicitly notes: *"Bots can mimic text content but struggle to mimic irregular timing and natural randomness."* This is the key insight we build on.
- Modern bots using LLMs can produce perfectly natural-sounding text. Semantic NLP would be arms-racing against the bot's language model.
- Behavioral signals (posting intervals, burst patterns, duplicate ratios) are much harder to fake without breaking the bot's operational goals.
- Simpler features = less overfitting risk on a small dataset.
- No external model dependencies = faster iteration and more reliable final submission.

**Trade-offs:**
- We may miss bot types that are detectable primarily through linguistic patterns (e.g., bots with a very restricted vocabulary even if timing looks human).
- `duplicate_tweet_ratio` and `avg_cosine_similarity` are partial text features, but they capture repetition behavior rather than semantics.

---

## ADR-004 — Asymmetric scoring drives threshold selection

**Status:** Accepted

**Context:**  
The competition scoring function is `2*TP - 2*FN - 6*FP`. A standard 0.50 decision threshold optimizes accuracy but not this score. The 6× FP penalty means flagging a human is catastrophically worse than missing a bot.

**Decision:**  
Search decision thresholds from 0.50 to 0.95 and select the one that maximizes the competition score on validation data. Use a high threshold (typically 0.80–0.90+) for the final model.

**Rationale:**
- At the default 0.50 threshold, the model flags too many users as bots, generating FPs that destroy the score.
- The break-even point: flagging an extra user as a bot is only beneficial if `P(bot | flagged) > 6/8 = 0.75`. Below that precision, flagging hurts you.
- Threshold search is cheap (no retraining required) and directly optimizes the metric we're graded on.

**Trade-offs:**
- A very high threshold increases false negatives (missed bots). Each FN costs -2 points. We accept missing some bots to avoid the -6 FP penalty.
- Threshold tuned on validation may not perfectly transfer to the final eval distribution.

---

## ADR-005 — Conservative rules engine as a safety layer

**Status:** Accepted

**Context:**  
The ML model may miss edge cases where bot behavior is so extreme it should be caught regardless of the learned threshold.

**Decision:**  
Add a deterministic rules engine (`src/rules_engine.py`) that runs in parallel with the ML model. A user flagged by either the model OR the rules is included in the submission.

**Rationale:**
- The rules target cases with extreme feature values where no reasonable threshold would produce a false positive (e.g., posting 25 tweets in 10 minutes, >30% duplicate tweets).
- Acts as a floor: even if the model is poorly calibrated on a new batch, obvious bots are still caught.
- Rules are interpretable and easy to audit — important for understanding false positives if they occur.
- The OR combination is conservative: it can only add flagged users, not remove them.

**Trade-offs:**
- Rules are hand-crafted and may not generalize — thresholds (e.g., `z_score > 2.5`) were set before seeing the data and may need recalibration.
- The OR combination means rules can introduce FPs even when the ML model correctly scores a user below the threshold. Rules must be conservative enough that their precision is near 100%.

**Rules as of initial implementation:**
```python
z_score > 2.5 AND cv_time_delta < 0.4     # high-volume + suspiciously regular timing
duplicate_tweet_ratio > 0.30               # > 30% identical tweets
avg_cosine_similarity > 0.90              # near-identical content across tweets
hour_entropy < 0.25 AND tweets_per_day > 20  # active but only in one hour
max_tweets_in_10min > 25                   # extreme burst
```

---

## ADR-006 — Separate models for English and French

**Status:** Accepted

**Context:**  
The dataset contains both English (batches 1, 3, 5) and French (batches 2, 4, 6) data. We could train one multilingual model or two language-specific models.

**Decision:**  
Train separate models for each language with language-specific hyperparameters.

**Rationale:**
- The competition FAQ explicitly states: *"From our past experience, effective bot detectors are quite different between languages."*
- Bot strategies, posting patterns, and text statistics may differ significantly between EN and FR corpora.
- A single model trained on both would need to learn two distributions simultaneously, likely performing worse on each than a specialized model.
- Language-specific TF-IDF (used for cosine similarity features) requires a separate vocabulary per language anyway.
- Independent thresholds can be tuned per language, which is important if bot density differs.

**Trade-offs:**
- Half the training data per model — more risk of overfitting.
- Double the training, tuning, and maintenance effort.
- If bot strategies turn out to be very similar across languages, we get no benefit from the split.

---

## ADR-007 — Cross-batch validation as the primary evaluation strategy

**Status:** Accepted

**Context:**  
Standard train/test split within a single batch risks optimistic validation — if batch-level quirks are learned, the model may fail on the unseen final eval batch.

**Decision:**  
Use leave-one-batch-out cross-validation (train on N-1 batches, validate on the held-out batch) as the primary metric for model and threshold selection.

**Rationale:**
- The final eval is a new batch with potentially different user populations and bot strategies.
- Leave-one-out simulates the exact deployment condition: train on everything seen, predict on something unseen.
- Score variance across folds tells us how robust the model is — high variance is a warning signal.
- This is stricter than random splits within a batch, which would be overly optimistic.

**Trade-offs:**
- With only 3 batches per language, leave-one-out gives 3 validation estimates — limited statistical power.
- The held-out batch may still not represent the final eval if new bot strategies are introduced (the competition states a few new algorithms will appear).
- Training the final model on all batches means we have no out-of-sample score estimate for the final submission.

---

## ADR-008 — No social graph features

**Status:** Accepted (forced)

**Context:**  
Social graph signals (follower count, retweet networks, who follows whom) are among the strongest bot detection signals used in academic literature.

**Decision:**  
Do not use any social graph features. Use only per-user behavioral and profile signals.

**Rationale:**
- The dataset does not include social graph data. No followers, no retweets, no engagement metrics are provided.
- This is explicitly noted in the dataset description: *"No followers, likes, retweets, engagement metrics."*
- This decision is not a choice — it is a hard constraint from the data.

**Impact:**
- We are limited to pure behavioral signals (timing, text patterns) and profile metadata.
- This makes our task harder but also more interesting — behavioral signals alone must carry the full weight of detection.
- It also means our detector is more privacy-preserving: it does not require access to social graphs.

---

## Decisions under consideration

| Decision | Options | Status |
|----------|---------|--------|
| Ensemble across batches | Train one model per batch, vote | Under consideration |
| Feature selection | Automatic (SHAP), manual pruning | To explore in phase 4 |
| FR-specific features | FR-specific stopwords, accent features | To explore in phase 4 |
| Calibration | Platt scaling on probabilities | Low priority |

---

*Last updated: March 30, 2026*
