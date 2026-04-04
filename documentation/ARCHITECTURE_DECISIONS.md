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
- Uses conservative imbalance handling: the French model applies `class_weight="balanced"` while the English model relies on careful threshold tuning and cross-batch validation.
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
- `duplicate_tweet_ratio` and `avg_similarity_between_tweets` are partial text features, but they capture repetition behavior rather than semantics.

---

## ADR-004 — Asymmetric scoring drives threshold selection

**Status:** Accepted

**Context:**  
The competition scoring function is `2*TP - 2*FN - 6*FP`. A standard 0.50 decision threshold optimizes accuracy but not this score. The 6× FP penalty means flagging a human is catastrophically worse than missing a bot.

**Decision:**  
Search decision thresholds from 0.50 to 0.99 for English, and from 0.30 to 0.99 for French, then select the one that maximizes the competition score on validation data. Use language-specific thresholds chosen from cross-batch out-of-fold predictions.
**Rationale:**
- At the default 0.50 threshold, the model flags too many users as bots, generating FPs that destroy the score.
- The break-even point: flagging an extra user as a bot is only beneficial if `P(bot | flagged) > 6/8 = 0.75`. Below that precision, flagging hurts you.
- Threshold search is cheap (no retraining required) and directly optimizes the metric we're graded on.

**Trade-offs:**
- Higher thresholds increase false negatives (missed bots). Each FN costs -2 points. We accept missing some bots to avoid the -6 FP penalty.
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

**Rules as of current implementation:**
```python
duplicate_burst_rule = (
    near_duplicate_ratio >= 0.50
    and tweet_count >= 20
    and burst_ratio_1h >= 0.35
)
coordinated_repost_rule = (
    cross_user_repost_ratio >= 0.60
    and template_duplicate_ratio >= 0.45
    and hour_uniform_chi2 >= 18.0
)
regularity_rule = (
    duplicate_tweet_ratio >= 0.45
    and cv_time_delta <= 0.12
    and hour_entropy <= 1.2
    and tweet_count >= 15
)
zscore_rule = (
    z_score >= 2.0
    and near_duplicate_ratio >= 0.35
    and tweets_per_hour >= 1.5
)

# Additional French-specific rules:
fr_periodic_campaign_rule = (
    periodic_interval_ratio >= 0.85
    and avg_similarity_between_tweets <= 0.07
)
fr_low_periodicity_volume_rule = (
    periodic_interval_ratio <= 0.02
    and tweet_count >= 18
)
```

The current rules engine is intentionally conservative, using extreme combination thresholds rather than broad single-feature cuts.

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

## ADR-007A — Small seed ensembles for score stability

**Status:** Accepted

**Context:**  
Single LightGBM runs were still producing a few very high-confidence false positives, especially in French. The data is small enough that seed variance matters.

**Decision:**  
Train a 3-seed LightGBM ensemble per language and average probabilities at validation and inference time.

**Rationale:**
- Averaging across seeds smooths out unstable tree splits without changing the feature interface or submission workflow.
- The ensemble improved cross-batch performance in both languages while keeping inference simple.
- This is a lower-risk stability improvement than changing the safety-layer semantics.

**Trade-offs:**
- Training and inference are roughly 3x heavier.
- Saved model artifacts now contain a list of models rather than a single estimator.

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

## Late-stage validation notes

The following ideas were tested or discussed late in the project and intentionally not shipped in the final detector:

- Weighted-fusion safety layers and probability-adjustment schemes replacing the current `model OR rules` decision.
- Human-profile veto logic such as "rich profile" exceptions for long bios and high hour entropy.
- DART-based LightGBM swaps intended to improve generalization.
- Topic-focus rules justified primarily by evaluation-sample observations rather than training-fold validation.
- Redundant French lexical rules that did not increase rule coverage in validation.

These were rejected because they either reduced cross-batch score, introduced recall collapse, or increased leakage risk relative to the validated training setup.

---

## Decisions under consideration

| Decision | Options | Status |
|----------|---------|--------|
| Batch-specific ensemble voting | Train one model per batch, vote | Deferred |
| Feature selection | Automatic (SHAP), manual pruning | Deferred |
| Calibration | Platt scaling on probabilities | Deferred |

---

*Last updated: April 3, 2026*
