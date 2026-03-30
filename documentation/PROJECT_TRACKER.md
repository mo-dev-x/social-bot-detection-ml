# Bot or Not — Project Tracker

**Team:** SignalGuard  
**Competition deadline:** April 4, 2026 at 1:00 PM EST  
**Final eval drops:** April 4 at 12:00 PM EST (1 hour to submit)

> Track progress by checking off tasks as you complete them. Phases must be done roughly in order — phase 1 unblocks everything else.

---

## Progress overview

| Phase | Title | Priority | Status |
|-------|-------|----------|--------|
| 1 | Data validation & schema alignment | 🔴 Critical | `[x]` |
| 2 | Baseline model & sanity check | 🟠 High | `[x]` |
| 3 | Cross-batch validation & generalization | 🟠 High | `[x]` |
| 4 | Feature engineering improvements | 🟡 Medium | `[x]` |
| 5 | Threshold tuning & rules engine refinement | 🟠 High | `[x]` |
| 6 | Final submission prep | 🔴 Deadline | `[ ]` |

---

## Phase 1 — Data validation & schema alignment

> **Do this first — it unblocks everything else.**  
> Your code was written before seeing the actual data. Several assumptions need verification.

### Understand the actual JSON structure

- [x] Load one dataset JSON and inspect top-level keys
  - Confirm: `users` list, `posts` list, `metadata` fields all present
- [x] Check user objects — do `id`, `username`, `name`, `description`, `location` actually exist?
  - Your profile features depend on this; a mismatch causes silent zero fallback
- [x] Confirm `author_id` in posts matches user `id` format (UUID strings throughout)
  - Some code paths do `int(line.strip())` — risk of silent mismatch
- [x] Check `z_score` field — is it per-user in the user object, or only in `metadata`?
  - Currently assumed to live in the user object
- [x] Run `python test_everything.py` and fix any import or path errors

### Fix code mismatches

- [x] Align `FeatureExtractor` with confirmed schema
  - Especially `_extract_profile_features` and `_extract_activity_features`
- [x] Verify user_id types are consistent (UUID strings everywhere, not ints)
  - The bot label files use UUID strings like `60f368e7-...`, not integers

---

## Phase 2 — Baseline model & sanity check

> Train once on real data, see what fires, get a first competition score.

### First training run — English

- [x] Merge all 3 EN datasets (batches 1, 3, 5) into one training set
  - Keep batch 5 aside as hold-out for cross-batch validation
- [x] Run `train_full_pipeline` on EN data
- [x] Check feature importance — which features actually drive predictions?
  - If `z_score` dominates everything, investigate whether it leaks label info
- [x] Validate on held-out batch 5 and compute competition score (`2*TP - 2*FN - 6*FP`)

### First training run — French

- [x] Merge batches 2 and 4, hold out batch 6
- [x] Run `train_full_pipeline` on FR data
- [x] Validate on batch 6 and record score

---

## Phase 3 — Cross-batch validation & generalization

> The final eval is unseen data. Your biggest risk is a model that memorized one batch's bot patterns.

### Leave-one-batch-out validation

- [x] Implement leave-one-out CV loop across EN batches
  - Train on 2 batches, validate on 1 — cycle all 3 combinations
- [x] Same for FR batches (batches 2, 4, 6)
- [x] Record per-batch score variance
  - High variance = fragile model; low variance = good generalization

### Diagnose and fix overfitting

- [x] If score variance is high: reduce `n_estimators`, increase `min_child_samples`
  - Small dataset makes it easy to overfit
- [x] Drop any feature that leaks batch-specific info
  - e.g. absolute timestamp values if batches span different date ranges
- [x] Try training on ALL 3 batches combined for the final model
  - More data usually beats withheld tuning data at this scale

---

## Phase 4 — Feature engineering improvements

> High impact if time allows. Skip if deadline pressure is high.

### New features to add

- [x] Inter-post interval regularity: variance of gaps between consecutive posts
  - Bots often post at machine-regular intervals
- [x] Burst detection: ratio of posts in the top-10% densest time windows
  - Bots tend to burst then go silent
- [x] Template similarity: cluster tweets by structure, not just content
  - Many bots fill in a fixed template (same sentence frame, different nouns)
- [x] Hour-of-day distribution: chi-squared test against uniform distribution
  - Bots often cluster in narrow hour windows
- [x] Fraction of posts that are exact re-posts of other users in the same dataset
  - Coordinated inauthentic behavior signal

### Feature hygiene

- [x] Remove or flag features with near-zero variance across all users
  - Dead features hurt generalization without adding signal
- [x] Add feature: agreement between actual post count and `tweet_count` field
  - A discrepancy between observed posts and the declared count could be a bot signal

---

## Phase 5 — Threshold tuning & rules engine refinement

> This directly controls your competition score. The 6× FP penalty means a high threshold is almost always correct.

### Threshold optimization

- [x] Expand threshold search grid to `0.50–0.95` in `0.05` steps
  - Current search stops at `0.90`
- [x] Use cross-batch validation score (not a single split) to pick the threshold
  - Avoids overfitting the threshold to one batch's distribution
- [x] Consider separate thresholds for EN and FR
  - The two languages may have different bot density and feature distributions

### Rules engine

- [x] Validate each rule against actual data — do they fire on real bots?
  - Rules were written before seeing the data and may be miscalibrated
- [x] Add rule: flag if `near_duplicate_ratio > 0.5` AND `tweet_count > 20`
  - High-volume copypasta is a strong bot pattern
- [x] Tune the `z_score` threshold in the rules — check the actual distribution
  - The current threshold of `2.5` may be too loose or too tight for this dataset

---

## Phase 6 — Final submission prep

> April 4th — you have exactly 1 hour. Everything here should be done the day before.

### Before April 4

- [x] Train final models on ALL available labeled data (all 6 batches)
  - Don't hold out anything for the final model — use every label
- [x] Save final model and threshold to `models/` directory
- [ ] Run `python final_submission.py` end-to-end on a training batch as a dry run
  - Do not discover bugs during the 1-hour window
- [ ] Confirm `TEAM_NAME` is set correctly in `final_submission.py`
- [ ] Write or update `README.md` explaining your approach
  - Required per competition rules; mention any EN/FR differences
- [ ] Make the GitHub repo accessible to the organizers

### During the 1-hour window (April 4, 12:00–1:00 PM EST)

- [ ] Receive `final_eval_en.json` and `final_eval_fr.json` at noon
  - Drop them into `data/final_eval/`
- [ ] Run `python final_submission.py`
  - Should complete in under 5 minutes
- [ ] Sanity-check output: does the flagged ID count seem reasonable?
  - If 0 or 500+, something has gone wrong
- [ ] Email to `bot.or.not.competition.adm@gmail.com` with your GitHub repo link
  - Must arrive before 1:00 PM EST

---

## Scoring reminder

```
score = 2 × TP  −  2 × FN  −  6 × FP
```

False positives are penalized 3× more than false negatives. When in doubt, **raise the threshold**.

---

*Last updated: March 30, 2026*
