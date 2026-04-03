# Bot or Not - Project Tracker

**Team:** SignalGuard  
**Competition deadline:** April 4, 2026 at 1:00 PM EST  
**Final eval drops:** April 4, 2026 at 12:00 PM EST

> Track progress by checking off tasks as you complete them. Phases are listed in the order they were completed.

---

## Progress overview

| Phase | Title | Priority | Status |
|-------|-------|----------|--------|
| 1 | Data validation and schema alignment | Critical | `[x]` |
| 2 | Baseline model and sanity check | High | `[x]` |
| 3 | Cross-batch validation and generalization | High | `[x]` |
| 4 | Feature engineering improvements | Medium | `[x]` |
| 5 | Threshold tuning and rules engine refinement | High | `[x]` |
| 6 | Final submission prep | Deadline | `[ ]` |

---

## Phase 1 - Data validation and schema alignment

### Understand the actual JSON structure

- [x] Load one dataset JSON and inspect top-level keys
- [x] Check user objects and confirm `id`, `username`, `name`, `description`, and `location`
- [x] Confirm `author_id` in posts matches user `id`
- [x] Check where `z_score` lives in the schema
- [x] Run `python test_everything.py`

### Fix code mismatches

- [x] Align `FeatureExtractor` with the confirmed schema
- [x] Verify user ID types are consistent across JSON and label files

---

## Phase 2 - Baseline model and sanity check

### First training run - English

- [x] Merge EN batches and run the baseline pipeline
- [x] Inspect feature importance
- [x] Validate on a held-out English batch

### First training run - French

- [x] Merge FR batches and run the baseline pipeline
- [x] Validate on a held-out French batch

---

## Phase 3 - Cross-batch validation and generalization

### Leave-one-batch-out validation

- [x] Implement leave-one-batch-out validation for English
- [x] Implement leave-one-batch-out validation for French
- [x] Record per-batch score variance

### Diagnose and fix overfitting

- [x] Adjust LightGBM regularization conservatively
- [x] Drop dead or near-constant features
- [x] Train the final language models on all available labeled batches

---

## Phase 4 - Feature engineering improvements

### New features added

- [x] Inter-post interval regularity features
- [x] Burst detection features
- [x] Template similarity and repetition features
- [x] Hour-of-day distribution features
- [x] Cross-user repost behavior
- [x] Accent density for French
- [x] Successive delay ratio and minimum rolling CV

### Feature hygiene

- [x] Remove or flag near-zero variance features
- [x] Add agreement checks between observed posts and declared `tweet_count`

---

## Phase 5 - Threshold tuning and rules engine refinement

### Threshold optimization

- [x] Tune thresholds using cross-batch validation
- [x] Use separate thresholds for English and French
- [x] Save the selected thresholds to `models/`

### Rules engine

- [x] Validate rules against real training data
- [x] Keep only conservative rules that help on cross-batch validation
- [x] Remove redundant or unvalidated late-stage rule ideas from the shipped logic

---

## Phase 6 - Final submission prep

### Before April 4

- [x] Train final models on all 6 labeled batches
- [x] Save final model and threshold artifacts
- [x] Run `python final_submission.py` end to end on training data as a dry run
- [x] Confirm `TEAM_NAME` is set correctly in `final_submission.py`
- [x] Update `README.md` to reflect the final shipped approach
- [ ] Make the GitHub repo accessible to the organizers

### During the 1-hour window

- [ ] Receive `final_eval_en.json` and `final_eval_fr.json`
- [ ] Place them in `data/final_eval/`
- [ ] Run `python final_submission.py`
- [ ] Sanity-check the generated output counts
- [ ] Email the submission files with the repo link before 1:00 PM EST

---

## Final validated state

- English cross-batch score: `188`
- French cross-batch score: `80`
- English threshold artifact: about `0.54`
- French threshold artifact: about `0.6723`
- Submission dry run: completed successfully

---

## Archived French-model follow-up notes

These notes absorb the older standalone French improvement document so the repo keeps one tracker for both shipped work and deferred follow-up ideas.

The final validated setup stayed conservative because the shipped French pipeline already produced an acceptable cross-batch score with the asymmetric FP penalty. The ideas below are useful only if French performance needs to be revisited after the competition or in a future branch.

### Deferred follow-up sequence

1. Widen French threshold search below `0.50` and retrain to see whether recall gains outweigh FP cost under the competition score.
2. Profile French false negatives to separate near-threshold misses from users the model is not recognizing at all.
3. Recalibrate French rules only if they remain inactive on real training folds after re-validation.
4. Add French-targeted features only if the diagnostic still shows many missed bots with very low model probabilities.

### Deferred ideas worth re-checking

- Lower-bound threshold search for French because the best operating point may sit below the default search floor.
- One-off diagnostics for missed French bots to inspect probability bands, duplicate behavior, burst patterns, and low-volume accounts.
- French-specific rule recalibration for repetition and burst combinations if rules show zero useful coverage.
- Extra French-targeted signals such as stronger periodicity and vocabulary-concentration features if the current feature set is blind to those bots.
- FR-specific LightGBM tuning only if it improves leave-one-batch-out validation without increasing false positives too much.

### Acceptance guardrails for any future FR changes

- Keep leave-one-batch-out validation as the gate, not evaluation-sample intuition.
- Accept changes only if French score improves without creating a negative held-out fold.
- Treat false-positive growth as the main failure mode because of the `-6 * FP` penalty.
- Leave the English model untouched unless there is separate evidence it regressed.

---

## Scoring reminder

```text
score = 2 x TP - 2 x FN - 6 x FP
```

False positives are penalized 3x more than false negatives. When in doubt, prefer the validated conservative setup.

---

*Last updated: April 3, 2026*
