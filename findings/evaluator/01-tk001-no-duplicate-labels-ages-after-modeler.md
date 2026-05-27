# evaluator/01 — tk-001's "no duplicate labels, no need for `newest` selector" ages out as soon as the Modeler runs

**Severity:** Low
**Category:** Skill issue (`capture-tacit-knowledge`) / Doc gap
**Component:** `tacit-knowledge.md` semantics; `capture-tacit-knowledge` skill guidance
**Filed by:** Evaluator (2026-05-27d run)

## Summary

The Curator's tk-001 audit entry asserts (verified true *at audit time*):

> 1500 rows in `Execution_Image_Image_Classification` covering 1500
> distinct images — no missing labels, no duplicate labels (no need
> for the `newest` selector when reading the feature).

This is **correct on a freshly-bootstrapped catalog** and the Curator's
direct query backs it up. But the `Image_Classification` feature is the
*same* table the Modeler's `record_test_predictions` writes prediction
rows into — so the instant the Modeler runs even one training execution,
the table contains both GT rows (`Confidence IS NULL`, written by loader
exec `FZC`) and prediction rows (`Confidence` populated, written by the
training execution). After this run there are **1800 rows / 1500 distinct
images / 100 images with 4 rows each** (FZC + XDP + XPR + XZT).

A reader who consumes tk-001 in good faith and writes
`ml.feature_values("Image", "Image_Classification")` without a filter
gets 1800 rows back, not 1500. The "no need for `newest` selector"
guidance is silently wrong in any post-Modeler state of the catalog.

The Analyst caught this — `analysis-scratch/rank_runs.py:41` filters
`gt_df[gt_df["Confidence"].isna()]` — but the catch happened by competent
reading of the prediction CSV / GT alignment, not because tk-001 warned
them.

## Direct vs claim

Direct query (this run, post-Modeler):
```
Total Image_Classification feature rows: 1800
Distinct Image RIDs with feature: 1500
Images with >1 label rows: 100 (4 rows each: FZC + XDP + XPR + XZT)
Example: image 482 → automobile (FZC, Conf=None), airplane (XDP, 0.25),
                     truck (XPR, 0.46), truck (XZT, 0.99)
```

tk-001's "no duplicate labels" reading is what a `deriva-ml` query
returns when scoped to `Execution == FZC`. It is not what a naive
unfiltered query returns.

## Why this is a finding

The `capture-tacit-knowledge` skill says tacit-knowledge entries should
capture rationale that **doesn't age**. tk-001's claim ages on first
write to the catalog after audit. The entry would be sounder if it
either:

- Scoped the claim explicitly to GT-only rows ("the loader-execution
  rows form a clean 1500-of-1500 GT layer"), or
- Recorded the *convention* ("`Image_Classification` is dual-purpose:
  loader-execution rows are GT, training-execution rows are predictions —
  always filter by execution or by `Confidence IS NULL` when treating it
  as GT") rather than the snapshot fact.

This is the convention-vs-snapshot distinction the skill calls out
explicitly ("convention entries are especially valuable when the
convention isn't otherwise documented"). The convention here — that GT
and predictions share a feature table — *is* tacit knowledge worth
preserving. The 1500-row count is not.

## What I'm NOT claiming

- I'm not saying the Curator's audit was wrong; they verified what they
  saw, at the time they saw it.
- I'm not saying the Analyst missed it; they didn't.
- I'm not saying this caused a wrong analysis result; the Analyst's
  numbers verify bit-for-bit (XDP 24%, XPR 38%, XZT 41%).

The finding is about *durability of the entry as documentation* — i.e.
the test §3.3 of the evaluator rubric explicitly asks.

## Suggested disposition

- **Defer.** Adjust the framing in the `capture-tacit-knowledge` skill's
  "convention entries" example to use the GT-vs-predictions shared-feature
  pattern as a worked example of snapshot-vs-convention. Or
- **Fix inline.** Edit the next Curator persona's framing prompt to call
  out "if the catalog will be modified after your audit, frame numerical
  claims as conventions or scope them to a snapshot, not as durable
  facts."

Either way, low priority — the run completed correctly because the
Analyst caught it independently.
