# Committed `prediction_probabilities.csv` doesn't reflect final-epoch model state

**Persona:** Analyst
**Phase:** §1 — Ranking executions by accuracy
**Severity:** Medium (provenance gap; affects what a downstream consumer thinks they're getting)
**Component:** `deriva-ml-model-template/src/models/cifar10_cnn.py` runner

## What happened

The Developer's tk-004 records final test accuracy from each
training run's log:

- XYG (`cifar10_default`, 10 epochs, seed 123): **42.00%**
- YAP (`cifar10_regularized`, 10 epochs, seed 2026): **37.33%**
- XNE (`cifar10_quick`, 3 epochs, seed 42): **24.00%**

The Analyst recomputed accuracy directly from each run's committed
`prediction_probabilities.csv` against the
`Image_Classification` ground-truth feature (`Confidence IS NULL`,
execution FZC):

- XYG (Y0E): **34.67%**  (delta -7.33%)
- YAP (YCP): **36.00%**  (delta -1.33%)
- XNE (XQC): **24.00%**  (delta  0.00%)

The CSV accuracy for XYG matches XYG's epoch-4 / epoch-6 test
accuracy in the log (34.67%), *not* the final epoch-10 (42.00%).
For YAP, the CSV accuracy (36.00%) matches epoch-5 or epoch-9 in
its log, not the final epoch (37.33%). The committed predictions
are not from the final-epoch model state.

## Reproduction

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/analyst_rank_executions.py
```

(Or any equivalent: download XYG's `prediction_probabilities.csv`
asset Y0E, join `Image_RID` against the GT feature's `Image_Class`,
compare `Predicted_Class == True_Class`.)

The exact-match for XNE (3 epochs, 24%) is suggestive — short runs
where training accuracy never overshoots the final test accuracy
match. Longer runs where later epochs improve over earlier ones
diverge.

## Impact on the persona's work

The Analyst cannot honestly tell a downstream consumer "YAP got
37.33% test accuracy" — the *reproducible* number from the asset
they would download is 36.00%. The ranking direction is the same
(YAP > XYG > XNE) under either metric, so the deliverable is
unaffected; the gap is the missing piece of provenance: *which model
state are the committed predictions from?*

Cost ~10 minutes of "wait, am I joining wrong?" before I had enough
evidence (XNE exact, others off by an amount that matches earlier
epochs) to be confident this is the runner's choice, not an Analyst
bug.

## Suggested classification

Bug or Missing feature (depends on intent). If the runner is doing
`--save-best` and emitting predictions from the best-test-acc
checkpoint, that's a sensible choice but it should be (a) stated in
the training log, and (b) included as a column in the prediction
CSV (`source_epoch=4` or similar) so the Analyst can correlate.

## Notes for the fix-pass

Three options, in increasing order of effort:

1. **Document the current behavior.** Add a "predictions are from
   model state at epoch N (where N is the best test_acc epoch)" line
   to the training log. One-line fix in the runner if it's
   `--save-best` already.
2. **Surface in the prediction CSV.** Add a `source_epoch` column to
   `prediction_probabilities.csv`. Two-line change.
3. **Make it a config choice.** Add `predictions_from: {best, final}`
   to the model config; default `best`. Cleanest but requires a
   config-key addition.

A first-pass investigation should check `src/models/cifar10_cnn.py`
for the prediction-emission code path — specifically what state the
model is in when the CSV is written. The Developer's tk-004 doesn't
call this out, suggesting the runner's behavior is invisible from
the training-log surface.
