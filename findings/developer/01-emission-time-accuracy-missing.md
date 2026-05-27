# developer/01 — "Emission-time accuracy" log line not emitted by refactored runner

**Persona:** Developer
**Severity:** low (cosmetic; doesn't block training or analyst handoff)
**Run:** multipersona e2e 2026-05-27c, catalog 95

## What was promised

The Developer-arc launch prompt for this run says (verbatim):

> `predict_batch` + `record_predictions` split — predictions are emitted ONCE,
> with a `source_label` (e.g. `"epoch_10"`) tagged into both the CSV and
> surfaced as an "Emission-time accuracy: NN.NN%" log line.

The promise: a log line of the form `Emission-time accuracy: NN.NN%` printed
when `record_predictions` writes the CSV. This line is what the Analyst arc was
told to reconcile against the CSV "so the previous-run's '42% in log vs
34.67% in CSV' panic shouldn't repeat".

## What actually happens

`src/models/cifar10_cnn.py` (PR #37, this worktree) prints:

```
  Recorded {len(feature_records)} predictions (source_label={source_label!r})
```

at `record_predictions:424-427`. **No accuracy figure is printed at emission
time.** A `grep -rn "Emission" .` across the whole repo returns nothing.

Reproduced in all three Developer-arc runs:

| Exec | Final-epoch train_acc | Final-epoch test_acc | Emission-time accuracy log line |
|------|----------------------:|---------------------:|---------------------------------|
| XRJ  | 30.25%                | 24.00%               | absent                          |
| Y1M  | 58.83%                | 38.00%               | absent                          |
| YDT  | 65.00%                | 36.67%               | absent                          |

## Why the analyst safety rail matters

The reconciliation concern is real: training-loop `test_acc` is computed by
`evaluate()` (argmax → match) and the CSV's `Predicted_Class` is also
argmax-based, so they SHOULD agree. But they're produced in two separate
`predict_batch` calls (one inside `evaluate` for the running metric, one
explicit before `record_predictions`) and both rely on the model being in
the same state. Anything that touches the model state between
`evaluate(test_loader)` at the end of the training loop and the
`predict_batch` call at line 724 will silently desync them.

Adding `print(f"  Emission-time accuracy: {acc:.2f}%")` inside
`record_predictions` (one line, ~5 LOC including the labeled-row count) gives
the analyst exactly the redundant channel they need to spot a drift.

## Proposed fix (not done here — out of scope for the arc)

In `record_predictions`, compute accuracy directly from `predictions`:

```python
labeled = [p for p in predictions if p.get("ground_truth") is not None]
if labeled:
    correct = sum(1 for p in labeled if p["predicted_class"] == p["ground_truth"])
    print(f"  Emission-time accuracy: {100 * correct / len(labeled):.2f}% "
          f"(n={len(labeled)}, source_label={source_label!r})")
```

`predict_batch` would need to include `ground_truth` in each prediction dict
for this to work (currently it omits the label). For an unlabeled test bag
this branch is skipped — same shape as `evaluate`'s `n_labeled` guard.

## Workaround used in this arc

For now I'm flagging emission-time accuracy as "computed downstream from the
CSV by the analyst" — the CSV carries `Source_Label` + `Predicted_Class` +
per-class probabilities; the analyst can join against ground truth via
`Image_Classification` features.

## Tangential observation (not a separate finding)

`Execution.description` for YDT shows as `"Simple model run"` while XRJ and
Y1M show `"Quick CIFAR-10 training: 3 epochs, ..."`. The difference is that
XRJ and Y1M used `+experiment=cifar10_quick` (which carries a `description`
field on the experiment config), while YDT used a bare
`datasets=cifar10_train_with_validation model_config=default_model` override
chain with no experiment wrapper, so deriva-ml falls back to the
DerivaModelConfig default. Worth knowing — not broken, just worth a doc note
that "if you compose Hydra overrides instead of using `+experiment=...`, your
execution gets a generic description".
