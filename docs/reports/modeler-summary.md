# Modeler Summary — 2026-05-28 e2e run

**Persona:** Modeler | **Catalog:** `localhost` id 27 (`e2e-test-20260528`) |
**Worktree branch:** `e2e-test/2026-05-28`

## What I ran

Three training executions on a new leakage-free dataset group
(`cifar10_toronto_pair` = M16 train + M1G test, per Curator [tk-002](../../tacit-knowledge.md#tk-002)),
spanning underfit → reasonable → overfit hyperparameters:

| Execution | Experiment | Arch | Epochs | Batch | Final test_acc | Best test_acc |
|---|---|---|---|---|---|---|
| **W76** | `cifar10_toronto_quick`   | 32→64ch,128h  |  3 | 128 | **24.00 %** | 25.64 % (ep 2) |
| **XCE** | `cifar10_toronto_default` | 32→64ch,128h  | 10 |  64 | **37.82 %** | 38.18 % (ep 9) |
| **YHP** | `cifar10_toronto_large`   | 64→128ch,256h | 20 |  64 | **41.09 %** | 42.36 % (ep 13) |

All three trained on M16 (550 images, 55/class) and evaluated against
M1G (550 images, 55/class). seed=42, lr=1e-3, weight_decay=0. CPU.
Single-worker DataLoaders per the macOS gotcha.

## What landed in the catalog

Each execution committed three Execution_Assets (weights, training log,
prediction CSV). All status `Uploaded`. Asset RIDs:

| Execution | weights (.pt) | training_log.txt | prediction CSV |
|---|---|---|---|
| W76 | W92 | W94 | **W96** |
| XCE | XEA | XEC | **XEE** |
| YHP | YKJ | YKM | **YKP** |

Wired into `src/configs/assets.py` ([E2E-DROP]) as four new groups:
`toronto_predictions` (the three CSVs together, for the ROC notebook),
plus `toronto_quick_outputs` / `toronto_default_outputs` /
`toronto_large_outputs` (per-execution bundles).

Two `[E2E-DROP]` commits on `e2e-test/2026-05-28`:
- `711de51` — Toronto-pair dataset group + 3 experiment configs.
- (final commit) — assets.py wiring + tacit-knowledge entries
  [tk-004](../../tacit-knowledge.md#tk-004), [tk-005](../../tacit-knowledge.md#tk-005), [tk-006](../../tacit-knowledge.md#tk-006).

## What worked vs surprised me

**Worked cleanly.** Dry-run resolved on first try. All three real runs
committed weights + log + CSV without manual intervention. The
emission-time accuracy print in `record_predictions` matched the
final-epoch `test_acc` in `training_log.txt` exactly, closing the
provenance loop the comment block promised.

**Surprised me.** The committed prediction CSV is **final-epoch**, not
**best-epoch** — see [tk-006](../../tacit-knowledge.md#tk-006). YHP's
best test_acc (42.36 %, epoch 13) lives only in the log; the CSV
reflects the memorised epoch-20 state at 41.09 %. The validation lane
is wired into the training loop but doesn't drive save-best. Not a
bug; explicitly intentional (`# Validation lane wiring` block in
`cifar10_cnn.py`), but the Analyst should know.

## What the Analyst can use

**Test set:** M1G (Toronto testing, 550 images, 55/class, no train
leakage). **Metric:** any test-set comparison drawn from `Image_RID` ×
`Predicted_Class` × per-class probs in `W96` / `XEE` / `YKP`. **Ground
truth:** `Image_Classification` feature filtered to execution `HSR`
(per [tk-001](../../tacit-knowledge.md#tk-001) — the bare feature now
contains 4 executions of rows after my three training runs added
predictions). **Ranking by accuracy:** YHP > XCE > W76. The three runs
should give clearly separable ROC curves.
