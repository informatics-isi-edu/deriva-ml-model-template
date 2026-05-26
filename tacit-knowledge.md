# Tacit Knowledge

This file records **tacit knowledge** — the *why*, the *intent*, and the
*background* behind decisions made about this project's models and data.

The **catalog** is the source of record for everything else: data contents,
RIDs, dataset versions, workflow URLs and checksums, executions, lineage.
Don't replicate catalog-stored facts here. Don't ask this file what's in
the catalog — query the catalog directly (resources first, tools next).
When this file *needs* to reference a catalog entity, link to it
(`deriva://catalog/{host}/{cat}/ml/...`) instead of inlining its contents.

Each entry captures a decision: what was chosen, what alternatives were
considered, what was rejected and why, and any background context a future
reader would need to evaluate whether the decision still holds.

---

### tk-001 — Bootstrap: e2e-test-20260526 catalog 18 created for 3-persona multipersona run
**When:** 2026-05-26T10:55:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/ec40f483-26ae-4a8b-aa24-5155ca94cb22)

Created localhost catalog **id 18** (alias `e2e-test-20260526`) for the
2026-05-26 multipersona e2e run (Curator → Developer → Analyst).
Bootstrap was scripted via `load-cifar10` in three phases:

```
uv run load-cifar10 --hostname localhost --create-catalog e2e-test-20260526 --phase schema
uv run load-cifar10 --hostname localhost --catalog-id 18 --num-images 500 --phase images
uv run load-cifar10 --hostname localhost --catalog-id 18 --num-images 500 --phase datasets
```

500-image CIFAR-10 sample — `--num-images 500` chosen for fast iteration
across all three persona arcs; matches the prior 2026-05-25 run for
cross-run comparability. Class distribution is balanced 50-per-class
across all 10 CIFAR-10 classes (post-#15 fix, no bird+ship skew).

13 datasets created by the loader (Toronto split + small variant +
training-derived labeled split via `split_dataset(seed=42)` and a small
variant via `seed=123`):

| Group | Complete | Split | Training | Testing |
|---|---|---|---|---|
| Toronto (Complete=`96E`) | 96E | 96R | 970 | 97A |
| Toronto small | — | 97T | 982 | 98C |
| Training-derived labeled (seed=42) | — | C7Y | C86 | C8G |
| Training-derived labeled small (seed=123) | — | CRR | CS0 | CSA |

All RIDs wired into `src/configs/datasets.py` at v0.1.0.post1.dev1
(dev versions — release via `ml.lookup_dataset(rid).release(minor=True)`
if downstream wants pinned snapshots). `src/configs/deriva.py`
`default_deriva.catalog_id=18`. Both edits are `[E2E-DROP]` commits
on `e2e-test/2026-05-26` and drop out at wrap-up.

Cross-channel verification (direct deriva-ml + MCP `deriva_ml_*` tools)
agreed on dataset count (13), feature count (1: `Image_Classification`),
feature value count (500), and class distribution.

**Sibling versions for reconstructability:**
- deriva-ml v1.39.2 (39d88f39)
- deriva-ml-mcp v0.5.1 (5c0390a)
- deriva-mcp-core 376df57 (no tag — running 0.1.0 series HEAD)
- deriva-skills v1.2.3 (ecdb9ab)
- deriva-ml-skills v1.4.7 (de96fac) — includes tk-NNN/**When**/**By**/**Supported by** entry headers + semantic-awareness bridge
- Docker `deriva-mcp-test` container rebuilt 2026-05-26T17:34, runs deriva-ml 1.39.2 + deriva-ml-mcp 0.5.1.

**Implications for collaborators:** Curator inherits a clean catalog
with all built-in datasets, no synthetic/curated subsets, no model
training runs, no analysis assets. The Curator's first job is to audit
this state, decide which dataset(s) the Developer will actually train
on, and add at least one curated variant. Open questions left for the
Curator (not directives — questions to answer through their own work):
why is `default_dataset` pointed at `CRR` rather than `C7Y`? Is the
seed=42 vs seed=123 split duplication intentional or vestigial? Are
all 13 datasets going to be used, or are several of them dead branches
that should be pruned/deleted before downstream consumption?

---

### tk-002 — Curator audit of catalog 18: 13 bootstrap datasets, only 3 non-redundant content families
**When:** 2026-05-26T11:10:00-07:00
**By:** Curator persona (sub-agent, `e2e-test/2026-05-26` worktree)
**Supported by:** Cross-channel verification — direct `deriva-ml`
Python (`ml.find_datasets()`, `ml.lookup_dataset(rid).list_dataset_members()`)
agreed with MCP resources (`deriva://catalog/localhost/18/ml/datasets`,
`.../ml/dataset/{rid}/members`, `.../ml/features/Image`,
`mcp__dev-localhost__deriva_ml_list_feature_values`). All counts and
class distributions match across both channels — see
`findings/curator/02-*` for the one MCP-side discrepancy noted (cursor
serialization in `list_feature_values`).

Audit of the bootstrapped catalog before adding curated variants:

**Dataset content topology (post-bootstrap, 13 datasets):**

| Family | Parent | Train | Test | What it actually is |
|---|---|---|---|---|
| Toronto (`Complete`) | — | — | — | `96E` (500 imgs), union of 970+97A |
| Toronto split | `96R` | `970` (250) | `97A` (250) | Disjoint by Toronto train/test batches |
| Toronto small | `97T` | `982` (250) | `98C` (250) | **byte-identical to 970/97A — see finding 01** |
| Labeled split seed=42 | `C7Y` | `C86` (200) | `C8G` (50) | Stratified pick from 970; both subsets in the *training* pool |
| Labeled split seed=123 | `CRR` | `CS0` (200) | `CSA` (50) | Same shape as seed=42, different random pick (200-train ∩ = 161, 50-test ∩ = 11) |

**Image_Classification ground truth:** 500 feature values, perfectly
balanced 50-per-class across all 10 CIFAR-10 classes (airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, truck). Single producing
execution: `854` (the bootstrap loader). Coverage is 100% — every image
in 96E / 970 / 97A / C86 / C8G / CS0 / CSA has a label.

**Answers to the open questions left by tk-001:**

1. *Why is `default_dataset` pointed at `CRR` (small_labeled_split) rather
   than `C7Y` (full labeled split)?* — Reasonable choice for an e2e
   default: CRR is labeled, small (250 imgs total across train+test
   children), and pulled from the genuine `Image_Classification`
   ground truth. Fast enough for sub-arc smoke runs. Not a bug,
   just an opinionated default the Developer can override per
   experiment.
2. *Is the seed=42 vs seed=123 split duplication intentional or vestigial?* —
   **Intentional but not for the obvious reason.** The two splits draw
   from the same 250-image training pool; intersection on the 200-image
   train subset is 161 (i.e. 80% overlap), and on the 50-image test
   subset is 11 (i.e. ~22% overlap). The splits are *genuinely
   different*. Why two? The loader hard-codes the `Labeled_Split` at
   seed=42 (deterministic, reproducible) and the `Small_Labeled_Split`
   at seed=123 (the small variant) — the seed difference is the
   `_cifar10_datasets.py:369 vs :384` constants. Likely just to ensure
   the small and full labeled splits don't share identical members.
   Keep both for now; the seed=42 family (C86, C8G) is the "canonical"
   labeled split, seed=123 (CS0, CSA) is the "small variant" that
   downstream defaulted to.
3. *Are all 13 datasets going to be used, or are several dead branches?* —
   **Three families are effectively single-family at this image count.**
   The Toronto-small (`97T → 982/98C`) is byte-identical to the
   Toronto-full (`96R → 970/97A`) at `--num-images 500` (finding 01).
   That's not "dead" so much as "degenerate": code that picks "small"
   gets the same data as code that picks "full", so any
   small-vs-full comparison is silently invalid. For this e2e run,
   the Developer should treat `cifar10_small_*` as aliases for
   `cifar10_*`, not as a true subset.

**Loader description hygiene (finding 03):** Bootstrap descriptions on
970/97A/97T/982/98C reference 50,000/10,000/1,000/500/500 images
regardless of `--num-images`; the actual member counts are 250 each.
Misleading for any external reader. Doesn't affect this run because
this tacit-knowledge file carries the truthful counts; carries forward
to the loader description fix already tracked as pending task C03.

---

### tk-003 — Curator decisions: create `cifar10_validation_from_test` and `cifar10_balanced_demo`
**When:** 2026-05-26T11:12:00-07:00
**By:** Curator persona (sub-agent, `e2e-test/2026-05-26` worktree)
**Supported by:** `scripts/curator_create_datasets.py` (committed as
`[E2E-DROP]` in `ee8f97e`), execution `D9P` on catalog 18, cross-channel
verification post-creation (direct `deriva-ml` and MCP both report
15 datasets total; DAP types=[Validation,Labeled], DB0
types=[Testing,Labeled] with the expected member counts).

Two curated dataset variants added on top of the bootstrap to serve
downstream personas with clear motivations distinct from "exercise the
API":

**1. `cifar10_validation_from_test` — RID `DAP`, `Dataset_Type=[Validation, Labeled]`, 250 images, version `0.1.0.post1.dev1`.**
Same image RIDs as `97A` (`cifar10_testing`) — a *semantic relabeling*,
not a resample. **Why a relabel and not a resample?** Because 97A is
already disjoint from 970 (Toronto's train/test batches don't overlap
by construction), it's the natural held-out evaluator. The bootstrap
labels it `Testing` for historical reasons, but the cifar10_cnn runner
(pending task D01) wants a `Validation`-typed bag distinct from
in-pool C8G/CSA. DAP wears the right type so the Developer/Analyst
arcs can pick `cifar10_validation_from_test` from the configs and have
the runner accept it as held-out. **Why not just retype 97A?**
Preserves the original Toronto-test naming for users who reach for
"the testing set" — the new dataset row is the alias-with-intent, not
a replacement.

Alternatives weighed:
- *Stratified split off 970 into a fresh val subset* — rejected: would
  put val and train inside the same image pool, which is exactly the
  problem this curation move solves.
- *Use the existing C8G or CSA as validation* — rejected: those are
  drawn from 970 (training pool) and overlap with C86/CS0 (training
  subsets). Not a clean held-out.

**2. `cifar10_balanced_demo` — RID `DB0`, `Dataset_Type=[Testing, Labeled]`, 50 images, version `0.1.0.post1.dev1`.**
Stratified hand-pick from 96E: exactly 5 images per CIFAR-10 class
(50 total). Reproducible (`seed=20260526`, deterministic per-class
shuffle in `stratified_pick()`). Two intended uses:
- **Sub-minute smoke runs** for the Developer. The Toronto-small
  variant (982/98C) is degenerate at this image count (finding 01),
  so DB0 fills the "actually small" niche.
- **Guaranteed-populated confusion matrix cells** for the Analyst.
  With 5/class minimum, every CIFAR-10 class is represented even in
  the smallest analysis slice — relevant for ROC and confusion-matrix
  work that breaks down silently when a class has zero predictions.

Alternatives weighed:
- *Use `Validation` instead of `Testing` type* — rejected: DB0 is for
  smoke runs and demo plots, not for early-stopping selection.
  `Testing` matches the intent.
- *Smaller per-class count (1, 2, 3)* — rejected: 5/class gives enough
  variance for plotting; smaller counts collapse to single-point
  scatter that's harder to interpret visually.
- *Use the existing 13 dataset families instead* — rejected: none of
  them are simultaneously *small*, *balanced*, and *deterministic*. The
  closest existing match (C8G) is 50 images stratified from a 250-image
  source, so per-class counts are 5 — but it's drawn from training,
  failing the held-out requirement that an *eval-style* small set
  should have.

Both datasets share the same Curation Execution `D9P` for clean
lineage. Workflow `Dataset_Split`-typed (matched the closest existing
Workflow_Type term; would have created `Data_Curation` if it didn't
already exist, but no Workflow_Type extension was needed). Description
generated to be self-explanatory at the catalog level rather than
defer-to-tacit-knowledge — the Chaise viewer should make the rationale
discoverable from the row itself.

Both RIDs wired into `src/configs/datasets.py` as named entries
(`cifar10_validation_from_test` and `cifar10_balanced_demo`) so the
Developer / Analyst can pin them with Hydra overrides. The
configuration smoke test (`tests/test_configs_load.py`) still passes.

---

### Handoff to Developer

**What's ready for the Developer arc:**

- **Catalog 18 (alias `e2e-test-20260526`) on `localhost`** — 15 datasets
  total now (13 bootstrap + 2 curator-added). 500 Image rows with
  perfectly balanced 50-per-class `Image_Classification` ground truth.
- **`src/configs/deriva.py`** — `default_deriva.catalog_id=18` (set in
  bootstrap, `[E2E-DROP]`).
- **`src/configs/datasets.py`** — 13 bootstrap RIDs + 2 curator RIDs,
  all `.post1.dev1` dev versions. Pinable directly by name.

**The Developer's preferred datasets for this run (Curator's recommendation):**

| Use case | Pick | RID | Why |
|---|---|---|---|
| Training (small, labeled, in-pool seed=42) | `cifar10_labeled_training` | `C86` | 200 imgs, stratified from 970. The canonical seed=42 split. |
| Training (small, labeled, in-pool seed=123) | `cifar10_small_labeled_training` | `CS0` | 200 imgs, different random seed. Use both for variance estimation across seeds. |
| **Validation (held-out)** | **`cifar10_validation_from_test`** | **`DAP`** | **250 imgs, disjoint from training pool. Use this for early stopping / model selection. New from curator.** |
| Smoke test (very fast, 5/class balanced) | `cifar10_balanced_demo` | `DB0` | 50 imgs. Sub-minute training; not for real model selection. New from curator. |
| Test (final) | Pick `cifar10_labeled_testing` (`C8G`) OR `cifar10_small_labeled_testing` (`CSA`) | C8G / CSA | 50 imgs each. Stratified from training pool; **not held out from training**, so these are *in-distribution* test sets. Use DAP for the truly-held-out evaluation. |

**Pinned things the Developer should NOT change:**

- `default_deriva` and `default_dataset` config entries on the e2e
  branch — they're `[E2E-DROP]` commits. If you need a different default
  for a particular experiment, override at the CLI
  (`deriva-ml-run datasets=cifar10_balanced_demo`).
- Catalog 18 itself. The Developer is expected to add Executions and
  Output Assets, not new dataset rows. If the Developer wants additional
  curated splits, that's a Curator request — flag back instead of
  curating in-arc.

**Gotchas:**

- **Toronto-small variant is degenerate at this image count** (finding 01).
  `cifar10_small_training` is byte-identical to `cifar10_training`. Don't
  try to use the "small variant" for a real small-vs-full comparison;
  use `cifar10_balanced_demo` (DB0) for a genuine smaller sample.
- **`Image_Classification` feature value pagination via MCP is buggy**
  (finding 02). If the Developer needs to enumerate feature values via
  the MCP wire, use `execution_rids=` to scope or fall through to the
  direct `deriva-ml` Python path. The cursor (`next_after_rid`) comes
  back as `""` and can't be advanced.
- **Loader descriptions on bootstrap datasets are misleading** (finding 03).
  E.g., dataset `970` describes itself as "CIFAR-10 training set with
  50,000 labeled images" — actually has 250 at `--num-images 500`. The
  curator-added DAP and DB0 descriptions are accurate; the bootstrap
  ones are not. Don't quote bootstrap descriptions in any Developer-facing
  output without cross-checking.
- **All ground-truth labels were produced by a single execution (`854`).**
  When using `selector="by_execution"` on `deriva_ml_list_feature_values`,
  `854` is the only Image_Classification source. If the Developer's
  runs ADD prediction-style classification feature values, those'll
  share the `Image_Classification` feature name but come from a
  different execution — the selector lets you separate them.
- **All datasets are at `.post1.dev1` dev versions.** Developer doesn't
  need to release them — pinning by RID + version is enough for
  reproducibility within this run. If a stable label is needed before
  Analyst arc, call `ml.lookup_dataset(rid).release(minor=True)`. Not
  required.

**Findings raised (in `findings/curator/`):** 01 small-variant-equals-full,
02 feature-values-cursor-empty, 03 bootstrap-dataset-descriptions-misleading.
None block the Developer arc; all are catalog-observational, not
behavioral.

---

### tk-004 — Developer training arc: 7 executions across two variants + one lr_sweep multirun
**When:** 2026-05-26T11:25:00-07:00
**By:** Developer persona (sub-agent, `e2e-test/2026-05-26` worktree)
**Supported by:** Direct ermrest cross-check
(`PathBuilder('Execution').filter(RID==...).entities().fetch()`) on all
8 executions agrees with the MCP `deriva_ml_get_execution` and
`deriva_ml_list_execution_children` views on status, RID, duration, and
parent/child linkage; **disagrees on `workflow_rid`** (direct says
`DY6`, MCP says `null` for every row — see `findings/developer/02-*`).
19 Execution_Asset rows confirmed in both channels.

Training arc on catalog 18, ran four configurations against the
Curator's curated datasets:

**Run 1 — `cifar10_quick` (single, baseline)**
| Field | Value |
|---|---|
| Execution RID | `DYC` |
| Workflow RID | `DY6` (`cifar10_cnn`) |
| Hydra config | `+experiment=cifar10_quick` (defaults: `cifar10_small_labeled_split` = CRR) |
| Dataset | CRR (250 imgs: 200 train + 50 test, seed=123) |
| Architecture | 32→64 channels, 128 hidden |
| Hyperparams | epochs=3, batch=128, lr=1e-3 |
| Final train_acc | 40.50% |
| Final test_acc | **28.00%** |
| Weights asset | `E06` (6.55 MB, `cifar10_cnn_weights.pt`) |
| Training log | `E08` (480 B) |
| Predictions CSV | `E0A` (7.2 KB, 50-row `prediction_probabilities.csv`) |
| Duration | 0.676 s training |

**Run 2 — `cifar10_extended` (single, max-config)**
| Field | Value |
|---|---|
| Execution RID | `E4A` |
| Hydra config | `+experiment=cifar10_extended` |
| Dataset | CRR (same as Run 1 — controlled comparison) |
| Architecture | 64→128 channels, 256 hidden, dropout 0.25, weight_decay 1e-4 |
| Hyperparams | epochs=50, batch=64, lr=1e-3 |
| Final train_acc | 100% (overfit) |
| Final test_acc | **24%** (peaked at 32% around epoch 29 — overfit thereafter) |
| Weights asset | `E64` (24.91 MB — 4× the quick model) |
| Training log | `E66` (3.5 KB, 50 epochs of metrics) |
| Predictions CSV | `E68` (8.0 KB) |
| Duration | 11.69 s training |

**Run 3 — `lr_sweep` (multirun, 4 children)**
Parent: `EA8` (description = LEARNING_RATE_SWEEP_DESCRIPTION markdown).
Children link back via `Execution_Parent_Execution_Child` association
(verified by `deriva_ml_list_execution_children(parent_rid="EA8")` →
4 rows).

| Sequence | Child RID | LR | Final test_acc | Weights / Log / Predictions |
|---|---|---|---|---|
| 0 | `EC0` | 0.0001 | 14% (undertrained — 10 epochs not enough at this LR) | `EDW` / `EDY` / `EE0` |
| 1 | `EJ0` | 0.001 | **30% (best)** | `EKW` / `EKY` / `EM0` |
| 2 | `ER0` | 0.01 | 12% (oscillating — LR too aggressive) | `ESW` / `ESY` / `ET0` |
| 3 | `EY0` | 0.1 | 10% (random — divergence; epoch 1 train_loss = 1269!) | `EZW` / `EZY` / `F00` |

The sweep tells a clean story: LR=1e-3 is the sweet spot at 10 epochs
on this 200-image training set; an order of magnitude higher
destabilizes; an order of magnitude lower undertrains. Analyst can use
this as the worked LR-comparison example.

**Run 4 — degenerate (kept in catalog as evidence of finding 01)**
| Execution RID | `F40` |
|---|---|
| Hydra config | `+experiment=cifar10_quick datasets=cifar10_validation_from_test` |
| Dataset | DAP (Validation type — what the Curator recommended) |
| What happened | `_bag_role(DAP)=="unknown"` → no train_loader → no-train path → 50-byte `training_status.txt` (`F5T`) written, status=Uploaded |
| Useful work | None |

Documented as `findings/developer/01-validation-typed-bag-silently-skipped.md`
(pending task D01 catalog instance). **Analyst should skip F40 when ranking
runs** — it has no model weights, no predictions, and would silently
appear in a "find all uploaded training executions" query.

**Why these four configs, and not others?**

- *Two single variants on the same dataset (CRR) for a controlled comparison*
  — quick vs extended isolates "what does more model + more epochs buy
  you?" on this 200-image training set. Answer: severe overfitting,
  modest improvement in best test_acc (32% peak) before degradation.
  This is the expected failure mode of training a 600K-parameter model
  on 200 images; the worth-keeping signal for the Analyst.
- *`lr_sweep` as the multirun* — Picked because the catalog ships it
  pre-defined in `src/configs/multiruns.py`. The 4-way grid is small
  enough to finish in seconds and large enough to be plottable on a
  4-bar chart. No new multirun config registered — `lr_sweep`
  already serves the goal.
- *DAP attempt* — Tried the Curator's recommendation as-is and let the
  silent-skip happen, so the finding is reproducible with the exact
  execution `F40` for the Analyst to inspect. Did not "route around"
  by quietly substituting; the friction is the point.
- *Did not run `epoch_sweep` or `lr_batch_grid`* — Time budget. lr_sweep
  is the more interpretable story (1 axis, 4 values) and covers the
  multirun success criterion. The grid would have added 4 more children
  without distinct signal.

**Seed strategy.** Honest version: there isn't one for this arc. The
Curator's tk-003 suggests using C86 (seed=42) and CS0 (seed=123) for
"variance across seeds," but those are *partition* seeds, not *training*
seeds — and `cifar10_cnn.py` has no `seed` parameter at all. So every
training run pulls from PyTorch's default uninitialized global RNG.
Logged as `findings/developer/03-no-seed-knob-on-cifar10-cnn.md`
(catalog-18 instance of pending task D02). True variance-across-seeds
work is blocked until D02 lands. Skipped the Curator's
seed=42-vs-seed=123 dataset comparison for this arc — it would have
conflated partition variance with training variance and produced a
muddled signal.

**What "success" looked like.** Not "best test accuracy" — at 200
training images, the ceiling is low and CIFAR-10 is a known
hard-at-small-N dataset. Success was: *the Analyst can rank these
executions by a defensible metric, see the lr_sweep pattern clearly,
and produce a confusion matrix that's interpretable.* All runs use
the same 50-image labeled test partition (CSA, the seed=123 small test
subset), so test_acc is apples-to-apples across DYC, E4A, EC0, EJ0,
ER0, EY0. The probability CSVs (`prob_<class>` columns) are the right
shape for ROC analysis — 50 rows × (Image_RID + Predicted_Class +
Confidence + 10 per-class probabilities).

**Cross-channel verification result.** Direct ermrest (deriva-ml
`PathBuilder`) and MCP `deriva_ml_get_execution` /
`deriva_ml_list_execution_children` / `deriva_ml_list_assets` agree on:
8 executions (DYC, E4A, EA8, EC0, EJ0, ER0, EY0, F40), all
`Status=Uploaded`, 19 Execution_Asset rows total (3+3+0+3+3+3+3+1).
**They disagree on `workflow_rid`**: direct says `DY6`, MCP says
`null` for every row. Logged as
`findings/developer/02-mcp-get-execution-drops-workflow-rid.md`.

---

### Handoff to Analyst

**What's ready for the Analyst arc:**

- **6 viable training executions on catalog 18 with weights + per-image
  predictions CSV**, all using the same 50-image test partition (CSA).
  Apples-to-apples comparison across all 6 on test_acc, AUC, confusion
  matrix, ROC, etc.
- **One multirun parent (`EA8`) with 4 children** for the LR-sweep
  visualization story.
- **All prediction CSVs share the same schema**:
  `Image_RID, Predicted_Class, Confidence, prob_airplane,
  prob_automobile, prob_bird, prob_cat, prob_deer, prob_dog, prob_frog,
  prob_horse, prob_ship, prob_truck`. 50 rows each.
- **All runs train+test on CRR (`cifar10_small_labeled_split`)**, so
  joining predictions to ground-truth labels uses the CSA test partition
  RIDs.

**Recommended executions to compare and the ranking by metric:**

| Rank | Execution | Variant | Test_acc | Notes |
|---|---|---|---|---|
| 1 | `EJ0` | lr_sweep child, lr=0.001 | **30%** | Best learning rate at 10 epochs |
| 2 | `DYC` | quick (default lr=1e-3, 3 epochs) | 28% | Cheapest run, near-best result |
| 3 | `E4A` | extended (50 epochs, larger model) | 24% (peak 32% at epoch 29) | Severely overfit by end of training |
| 4 | `EC0` | lr_sweep child, lr=0.0001 | 14% | Undertrained |
| 5 | `ER0` | lr_sweep child, lr=0.01 | 12% | Unstable (loss oscillates) |
| 6 | `EY0` | lr_sweep child, lr=0.1 | 10% | Diverged (train_loss spiked to 1269) |

**Recommended prediction-CSV assets to load for analysis:**

- DYC → `E0A`
- E4A → `E68`
- EC0 → `EE0`, EJ0 → `EM0`, ER0 → `ET0`, EY0 → `F00`

**Skip in any comparison:**

- **`F40`** — degenerate execution from finding 01 (Validation bag
  silently skipped). Status=Uploaded, no weights, no predictions, just
  a 50-byte `training_status.txt`. If you use `find_executions(...)` to
  build the comparison list, filter by checking `len(execution_assets) >= 2`
  or filter out workflow type "Training" with no `cifar10_cnn_weights.pt`
  asset.
- **`EA8`** — multirun parent; descriptive shell with no model
  artifacts. Use its 4 children instead.

**Suggested analysis angles:**

1. **LR-sweep comparison plot** — 4-bar chart of EC0/EJ0/ER0/EY0
   test_acc by learning rate. The Analyst's job is to pick the
   right plot type (log-scale x-axis for LR is natural; the divergent
   EY0 case is the spike worth annotating).
2. **Quick-vs-Extended overfit story** — DYC vs E4A, with E4A's
   training_log showing the test_acc peak at epoch 29 then decline.
   The `training_log` assets carry per-epoch metrics; load them as
   text and parse the `Epoch N/50:` lines.
3. **ROC analysis using the prediction probability CSVs.** With CRR
   training on a 200-image pool and CSA test on 50 images (5 per class,
   per CSA's stratified construction), each class has only 5 test
   examples — be aware that ROC AUC is noisy at this sample size.
   This is why the Curator added `cifar10_balanced_demo` (DB0, 50
   images, 5/class) as the "guaranteed-populated confusion matrix"
   evaluator — but **none of the executions in this arc were evaluated
   against DB0**, only against CSA. If the Analyst wants a confusion
   matrix on DB0, they'd need to do a `test_only` run (use existing
   weights from one of the executions, load via `assets=` config
   pointing at the weights RID, run on DB0). Not required by the
   success criteria, but a natural extension.

**Caveats:**

- **No held-out validation set was used in training.** The Curator's
  intended use of DAP (`cifar10_validation_from_test`, 250 imgs disjoint
  from training pool) is blocked by finding 01 (`cifar10_cnn` skips
  Validation bags). Every reported test_acc above is on the
  in-distribution CSA partition. There's no overfitting-to-test concern
  because the model never saw the test images during training, but
  also no proper generalization check on truly held-out data.
- **Test set is small (50 images, 5 per class).** Confusion matrices
  will have at most 5 in any diagonal cell and 0 in many off-diagonal
  cells. Plot accordingly.
- **MCP `workflow_rid` is null on every row** (finding 02). If the
  Analyst's notebooks reach for MCP to look up the workflow URL or git
  hash, they'll get None — use the direct deriva-ml Python path or
  query the `Workflow` table directly by `DY6`.
- **No `seed` was set** (finding 03). Re-running any of these
  experiments will produce *different* weights and *different* test_acc.
  The numbers above are accurate for *these specific* execution RIDs
  (DYC, E4A, EC0...) and won't reproduce from the configs alone.

**Pinned things the Analyst should NOT change:**

- The 6 viable + 1 degenerate executions are committed; do not delete
  or modify their Execution rows. Add analysis assets as new
  Execution_Asset rows under new analysis executions.
- `src/configs/datasets.py` and `src/configs/deriva.py` are
  `[E2E-DROP]` — don't repoint.

**Open questions left for the Analyst (not directives — questions to
answer through their own work):**

1. Is the 32% peak at epoch 29 of E4A meaningful, or noise? Without a
   seed (finding 03), re-running the same config might land anywhere
   from 24% to 36%. The Analyst's ROC + confusion-matrix work might
   surface whether E4A is *qualitatively* different from EJ0 (e.g.,
   confusing different class pairs) or just a noisier estimate of the
   same underlying behavior.
2. Should the LR sweep be redone with longer epoch budgets per
   learning rate (smaller LRs given more time)? EC0 at lr=1e-4 with
   only 10 epochs barely got out of random-init territory.
3. The Curator added DAP as a held-out validator that the runner
   doesn't currently honor. If the Analyst wants a true generalization
   metric, the workaround is a `test_only` run pointing at a
   `cifar10_cnn` weights asset and using `datasets=cifar10_testing`
   (97A — Testing-typed, same images as DAP) instead of
   `cifar10_validation_from_test`.

---

### tk-005 — Analyst metric choice: AUC_Macro is the right summary at N=50, accuracy is the secondary
**When:** 2026-05-26T11:45:00-07:00
**By:** Analyst persona (sub-agent, `e2e-test/2026-05-26` worktree)
**Supported by:** ROC notebook execution `F6C` on catalog 18,
`roc_metrics.csv` asset `F9P`, plus direct deriva-ml cross-check
(`scripts/analyst_verify_executions.py`). Both `roc_metrics.csv`
columns (`Accuracy`, `AUC_Macro`, `AUC_Micro`) and the tk-004 test_acc
table agree on the accuracy rank.

Compared the Developer's 6 viable executions (DYC, E4A, EC0, EJ0,
ER0, EY0 — all trained on `CRR`, all tested on `CSA`). Final
ranking by **AUC_Macro** is the summary metric for this report:

| Rank | Exec | Accuracy | AUC_Macro |
|---|---|---|---|
| 1 | **E4A** | 24% | **0.695** |
| 2 | EJ0 | 30% | 0.647 |
| 3 | EC0 | 14% | 0.642 |
| 4 | DYC | 28% | 0.616 |
| 5 | ER0 | 12% | 0.569 |
| 6 | EY0 | 10% | 0.500 (chance) |

**Why AUC_Macro and not accuracy.** At N=50 test images (5/class),
argmax-accuracy has a binomial std error of ~7 percentage points.
The 8-point spread between the accuracy-best (EJ0 at 30%) and the
accuracy-3rd (E4A at 24%) is well within noise. AUC integrates
over the threshold sweep and uses the full probability vector,
so it's the more stable estimate at this sample size. Whether
"top-1 is right" (accuracy) is the user-facing experience that
matters in production for a classification model is a separate
question — for *ranking model variants*, AUC is the cleaner
signal here.

**The surprise this metric choice surfaces:** E4A wins on AUC
despite losing on accuracy. The Developer's tk-004 noted E4A
peaked at ~32% test_acc around epoch 29 then degraded. The AUC
numbers say the *underlying score distribution* on E4A is the
best-discriminating of the six — its argmax just lands on the
wrong class slightly more often than EJ0's at the end of training.
Two reasonable reads:

1. *E4A is genuinely the best model and its accuracy is unlucky
   on these 50 test images.* Without a seed (D02), can't
   distinguish from (2) by re-running.
2. *E4A's well-calibrated probabilities are an artifact of
   training-to-100% on the training set;* the model is confident
   but slightly miscalibrated for the test distribution.

Honest assessment: at this sample size both reads are admissible.
For decisions like "which weights to ship," I would want a held-
out validation set (DAP, blocked by developer/01) before declaring
a winner. For *this report's purpose* (compare what the Developer
arc produced), AUC_Macro is the best metric available.

Alternatives weighed:

- *Accuracy alone* — rejected: high noise floor at N=50, ranks
  on argmax which doesn't use the full prediction vector.
- *Top-3 accuracy* — rejected: not requested by the Developer's
  prediction CSV schema (would need to recompute, low marginal
  value at 10 classes × N=50).
- *Per-class F1, averaged* — rejected: even noisier than
  accuracy at 5 imgs/class (e.g. EJ0's recall on `dog` is 0/5 =
  undefined precision territory).
- *AUC_Micro* — close call, same ranking direction as AUC_Macro
  but slightly compressed (the EY0 chance run sets micro=0.500
  too, so EY0 is unambiguously last either way). Macro
  highlights class-imbalanced behavior even though the test set
  is class-balanced, so kept it as primary for future
  comparability when test sets aren't balanced.

Skipped executions:

- **F40** (Validation bag skipped, developer/01) — 1 asset, no
  predictions. Filtering by `len(execution_assets) >= 3` cleanly
  excludes it without hardcoding the RID.
- **EA8** (lr_sweep parent) — 0 model artifacts. Same filter
  catches it.

---

### tk-006 — Analyst denormalize rationale: row_per=Execution_Image_Image_Classification on CSA
**When:** 2026-05-26T11:50:00-07:00
**By:** Analyst persona (sub-agent, `e2e-test/2026-05-26` worktree)
**Supported by:** `scripts/analyst_denormalize_check.py` reconciliation
artifact (`findings/analyst/_artifacts/denormalize_check.json`),
cross-channel agreement between MCP `deriva_ml_denormalize_dataset`
and direct `Dataset.get_denormalized_as_dataframe`, plus
`list_dataset_members(CSA)['Image']` and `feature_values('Image',
'Image_Classification')` both confirming the 50-image set + 5/class
balanced distribution.

Exercised `Dataset.get_denormalized_as_dataframe` on dataset `CSA`
(50-image test partition, seed=123) with:

```python
ds.get_denormalized_as_dataframe(
    include_tables=["Image", "Execution_Image_Image_Classification"],
)
# row_per auto-inferred to "Execution_Image_Image_Classification"
# returns 350-row wide table (50 imgs × 7 producing executions)
```

**Why the feature *table* name instead of the feature name in
`include_tables`.** Tried `include_tables=["Image",
"Image_Classification"]` first, which is the natural mental
model (feature name is the API-public identifier everywhere
else). `describe_denormalized` accepts it; `get_denormalized_as_dataframe`
rejects it with "The table Image_Classification doesn't exist."
The fix is `find_features('Image')` → use the
`feature_table` attribute. Logged as
`findings/analyst/01-describe-vs-run-include-tables.md`.

**Why `row_per` was left auto-inferred (`Execution_Image_Image_Classification`)
instead of pinned to `Image`.** Two reasons:

1. The `Execution_Image_Image_Classification` table carries the
   `Execution` discriminator. With `row_per="Image"`, the
   denormalizer refuses to aggregate over multiple feature rows
   per image — and rightly so, since one image has 7 different
   feature rows (one ground truth + 6 predictions). The
   auto-inferred long-format view is the *correct* shape for an
   analysis that needs both ground truth and predictions visible.
2. The wide-table-as-feature-value-list view lines up with the
   right analysis primitive: filter
   `Execution_Image_Image_Classification.Execution == "854"` for
   ground truth, `Execution_Image_Image_Classification.Execution
   in {DYC, E4A, EC0, EJ0, ER0, EY0}` for predictions. Two
   filters of the same DataFrame replace what would otherwise be
   "load all 6 prediction CSVs from hatrac, join each to the
   ground-truth feature query, then concatenate."

If the question were "give me one row per Image with the GT
label and the 6 prediction confidences as separate columns,"
that would require aggregation (which the denormalizer
explicitly refuses) or post-processing the long-format view
(pivot or pandas `groupby`). For this analysis the long-format
view was sufficient — the existing ROC notebook downloads each
prediction CSV separately and joins per-experiment, so the
denormalize call was *additional cross-check infrastructure*
rather than the primary data path.

**Reconciliation result.** All three channels (denormalize wide
table, `list_dataset_members(CSA)['Image']`, and
`feature_values('Image', 'Image_Classification')` filtered to
the CSA image RIDs) agree on:

- 50 image RIDs, exact set match
- 5-per-class balanced label distribution across all 10 CIFAR-10
  classes
- 50 ground-truth rows (Execution=854), 50 prediction rows per
  Developer execution (350 total)

No denormalize-specific finding raised — the surface is
behaving correctly on this catalog.

**The describe-vs-run inconsistency is the one rough edge** and
is filed as analyst/01 (Low severity). The output column naming
(`Table.column`, with `Image.RID` and
`Execution_Image_Image_Classification.Image` both present and
equal-valued) is mildly redundant but predictable; ergonomic
nit, not a finding.

**For future Analysts using this surface:** the
`describe_denormalized` preview's `estimated_row_count` field
correctly reports 50 when `row_per="Image"` is set, and 350 (well,
"unknown N" because the FK direction is downstream) when
`row_per` is left to auto-infer to the feature table. Use the
preview to validate the join shape before fetching the data; it's
faster than running and inspecting the result.

---

### Handoff to Wrap-up

**What's ready in catalog 18 from the Analyst arc:**

- **Analysis execution `F6C`** (`Workflow_Type=ROC Analysis Notebook`,
  Status=Uploaded). 14 new `Execution_Asset` rows:
  - 6 per-experiment ROC curve JPGs: `F8W` (DYC), `F8Y` (E4A),
    `F90` (EC0), `F92` (EJ0), `F94` (ER0), `F96` (EY0)
  - 1 ROC comparison overlay JPG: `F98`
  - 6 confusion matrix JPGs: `F9A` (DYC), `F9C` (E4A), `F9E`
    (EC0), `F9G` (EJ0), `F9J` (ER0), `F9M` (EY0)
  - 1 summary CSV: `F9P` (`roc_metrics.csv`, 6 rows × 16 cols)
  - 1 executed notebook + 1 markdown export: `FBP` / `FBR`
- **`src/configs/assets.py`** now has three named asset groups
  wired in for this catalog: `roc_quick_vs_extended` (E0A, E68),
  `roc_lr_sweep` (EE0, EM0, ET0, F00), and `roc_all_six` (all 6).
  All `[E2E-DROP]`.
- **`src/configs/roc_analysis.py`** gained a `roc_all_six`
  notebook config, also `[E2E-DROP]`.

**The markdown report:**
[`docs/reports/2026-05-26-multipersona-analysis.md`](docs/reports/2026-05-26-multipersona-analysis.md)
— 5-minute read covering ranking, the accuracy-vs-AUC divergence,
denormalize experience, cross-channel verification table, and
open questions.

**Cross-channel verification table (full):** see report §5. All
direct-vs-indirect checks pass for the analysis assets and the
denormalize surface; the previously-known `workflow_rid: null`
disagreement (developer/02) persists on F6C as expected.

**Findings raised** (in `findings/analyst/`):

1. `01-describe-vs-run-include-tables.md` — Low. Denormalize
   describe accepts feature names that the runner rejects.
2. `02-execution-description-stale-on-asset-override.md` — Low.
   Notebook execution row's description ignores Hydra asset
   overrides.

**Pinned things the wrap-up should NOT change:**

- Execution `F6C` and its 14 output assets — committed catalog
  state.
- `src/configs/assets.py` and `src/configs/roc_analysis.py`
  asset/notebook config entries are `[E2E-DROP]` and drop at
  wrap-up along with `default_deriva`/`default_dataset`.

**Open questions for the user at checkpoint:**

1. Is the accuracy-vs-AUC ranking divergence (E4A best on AUC,
   EJ0 best on accuracy) interesting enough to spawn its own
   investigation, or expected behavior at this sample size?
2. Should `findings/analyst/01-*` (describe/run inconsistency)
   be promoted to a GitHub issue with my preferred Option 2 fix
   suggestion (resolve feature names to feature tables at runner
   call time), or left as Low-severity backlog?
3. Should the `roc_all_six` asset config be promoted to the
   default for future analysts, or kept ad-hoc per run?
4. The Curator's `DB0` (`cifar10_balanced_demo`) was not used —
   would a follow-up `test_only` analysis run on DB0 add value,
   or is the CSA-only analysis enough for this multipersona
   run's purposes?

