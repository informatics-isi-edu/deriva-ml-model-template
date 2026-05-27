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

## tk-001 — Bootstrap

**When:** 2026-05-27 (multipersona rerun, post-#246 row-completeness fix)
**By:** Orchestrator (Phase 0)
**Supported by:** —

### What was set up

Fresh catalog **`e2e-test-20260527`** (catalog id `93`) on `localhost`. Bootstrapped
end-to-end via `load-cifar10`:

| Phase | Command | Output |
|---|---|---|
| schema | `load-cifar10 --hostname localhost --create-catalog e2e-test-20260527 --phase schema` | catalog 93 created, domain schema in place |
| images | `load-cifar10 --hostname localhost --catalog-id 93 --num-images 1500 --phase images` | 1500 Image rows (750 train / 750 test), 1500 Image_Classification features with perfectly balanced class distribution (150 per class × 10 classes) |
| datasets | `load-cifar10 --hostname localhost --catalog-id 93 --num-images 1500 --phase datasets` | 13 datasets (see deriva://catalog/localhost/93/ml/Dataset for the full list) |

### Why `--num-images 1500`

The small-variant guard (`SMALL_TRAIN_SIZE`/`SMALL_TEST_SIZE` = 500/500)
needs `--num-images >= 1002` to keep the Toronto small-split distinct
from the full split. 1500 gives headroom; matches the catalog scale
intended for the 2026-05-27 platform-test run.

### Why this run

Validates the **deriva-ml v1.39.4 PagedFetcher row-completeness fix** (PR #246):
the prior 2026-05-26 run used `--num-images 500` and never exercised the
URL-length guard. The 2026-05-27 first attempt at `--num-images 1500`
exposed a fetcher bug where oversized GET requests were silently dropping
RIDs. PR #246 lands the chunk-loop replacement; this run is the e2e validation.

### Sibling versions pinned for this run

- deriva-ml: `v1.39.4` (`4ed88122`) — contains the #246 fix
- deriva-py: pinned via `@deriva-ml` branch (`e944ad8e`)
- deriva-ml-mcp: `v0.5.3`
- deriva-mcp-test container: rebuilt against above

### Handoff to Curator

- The catalog id, schema, image counts, and dataset RIDs are now
  visible via the standard MCP resources — query them rather than
  reading them out of this file.
- Default Hydra connection (`default_deriva`) and dataset config
  (`default_dataset` → small labeled split WD2) already point at
  catalog 93. Just `uv run deriva-ml-run --info` from this worktree.
- Class distribution is perfectly balanced (150 per CIFAR-10 class
  × 10 classes = 1500). This is post-fix #15 behaviour; if the
  Curator finds skew, that's a finding against the loader.

---

## tk-002 — Catalog audit (2026-05-27, catalog 93)

**When:** 2026-05-27 (Curator persona arc, post-bootstrap)
**By:** Curator
**Supported by:** tk-001

### Cross-channel verification

Both MCP and direct deriva-ml Python API report identical state. No
disagreement surfaced — the PagedFetcher row-completeness fix
(deriva-ml v1.39.4 / PR #246) holds under the actual catalog scale
of this run.

### Dataset inventory (13 bootstrap datasets, all `0.1.0.post1.devN`)

| RID | Types | Members | Description (truncated) |
|---|---|---|---|
| JZ8 | Complete, Labeled | 1500 Image | Complete CIFAR-10 dataset (750 train + 750 test) |
| JZJ | Split | 2 Dataset (→ JZT, K04) | Full split into training/testing |
| JZT | Training, Labeled | 750 Image | Full training partition |
| K04 | Testing, Labeled | 750 Image | Full testing partition |
| K0M | Split | 2 Dataset (→ K0W, K16) | Toronto small split (500/500) |
| K0W | Training, Labeled | 500 Image | Toronto small training |
| K16 | Testing, Labeled | 500 Image | Toronto small testing |
| TX0 | Split | 2 Dataset (→ TX8, TXJ) | 80/20 labeled split of full training |
| TX8 | Training, Labeled | 600 Image | 600 stratified from JZT, seed=42 |
| TXJ | Testing, Labeled | 150 Image | 150 stratified from JZT, seed=42 |
| WD2 | Split | 2 Dataset (→ WDA, WDM) | 80/20 small labeled split |
| WDA | Training, Labeled | 400 Image | 400 stratified from JZT, seed=42 |
| WDM | Testing, Labeled | 100 Image | 100 stratified from JZT, seed=42 |

Note: TX*/WD* sub-splits all draw from JZT (the *training* pool),
not K04 — leaving K04 fully untouched as a held-out testing source.
This shaped the curated variant choice in tk-003.

### Image_Classification feature distribution

Total: 1500 records (one per Image). Perfectly balanced across all
10 CIFAR-10 classes, 150 per class:

| Class | Count | Class | Count |
|---|---|---|---|
| airplane | 150 | dog | 150 |
| automobile | 150 | frog | 150 |
| bird | 150 | horse | 150 |
| cat | 150 | ship | 150 |
| deer | 150 | truck | 150 |

K04 sub-distribution is also perfectly balanced: 75/class × 10.

### Vocabulary `Dataset_Type` — 8 terms

Bootstrap-installed: Complete, File, Training, Testing, **Validation**,
Split, Labeled, Unlabeled. **`Validation` was unused** — no dataset
in the bootstrap was Validation-typed. The cifar10_cnn runner
dispatches on `Dataset_Type` (D01 lineage from 2026-05-26 run); a
Validation-typed bag is consumed as held-out evaluation. With no
Validation dataset present, the dispatch lane has no data. This
gap motivated tk-003's curated variant.

### Friction during the audit

None blocking. Findings filed separately if any (see
`findings/curator/`).

---

## tk-003 — Curated Validation dataset XEM + handoff to Developer

**When:** 2026-05-27 (Curator persona arc)
**By:** Curator
**Supported by:** tk-001, tk-002

### Why a Validation dataset

The bootstrap shipped Training/Testing/Split/Complete coverage but
not Validation. The cifar10_cnn runner's Dataset_Type dispatch
(D01, completed in the 2026-05-26 run) treats Validation-typed
inputs as held-out evaluation, separate from training-set
consumption. Without a Validation dataset, that lane is unreachable
from configuration alone — the Developer would have to either
fabricate one (defeats the dispatch test) or skip the lane (defeats
the safety rail). One Validation dataset closes the gap.

### What was created

| RID | Types | Members | Drawn from | Seed | Notes |
|---|---|---|---|---|---|
| **XEM** | Validation, Labeled | 100 Image (10/class) | K04 (testing) | 2026 | Stratified pick. v=`0.1.0.post1.dev1`. |

Construction details (see `scripts/curator_create_validation.py`):

- Source pool: K04 (the 750-image full testing partition), chosen
  because no existing Training subset (TX8/WDA) draws from it, so
  XEM is naturally disjoint from any training set.
- Stratification: exactly 10 per class, deterministic seed=2026
  (distinct from the seed=42/43 family used by Toronto small-split
  and seed=42 TX*/WD* sub-splits — XEM and any training subset can
  be re-derived independently without collision).
- Provenance: created under a Curator-tagged Execution (RID `XDM`)
  with a fresh workflow `Curator Validation Subset (cat 93)` of
  type `CIFAR_Data_Load`.
- Verified cross-channel: MCP `deriva_ml_get_dataset(XEM)` and
  direct `ml.lookup_dataset(XEM).list_dataset_members()` both
  report 100 Image members.

### Handoff to Developer: use case → dataset table

| Use case | Recommended dataset | Why |
|---|---|---|
| **Quickest training smoke test** | WD2 v=0.1.0.post1.dev1 (Split with WDA train 400 / WDM test 100) | Smallest labeled split that exercises stratified data. Default in `src/configs/datasets.py:default_dataset`. |
| **Mid-size training (default flow)** | TX0 v=0.1.0.post1.dev1 (Split with TX8 train 600 / TXJ test 150) | 80/20 stratified from JZT. Larger than WD2; still fast on macOS. |
| **Full-size training (this run)** | JZJ v=0.1.0.post1.dev1 (Split with JZT train 750 / K04 test 750) | Whole 1,500-image catalog scope. K04 testing intersects with XEM (Validation) — see caveat below. |
| **Toronto small-split (no relabeling work)** | K0M v=0.1.0.post1.dev1 (Split with K0W train 500 / K16 test 500) | Original Toronto-style flat 500/500. Use when comparing against the pre-curation baseline. |
| **Held-out validation** (NEW, this arc) | **XEM v=0.1.0.post1.dev1** (Validation, 100 Image) | 100 stratified images from K04, seed=2026. Drives the Dataset_Type Validation dispatch lane in cifar10_cnn. |
| **Whole-catalog evaluation / denormalize stress test** | JZ8 v=0.1.0.post1.dev3 (Complete, 1500 Image) | All 1,500 images with ground-truth labels — right shape for the Analyst's denormalize work. |

### Caveats for the Developer

1. **XEM overlaps with K04.** XEM's 100 Image RIDs are a
   stratified subset of K04 (the 750-image full testing
   partition). If you train on JZJ → testing-eval on K04 → also
   validate on XEM, the 100 XEM images are in *both* the test set
   and the validation set. This is fine for the dispatch lane test
   but means XEM is not a true "unseen" set for any model trained
   on JZJ's K04 half. If the experiment needs strict held-out
   validation, train on TX0 instead (TX8/TXJ are stratified from
   JZT only, so K04 and XEM are both unseen).
2. **All dataset versions are dev labels** (`0.1.0.post1.devN`),
   per ADR-0003. None have been released. If the Developer wants
   citable RIDs for the final experiment notes, call
   `deriva_ml_release(rid)` after their training runs settle the
   final dataset state, then update `src/configs/datasets.py`
   with the released version strings.
3. **Class distribution is perfectly balanced everywhere** —
   1500/10, 750/10, 500/10, 400/10, 150/10, 100/10, all stratify
   cleanly. No class-rebalancing work needed; standard
   `CrossEntropyLoss` is appropriate.
4. **`src/configs/datasets.py` was wired by Phase 0** for the 13
   bootstrap RIDs but **does not yet include XEM**. If the
   Developer wants a named handle (e.g. `cifar10_validation`),
   add a `DatasetSpecConfig(rid="XEM", version="0.1.0.post1.dev1")`
   entry — the dataset-lifecycle skill's "wire into datasets.py"
   offer should pick this up automatically.

### Open questions for the Developer

- Should XEM be released (promoted to a stable label) before any
  training run consumes it? Curator's call: yes if the Developer
  pins it into config, no if it's only used ad-hoc.
- Is 100 images enough to drive the Validation dispatch lane in a
  way that produces a meaningful metric, or should the Curator
  expand to 200/class? Default: 100/class felt right for a smoke
  test of the dispatch path; expand if the metric is too noisy.

---

## tk-004 — Developer: training runs (catalog 93, workflow XN8)

**When:** 2026-05-27 (Developer persona arc)
**By:** Developer
**Supported by:** tk-001, tk-002, tk-003

### Workflow used

Single workflow **`XN8` — CIFAR-10 2-Layer CNN** (type
`Training` + `Image Classification`,
checksum `ed8fbff538bf20f3e692ec45be053ed1bd034a0b`). All three
executions reuse this workflow; only the Hydra overrides differ.
Workflow URL pins to commit
`6c9a78157773f4014ff7974f4c13a0423c88adf3` of
`src/models/cifar10_cnn.py`. `allow_dirty=True` is set on each
execution because the worktree carries unpushed `[E2E-DROP]`
commits — that's the documented dev override (`DERIVA_ML_ALLOW_DIRTY=true`).

### Executions (ranked by final test accuracy)

| Rank | Exec RID | Variant | Dataset(s) | Epochs | Seed | Final test_acc | Final val_acc | Weights | Log | Predictions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **XYG** | `cifar10_default` (default_model on TX0) | TX0 (600/150) | 10 | 123 | **42.00%** | — | Y0A | Y0C | Y0E |
| 2 | **YAP** | `cifar10_regularized` (dropout 0.25, wd 1e-4) on TX0+**XEM** | TX0 (600/150) + **XEM (100 Validation)** | 10 | 2026 | 37.33% | **43.00%** | YCJ | YCM | YCP |
| 3 | **XNE** | `cifar10_quick` (3-epoch smoke) on WD2 | WD2 (400/100) | 3 | 42 | 24.00% | — | XQ8 | XQA | XQC |

All three: `status=Uploaded`, `workflow=XN8`, 3 Execution_Assets each
(weights `.pt` + training log `.txt` + per-image
`prediction_probabilities.csv`). Total catalog state added by this
arc: **3 executions, 9 Execution_Asset rows**.

### Decisions

**Dataset choice.**

- **XNE smoke (WD2):** Pick the smallest labeled split that
  exercises stratification so the runner pipeline is end-to-end
  validated before committing compute to the longer runs. 3 epochs
  on a CPU finishes in ~1 second of wall training time. Threshold
  metric not the point — this run is a pipeline smoke test.
- **XYG main run (TX0):** The Curator's tk-003 recommends TX0
  (mid-size) for the main training flow because it's stratified
  80/20 from the *training* pool JZT and is naturally disjoint
  from K04 (and therefore from XEM). Picked seed=123 to introduce
  run-to-run variance against the seed=42 default already pinned
  in `cifar10_cnn.py`.
- **YAP Validation lane (TX0 + XEM):** Designed to exercise the
  D01 Dataset_Type dispatch that the cifar10_cnn runner gained on
  2026-05-26 but that had no Validation-typed input in the
  bootstrap. Added the composite `cifar10_labeled_split_with_validation`
  config so the lane is reachable via plain Hydra CLI. seed=2026
  matches Curator XEM construction seed for cross-arc symmetry.
  XEM only overlaps with K04 (which TX0 doesn't draw from) so the
  Validation set is genuinely unseen by the training half. This is
  the run worth re-doing if Analyst wants to test stricter
  hyperparameter choices.

**Model variant choice.**

- One variant per run, each with a different question:
  - `cifar10_quick`: pipeline-only (does the runner work?).
  - `default_model`: baseline (what does 10 epochs of a vanilla
    config buy you on 600 stratified images?).
  - `cifar10_regularized`: regularized variant (does dropout +
    weight decay help on this scale?).
- Did *not* run the 50-epoch `cifar10_extended` variant. Reasoning:
  at 1500 catalog images and 600-image training sets, the
  Curator's perfectly-balanced class distribution + small data
  ceiling means an extra 40 epochs would mostly overfit. The
  Analyst is welcome to run it if they want an overfitting
  signal in the ROC analysis.
- Did *not* run a multirun (sweep). The plan rated this useful
  but here three single-experiment runs at different points in
  the dataset×variant×seed space produces a more interpretable
  ranking than four learning-rate-sweep points all on
  `cifar10_quick`. Logged here so it isn't read as a gap — it's a
  deliberate choice given the small dataset.

**Hyperparameter choices.**

- Default learning rate (1e-3) and batch size (64) on XYG/YAP
  match the model-config defaults; deliberately *not* swept
  because run 2 / run 3 differ on model variant + dataset already.
  More moving variables would make the ranking noisier without
  enough runs to disambiguate.
- Three distinct seeds (42, 123, 2026) — the byte-reproducibility
  knob added in #30 means re-running any of these is exact.

### Cross-channel verification (done)

**Indirect channel (MCP):**
- `deriva_ml_list_executions(sort=True)` reports 8 executions
  total in the catalog. The 3 most recent (YAP, XYG, XNE) all
  carry `workflow_rid=XN8` and `status=Uploaded`. Older 5 are
  Curator/loader executions and check out.
- `deriva_ml_list_assets("Execution_Asset")` reports `count=9` —
  exactly the expected 3×3 set, RIDs match what `commit_output_assets`
  printed verbatim during each run.
- `deriva_ml_get_lineage(YAP, depth=1)` shows YAP consumed
  `TX0 v0.1.0.post1.dev1` and `XEM v0.1.0.post1.dev1` as input
  datasets — confirming the Validation dispatch lane was actually
  fed, not just intended.

**Direct channel (deriva-ml Python via path-builder):**
- `Execution.filter(RID==<rid>).link(Execution_Asset_Execution)
  .link(Execution_Asset).entities()` returns the same 3-row sets
  per execution. Set equality check passed for all three
  (`rids_found == expected` True three times).
- `lookup_execution(rid)` confirms `status=Uploaded` and workflow
  metadata for each.

No disagreement between channels. Both surfaces stayed exactly in
sync — no friction-map finding to file against the read paths for
this arc.

### Friction observed

1. **`Execution` vs `ExecutionRecord` API surface.** The
   `lookup_execution(rid)` call returns an `ExecutionRecord`
   (read-only metadata view), not the `Execution` *handle* that
   has methods like `execution_assets()`. The naming overlap is
   confusing — both are called "Execution" in different contexts.
   Workaround: drop to the path-builder. Not blocking, but the
   first time you hit it from a fresh shell you waste a few
   minutes finding the right API. Filed as
   `findings/developer/01-execution-vs-executionrecord-api.md`.
2. **`pathBuilder` is a method, not a property,** at least at the
   v1.39.4 surface I had pinned. Minor; one-character fix once
   you see the error. Worth noting because the deriva-ml
   docstrings I've seen treat it as a property.

### Handoff to Analyst

**Focus run for analysis:** **XYG** (highest test_acc 42.00%) is
the natural baseline for ROC/confusion-matrix analysis. **YAP**
is the interesting alternative because it carries the only
per-epoch Validation curve and tests the regularized variant.
**XNE** is mostly a smoke-test trophy; include it in the ranking
table so the comparison-spread is visible, but don't expect deep
signal from a 3-epoch run.

**Success metric used:** Final-epoch test accuracy. Cheap to
compute, available in every training log, ranks the runs cleanly.
The Analyst can override if they want top-k or AUC instead — the
predictions CSV per execution has the per-image class probabilities
needed to compute any soft metric (ROC, AUC, calibration).

**Where the outputs live:**
- Weights (per execution): `Execution_Asset RID` ∈ {XQ8, Y0A, YCJ}.
- Per-image prediction probabilities (CSV, ground truth +
  predicted class + 10-class soft probabilities, keyed by Image
  RID): `Execution_Asset RID` ∈ {XQC, Y0E, YCP}.
- Training logs (plain text, epoch-by-epoch loss + acc):
  `Execution_Asset RID` ∈ {XQA, Y0C, YCM}.
- Lineage for any of the 3 executions points back through the
  dataset(s) consumed → through the upstream Curator/loader
  executions. `deriva_ml_get_lineage(<exec_rid>, depth=None)`
  walks the full chain.

**Caveats:**
- The three runs are not strictly comparable: they used different
  datasets *and* different model variants *and* different seeds.
  This is intentional (variety > a-b comparability for the e2e
  exercise) but the Analyst should note it in their report.
- YAP's `val_acc` came from per-epoch evaluation on 100 images —
  small sample, noisy metric. Use it as a sanity check on the
  Validation lane, not as the primary ranking signal.
- TX0 testing partition (TXJ, 150 images) intersects with XEM
  for the YAP run *only if* the Analyst tries to compare XYG and
  YAP on the same test bag. They were trained on the same TX0
  split but YAP also saw the XEM Validation bag; the test metric
  per epoch in YAP's log is still TXJ-only — TXJ and XEM are
  disjoint subsets (TXJ from JZT, XEM from K04). So the rankings
  in the table above are clean: same TXJ test partition for both
  the XYG and YAP rows.

### Open questions for the Analyst

- Should the ranking metric switch to validation accuracy for
  runs that have it? YAP got val_acc 43% — higher than its
  test_acc 37.33% — but only one run carries this metric, so
  including it would skew the ranking. Default: stay on
  test_acc for ranking, surface val_acc as a side column.
- Does the Analyst want a multirun-style sweep before they
  start? If yes, this is the moment — `lr_sweep` and
  `quick_vs_extended` multiruns are pre-wired and would
  produce a tidier parent-child execution graph for the
  denormalize step to chew on. Default: the three runs above
  are enough for an analysis pass.

---
