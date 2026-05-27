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
