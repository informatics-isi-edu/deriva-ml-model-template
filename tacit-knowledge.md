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
