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

**When:** 2026-05-27 (third-pass multipersona run, post-path_walker + cifar10_cnn refactor)
**By:** Orchestrator (Phase 0)
**Supported by:** —

### What was set up

Fresh catalog **`e2e-test-20260527c`** (catalog id `95`) on `localhost`.
Bootstrapped via `load-cifar10` in three phases (schema → images →
datasets). 1500 Image rows (750 train + 750 test), 1500
`Image_Classification` features, perfectly balanced class
distribution (150 per CIFAR-10 class × 10 classes). 13 datasets
created — the canonical hierarchy:

- `JZ8` Complete (1500) — root container
- `JZJ` Split → `JZT` Training (750) + `K04` Testing (750)
- `K0M` Small_Split → `K0W` Small_Training (500) + `K16` Small_Testing (500)
- `TX0` Labeled_Split → `TX8` Labeled_Training (600) + `TXJ` Labeled_Testing (150)
- `WD2` Small_Labeled_Split → `WDA` (400) + `WDM` (100)

All RIDs wired into `src/configs/datasets.py`. Default Hydra
connection (`default_deriva`) and dataset config (`default_dataset`
→ `WD2`) already point at catalog 95.

### Sibling versions

- deriva-ml: `v1.39.4` (`4ed88122`) — includes PR #246 PagedFetcher
  row-completeness fix, PR #243 resolver dedup, PR #237 defensive
  one-liners
- deriva-py: pinned via `@deriva-ml` branch at `e944ad8e` —
  includes `SchemaPathWalker` (PR #263)
- deriva-ml-mcp: `v0.5.4`
- deriva-ml-skills: `v1.4.8` (latest e2e polish + concepts.md
  cleanup)
- deriva-mcp-test container: rebuilt against above

### Notable platform changes since 2026-05-27 (first pass)

- **cifar10_cnn refactor (PR #37)**: the runner is now ~730 lines
  with three clearly-labeled sections (ML primitives / DerivaML
  harness / entry point). Eval logic collapsed into one
  `evaluate()` primitive; predict logic into `predict_batch`;
  bag-to-loader dispatch into one `build_loaders` function. The
  Developer arc will exercise this refactored surface.
- **PagedFetcher fix (PR #246)** prevents the 50% row-loss bug
  that blocked the first 2026-05-27 attempt at `--num-images 1500`.
- **path_walker pin (PRs #38/#59)** unblocks the denormalize
  planner's runtime import; was the blocker on this run's second
  attempt.

### Handoff to Curator

- Catalog id, schema, image counts, and dataset RIDs are visible
  via the standard MCP resources — query them, don't read them
  out of this file.
- Class distribution is perfectly balanced (150 per class × 10).
  Any skew the Curator finds is a finding against the loader.
- `default_dataset` is `WD2` (small labeled split) — the typical
  fast-run choice. Override with `datasets=cifar10_labeled_split`
  (or others) for larger runs.

---

## tk-002 — Catalog 95 audit (clean)

**When:** 2026-05-27 (Curator arc, multipersona pass C)
**By:** Curator
**Supported by:** tk-001

### Audit results

Cross-channel verified (MCP resources ↔ direct deriva-ml Python API)
against catalog 95. **Everything tk-001 promised is exactly what's
in the catalog.** No findings.

| Check | MCP | Direct | Match |
|---|---|---|---|
| Image rows | 1500 (via JZ8 summary) | 1500 (pathBuilder fetch) | yes |
| Image_Classification features | 1500 (1/Image) | 1500 | yes |
| Class distribution | not directly queryable via resource | 150/class × 10 (perfectly balanced) | yes |
| Dataset count | 13 | 13 | yes |
| JZ8 members | 1500 Images, 0 Datasets | 1500 / 0 | yes |
| JZJ / K0M / TX0 / WD2 (Split containers) | 2 nested datasets each | 2 / 2 / 2 / 2 | yes |
| JZT / K04 (full partitions) | 750 each | 750 / 750 | yes |
| K0W / K16 (small partitions) | 500 each | 500 / 500 | yes |
| TX8 / TXJ / WDA / WDM (labeled subsets) | 600/150 and 400/100 | match | yes |
| K04 class balance | not directly queryable | 75/class × 10 | balanced |
| Image_Class vocabulary | 10 terms (canonical CIFAR-10) | 10 | yes |
| Dataset_Type vocabulary | 8 terms (incl. Validation, rid 3KT) | n/a | n/a |

### Methodology notes

- The Image_Classification feature table lives in the **domain
  schema** (`e2e-test-20260527c.Execution_Image_Image_Classification`),
  not in `deriva-ml`. The MCP `features/Image` resource correctly
  surfaces it; a naive direct query against `deriva-ml.<table>` will
  KeyError, which is a one-line tweak the Developer should know.
- The Image_Class vocabulary is also in the domain schema, not
  `deriva-ml`. The Dataset_Type / Workflow_Type / Asset_Type
  vocabularies live in `deriva-ml`.

---

## tk-003 — Handoff to the Model Developer

**When:** 2026-05-27 (Curator arc, multipersona pass C)
**By:** Curator
**Supported by:** tk-001, tk-002

### What's curated

A new Validation slice **`XEM` (`cifar10_validation_150`)** has been
carved from K04 (Testing partition). 150 images, 15 per CIFAR-10
class, stratified deterministically via
`random.Random(20260527).shuffle` on sorted image RIDs.

- **Why:** PR #29 made the cifar10_cnn runner dispatch a Validation
  lane when a `Dataset_Type=Validation` member is present; the
  bootstrap left no such dataset. Without one, the Developer would
  either peek at K04 (contaminating held-out evaluation) or build
  an ad-hoc per-run slice (not reproducible across runs).
- **Source:** K04 (`Dataset_Type=[Testing, Labeled]`, 750 images
  total). The script reads K04 membership + the
  `Execution_Image_Image_Classification` feature values to bucket
  by class, then takes the first 15 RIDs per class after a seeded
  shuffle.
- **Reproducibility:** Re-running
  `scripts/curator_create_validation.py` against the same K04
  members yields the same 150 RIDs by construction (seeded RNG +
  sorted pools). The dataset itself was created under a
  Dataset_Management workflow with execution RID `XDM`.
- **Containment:** K04 is **not modified** — XEM is a slice, not
  a partition. The Developer can train on JZT / K0W / WDA / TX8
  (Training partitions), validate on XEM, and still hold K04 out
  as the canonical Testing partition.

### Use-case → dataset table

| Downstream use case | Use this RID | Config name |
|---|---|---|
| Quick smoke run (small data) | `WD2` (Split: WDA + WDM) | `cifar10_small_labeled_split` (default) |
| Full Toronto-source training | `JZT` (Training) | `cifar10_training` |
| Full Toronto-source eval | `K04` (Testing) | `cifar10_testing` |
| Stratified labeled training | `TX8` (600 images) | `cifar10_labeled_training` |
| Stratified labeled eval | `TXJ` (150 images) | `cifar10_labeled_testing` |
| **Validation during training** | **`XEM` (150 images)** | **`cifar10_validation_150`** |
| End-to-end on full data | `JZ8` (1500, root) | `cifar10_complete` |
| Multi-input train+val+test | combine `JZT` + `XEM` + `K04` | compose via Hydra |

### Gotchas

- The Image_Classification feature and Image_Class vocabulary both
  live in the **domain schema** (`e2e-test-20260527c`), not in
  `deriva-ml`. Hard-code accordingly when querying directly; MCP
  resources handle this automatically.
- `K04` and `XEM` overlap by design — XEM ⊂ K04 (150 of K04's 750
  images). Using both in the same execution would double-count
  those 150 images. Treat them as alternative partitions, not
  composable ones.
- XEM's stratified balance (15/class) matches K04's balance
  (75/class) — the seeded shuffle preserves uniform class
  representation. If a downstream consumer needs more imbalance
  (e.g., long-tail evaluation), file a separate curator request;
  don't re-balance XEM in place.

### What the Developer can assume

- Class distribution: perfectly balanced everywhere. Any per-class
  metric skew the Developer sees is from the *model*, not the
  data.
- All 1500 Image_Classification features carry a non-null
  `Image_Class` term column; `Confidence` is also populated
  (loader-provided ground truth).
- Workflow types `Image Classification` (rid 46M) and
  `Training` (rid 3M2) are both already registered — no need to
  extend Workflow_Type.
- `cifar10_validation_150` is wired into
  `src/configs/datasets.py` and immediately usable as a Hydra
  override: `datasets=cifar10_validation_150`.

---

## tk-004 — Developer arc: three executions, refactored runner exercised

**When:** 2026-05-27 (Developer arc, multipersona pass C)
**By:** Developer
**Supported by:** tk-001, tk-002, tk-003

### What was run

Three training executions against catalog 95 via
`uv run deriva-ml-run` (with `DERIVA_ML_ALLOW_DIRTY=true`). All
three drove the refactored `cifar10_cnn.py` (PR #37): one
`build_loaders` dispatch + one shared `evaluate()` + one
`predict_batch` → `record_predictions` split with `Source_Label`
tagging.

### Ranked executions (by final-epoch test_acc)

| Rank | Exec RID | Dataset(s) in       | Model variant                   | seed | epochs | train_acc | test_acc | val_acc | Source_Label CSV |
|-----:|----------|---------------------|---------------------------------|-----:|-------:|----------:|---------:|--------:|------------------|
| 1    | **Y1M**  | TX0 (Labeled_Split) | cifar10_quick (32/64, bs=128)   | 123  | 10     | 58.83%    | **38.00%** | —     | `epoch_10`       |
| 2    | **YDT**  | TX0 + XEM (Val)     | default_model (32/64, bs=64)    | 7    | 10     | 65.00%    | 36.67%   | 34.67%  | `epoch_10`       |
| 3    | **XRJ**  | WD2 (Small_Labeled) | cifar10_quick (32/64, bs=128)   | 42   | 3      | 30.25%    | 24.00%   | —       | `epoch_3`        |

All three uploaded weights (`cifar10_cnn_weights.pt`),
`training_log.txt`, and `prediction_probabilities.csv` as
`Execution_Asset` rows. CSV `Source_Label` populated correctly
(`epoch_3` / `epoch_10`). Status = `Uploaded` for all three.
Workflow RID = `XRC` (shared, registered by deriva-ml-run on first
call).

Wall-clock training time: XRJ ~1s, Y1M ~4s, YDT ~4s. None
approached the 5-minute finding threshold.

### Decision log

**Datasets — why each:**

- **WD2 (Run 1 / XRJ):** smoke test. Quickest possible loop to
  confirm the refactored runner builds DataLoaders, trains, saves,
  and uploads. WD2 is also the `default_dataset` per tk-001, so
  this exercises the zero-override path.
- **TX0 (Run 2 / Y1M):** mid-data scale (600 train / 150 test).
  Larger than WD2 (more signal for the test_acc ranking) but still
  fast (~4s wall-clock). Picked TX0 over JZJ because it's
  stratified + labeled on both sides — clean comparison with Run 3.
- **TX0 + XEM (Run 3 / YDT):** exercise the PR #29 Validation
  dispatch lane (D01). XEM was newly carved by the Curator (tk-003)
  precisely for this. Composed via a new
  `cifar10_train_with_validation` Hydra config in
  `src/configs/datasets.py` — DatasetSpecConfig accepts a list,
  so `[TX0, XEM]` in one entry produces a single execution with
  both bags. `build_loaders` then flattens TX0 (Split) to
  TX8+TXJ and routes XEM to the val_loader. Per-epoch `val_acc`
  surfaces in the training log (it does NOT drive save-best —
  that's intentional per the refactor's design).

**Hyperparameters — why these:**

- Three different seeds (42 / 123 / 7) to confirm the
  byte-reproducibility surface added by D02 — same model, three
  different RNG trajectories, three different test_acc curves.
  Not designed to pick a winner — designed to confirm seeds
  actually do something.
- Held architecture nearly constant (32/64 channels). Y1M and YDT
  differ only in dataset, seed, batch size, and the val_loader
  presence — that lets the analyst attribute differences to the
  Validation lane vs the underlying model.
- Epoch budget: 3 for the smoke, 10 for the two main runs.
  Sufficient to see test_acc lift off the floor (24% → 38% across
  the three runs) without burning analyst's time. Longer runs are
  available via `+experiment=cifar10_extended` but unnecessary
  for the platform-stress goal.

**Composite dataset config (Run 3):**

`cifar10_train_with_validation` in `src/configs/datasets.py` is
the multi-input pattern from tk-003. Single store entry, two
DatasetSpecConfigs in the list. `build_loaders` does the rest.
This is the canonical shape for any future "train + val" run on
this catalog.

### Cross-channel verification

Verified MCP ↔ direct deriva-ml agree on every claim:

| Check | MCP | Direct | Match |
|---|---|---|---|
| XRJ status | Uploaded | Uploaded | yes |
| Y1M status | Uploaded | Uploaded | yes |
| YDT status | Uploaded | Uploaded | yes |
| XRJ inputs | WD2 | [WD2] | yes |
| Y1M inputs | TX0 | [TX0] | yes |
| YDT inputs | TX0 + XEM | [TX0, XEM] | yes |
| YDT lineage parents (data-flow) | XDM (XEM creator) + TW0 (TX0 splitter) | n/a | resource-only |
| Asset count / exec | 9 (3 Execution_Asset + 6 metadata) | 9 | yes |
| Source_Label in CSV | n/a (not surfaced in MCP) | populated (`epoch_3`/`epoch_10`) | yes |

MCP `deriva_ml_get_lineage(YDT, depth=1)` cleanly walked back to
both XDM (the Curator's XEM-creator execution) and TW0 (the
upstream TX0 splitter) — provenance graph is intact.

### Findings

One filed: `findings/developer/01-emission-time-accuracy-missing.md`
— low severity, cosmetic. The "Emission-time accuracy: NN.NN%"
log line the prompt promised does not actually exist in the
refactored runner; `record_predictions` only prints
`Recorded N predictions (source_label='epoch_K')`. The analyst
can still reconcile test_acc vs CSV by joining the CSV's
`Predicted_Class` against `Image_Classification` ground truth,
but the in-process redundant channel is absent.

Tangential observation (in the finding, not a separate file):
YDT's execution `description` reads `"Simple model run"` because
it used a bare Hydra-override chain instead of `+experiment=...`.
Not a bug — just a documentation gap for users who compose
overrides manually.

### Handoff to the Analyst

- **Focus on Y1M and YDT** for the ranking + ROC. XRJ is the
  smoke run and its 3-epoch curve sits well below the others; it's
  in the table for completeness, not for headline ranking.
- **For denormalize exercise**, the cleanest targets are:
  - **TX0** (the Y1M input + half of YDT's input) — labeled
    Split, 600+150 images, two dataset_types (Training, Testing,
    Labeled).
  - **JZ8** (the 1500-image root) — biggest, exercises PR #246
    PagedFetcher completeness fix at 1500 rows.
- **Source_Label is in the CSV, NOT in the catalog feature row.**
  `record_predictions` documents this explicitly
  (`cifar10_cnn.py:407-408`): "The catalog feature row does NOT
  carry Source_Label (would require a schema migration); CSV is
  the source-label surface." If the analyst needs to filter
  prediction features by run, they should use the CSV asset, not
  the `Image_Classification` feature table.
- **YDT's val_acc trajectory** is in the training_log.txt
  (asset YFR). It rises 22% → 34.67% over 10 epochs — modest but
  reflects the small (150-image) Validation set. The analyst
  doesn't need to do anything with this; it's there as evidence
  the Validation lane works.
- **For ROC notebook execution**, both Y1M and YDT produce 150-row
  test predictions on the same TXJ test bag (Y1M directly, YDT
  via TX0's child). Their CSVs are directly comparable.
- **Wire executions into `src/configs/assets.py`?** No — for this
  arc the analyst pulls assets via `lookup_execution(rid)`
  rather than asset_RID, so wiring is unnecessary. Could be
  added if a follow-up notebook needs to test-load weights.

### What I would want to know if this arc breaks again

- **PR #29 dispatch lane** absolutely works — TX0 (Split) +
  XEM (Validation) coexist cleanly. The only ambiguity is
  *which* test bag wins when a Split contains a Testing partition
  AND a separate Validation bag is also present; build_loaders
  picks the first matching role per bag, and Split-children
  inherit their parent's order. For (TX0=[TX8 Train, TXJ Test],
  XEM=Val) the test_loader is TXJ. Good.
- **The composite dataset config pattern is `[Spec1, Spec2]` in
  one `datasets_store` entry**, not two separate entries. This
  isn't obvious from the existing examples — every other entry
  has a 1-element list.
- **WD2 has no Validation child by design** (Small_Labeled_Split
  was carved before XEM existed). Don't try to compose
  `WD2 + XEM` — XEM was carved from K04, not from WD2's pool.

---
