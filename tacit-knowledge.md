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

### tk-001 — Curator audit of bootstrapped catalog 96 (e2e-test-20260527d)
**When:** 2026-05-27T06:49:00+00:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/ec40f483-26ae-4a8b-aa24-5155ca94cb22)

Audited the catalog as inherited from Phase 0 (`load-cifar10` loader,
1500 images, 13 datasets, 1 feature, 4 bootstrap executions all status
`Uploaded`). The substrate is clean and the downstream personas can
proceed without curation changes — recording this characterisation so
the Modeler and Analyst do not have to redo the smell test.

Headline numbers (verified against the catalog directly via the
`deriva-ml` Python API; all aggregates re-derived from `Dataset_Image`
and `Execution_Image_Image_Classification`, not taken from descriptions):

- **Class balance is perfect at every level.** All 10 `Image_Class`
  terms have exactly 150 images each in `cifar10_complete` (JZ8). The
  same uniformity holds in every train/test partition: 75/class in
  JZT (cifar10_training, 750 imgs) and K04 (cifar10_testing, 750 imgs);
  50/class in K0W/K16 (small training/testing); 60/class in TX8
  (labeled_training, 600); 40/class in WDA (small_labeled_training,
  400); 15/class in TXJ (labeled_testing, 150); 10/class in WDM
  (small_labeled_testing, 100). No class-imbalance handling is needed
  for any partition.
- **Every Image has exactly one ground-truth label.** 1500 rows in
  `Execution_Image_Image_Classification` covering 1500 distinct images
  — no missing labels, no duplicate labels (no need for the `newest`
  selector when reading the feature). The catalog-wide CLAUDE.md note
  "use labeled datasets for evaluation" is conservative phrasing — in
  *this* catalog every dataset is fully labeled, and the `Labeled`
  Dataset_Type is informational rather than functional.
- **The four `*_split` parent datasets carry zero direct Image
  members** (JZJ, K0M, TX0, WD2). This is by design: their type is
  `Split`, and they carry their training/testing children as nested
  `Dataset` rows. A Modeler should resolve splits to their child
  partitions before consuming. See tk-002 for the leakage map this
  implies.
- **Bag previews report zeros until the bag is materialised.** The
  `ml/dataset/{rid}/bag-preview` resource returns all-zero
  `row_count`/`asset_bytes` and `status: "not_cached"` until
  `ml.cache_dataset(spec)` has run locally. The resource is useful for
  shape-confirmation post-cache, not as an a-priori sizing estimator
  before the first download.
- **Bootstrap executions** (status `Uploaded`): 46Y, FZC, JY8 (loader
  phases) and TW0 (described as "Create Labeled_Split and
  Small_Labeled_Split from the training set" — confirms the
  labeled-family provenance documented in tk-002). All four point at
  one Workflow row (46T) at loader commit
  `990474da1dfedf8cc584461d928b32bd31a47d1c`.

Implications for collaborators:
- The Modeler can pick *any* labeled partition for training; there is
  no rebalancing or filtering work to do up front. The smallest
  smoke-test pair (WDA train / WDM test, 400/100) and the larger main
  pair (TX8 train / TXJ test, 600/150) are both pre-stratified at
  seed=42 and have disjoint contents *within their family* (see
  tk-002 before picking across families).
- The Analyst can read predictions joined to ground truth without
  worrying about null labels — every prediction can be paired with a
  Class.

### tk-002 — Train/eval leakage map for the labeled-family datasets
**When:** 2026-05-27T06:49:00+00:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/ec40f483-26ae-4a8b-aa24-5155ca94cb22)
**Supported by:** tk-001 (audit that established the partition-membership facts this entry interprets)

The bootstrap-time TW0 execution describes itself as "Create
Labeled_Split and Small_Labeled_Split from the training set" —
verified: every Image in TX8 (600), TXJ (150), WDA (400), and WDM
(100) is also a member of JZT (the Toronto-source training partition,
750 images). The labeled-family datasets are stratified resamples of
the training side, *not* of the held-out test side. This is correct
DerivaML behaviour and matches the loader's intent, but it creates
combinations a Modeler can pick that silently leak training data into
evaluation:

| Train on                       | Test on                  | Disjoint? |
|--------------------------------|--------------------------|-----------|
| WDA (small_labeled_training)   | WDM (small_labeled_test) | yes (40/cls vs 10/cls, disjoint within JZT) |
| TX8 (labeled_training)         | TXJ (labeled_testing)    | yes (60/cls vs 15/cls, disjoint within JZT) |
| K0W (small_training)           | K16 (small_testing)      | yes (50/cls vs 50/cls; K0W⊆JZT, K16⊆K04) |
| JZT (cifar10_training)         | K04 (cifar10_testing)    | yes (750 vs 750; canonical Toronto split) |
| **JZT** (cifar10_training)     | **TXJ** (labeled_testing)| **NO — 150 images shared (all of TXJ)** |
| **JZT** (cifar10_training)     | **WDM** (small_lab_test) | **NO — 100 images shared (all of WDM)** |
| **K0W** (small_training, ⊆JZT) | **TXJ**                  | **NO — 101 images shared** |
| **K0W**                        | **WDM**                  | **NO — 69 images shared** |

Background — term-of-art for non-ML readers: **train/test leakage** means
a model is evaluated on images it saw at training time. Performance
numbers from a leaky pair overstate held-out accuracy by an unknowable
margin (5–50%+ depending on how memorised the leaked rows are).

Additional non-obvious property of the small_labeled family: **WDA is
*not* a subset of TX8, and WDM is *not* a subset of TXJ.** Both
labeled splits are seeded with `42` but the stratified-sampling
selection differs because the target sizes differ, so the two
families partition JZT in different ways. An Analyst looking at small
vs full labeled performance should treat them as two independent
seeds against the same underlying source, not as a sub-sample
relationship.

Implications for collaborators:
- **Modeler:** prefer the within-family pairs (WDA→WDM, TX8→TXJ) or
  the canonical Toronto pair (JZT→K04). Avoid the cross-family pairs
  flagged in the table above. The Hydra config defaults
  (`default_dataset = WD2 → WDA+WDM`) are within-family and safe.
- **Modeler:** if a `cifar10_training` (JZT) or `cifar10_small_training`
  (K0W) run is wanted for some reason, the only safe evaluation
  partition is K04 or K16 (the held-out Toronto test side). Do not
  pair JZT with TXJ or WDM.
- **Analyst:** when comparing runs that used different training sets,
  treat WDA-trained, TX8-trained, JZT-trained, and K0W-trained models
  as four separate populations — they share underlying images but in
  different proportions, so their effective training sizes are not
  directly comparable.

No catalog change was made; the existing 13 datasets are correct and
sufficient. This entry documents the rule that prevents the most
likely silent mistake.

---

### tk-003 — Modeler arc: three differentiated runs on the WD2 small-labeled split
**When:** 2026-05-27T07:30:00+00:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/ec40f483-26ae-4a8b-aa24-5155ca94cb22)
**Supported by:** tk-001 (catalog audit — labels and class balance verified clean),
tk-002 (leakage map — pinned the choice of WD2 as a safe within-family pair).

Ran three training executions to (a) confirm the pipeline produces
differentiated output as hyperparameters vary, and (b) hand the
Analyst a 3-run comparison on identical test images so ROC curves
and confusion matrices line up directly. All three executions share
workflow `XDG` (`CIFAR-10 2-Layer CNN`, commit
`4b7f48bdd368…`), all status `Uploaded`, all input dataset = `WD2`
(`cifar10_small_labeled_split`, 400 train / 100 test).

| Exec | model_config | Epochs | Arch | Reg | Train acc | Test acc | Weights size |
|------|--------------|-------:|-----:|-----|----------:|---------:|-------------:|
| `XDP` | `cifar10_quick` | 3 | 32→64ch, 128h | none | 30.25% | **24.00%** | 6.5 MB |
| `XPR` | `default_model` | 10 | 32→64ch, 128h | none | 59.00% | **38.00%** | 6.5 MB |
| `XZT` | `cifar10_extended` | 50 | 64→128ch, 256h | dropout 0.25, wd 1e-4 | 100.00% | **41.00%** | 26 MB |

(Test-accuracy column is the emission-time reading at the final
epoch from the model's own training log; the Analyst should
re-derive it from the prediction CSV joined to ground-truth, per
the `record_test_predictions` docstring.)

**Output asset RIDs** (`weights` / `training_log` / `prediction_probabilities`):

| Exec | weights | log | pred_probs |
|------|--------|-----|-----------|
| `XDP` | `XFG` | `XFJ` | `XFM` |
| `XPR` | `XRJ` | `XRM` | `XRP` |
| `XZT` | `Y1M` | `Y1P` | `Y1R` |

Asset RIDs verified by cross-channel read: direct `deriva-ml`
`PathBuilder` query on `Execution_Asset_Execution` matched the MCP
`deriva_ml_get_execution` summary for all three RIDs.

**Variation took.** The runs *do* differentiate — XPR clearly
beats XDP (more epochs → +14pp test acc, expected). XZT's
behaviour is the textbook overfit signature: 100% training
accuracy and only +3pp over XPR on held-out, with training/test
loss diverging hard around epoch 15 (train→0.01, test→3.8). The
larger arch (4× more parameters: 26 MB vs 6.5 MB weights file) plus
50 epochs blew past WD2's 400-image train pool. That's a useful
signal for the Analyst — and a warning that on this catalog size,
the "extended" config is **not** automatically the best choice.

**Why WD2 for all three (not a sweep over multiple datasets):** the
Analyst's stock comparison surface is ROC curves and confusion
matrices, both of which require predictions on a *common* test
partition to be meaningfully comparable. Holding the dataset
constant and varying only the model_config is the cleanest
controlled experiment the catalog supports. Cross-dataset
comparisons would have introduced confounds the Analyst would
then have to factor out by hand.

**Why no `cifar10_default` experiment as-is:** the stock
`cifar10_default` experiment in `experiments.py` pairs the
`default_model` config with `cifar10_small_training` (K0W) alone
— a single Training-typed dataset with no companion Testing bag.
That trains a model but emits no test predictions and is therefore
invisible to the Analyst's notebook. The (`model_config=default_model`,
`datasets=cifar10_small_labeled_split`) override on the CLI is the
combination an Analyst-facing run actually wants. The stock
experiment is fine as a debug-only path; for any comparison run,
override `datasets=` to a Split.

**Friction noted, not fixed:**
- The XPR description in the catalog reads "Simple model run" —
  the generic `BaseConfig.description` default. When you compose
  a run from `model_config=` + `datasets=` CLI overrides without
  `+experiment=…`, the description doesn't pick up the variant
  intent. The training_log embedded in the execution asset bundle
  *does* carry the hyperparameters, so provenance is recoverable;
  but `deriva_ml_list_executions` output is harder to skim across a
  mixed batch. **Workaround:** pass an explicit
  `description='...'` override or define an experiment config
  before any production run. Not severe enough to block the arc.
- Default Hydra mode emits a screen of `InsecureRequestWarning`s
  from every HTTPS call to the self-signed localhost cert. Each
  successful asset upload also re-prints the warning. Cosmetic.

**Wiring for the Analyst:** populated `src/configs/assets.py`:
- `roc_quick_vs_extended` → `["XFM", "XRP", "Y1R"]` in
  (quick, default, extended) order — matches the notebook's
  expected slot list and feeds the existing `roc_quick_vs_extended`
  notebook config without further edits.
- `quick_weights` → `XFG`, `default_weights` → `XRJ`,
  `extended_weights` → `Y1M` for any test-only inference re-runs.

**For the Analyst:**
- Start with `roc_quick_vs_extended` — three runs, one held-out
  set, ready to plot.
- The "extended" config is *not* the best model here; XPR
  (`default_model`) generalizes about as well at a fraction of the
  cost. Per-class breakdowns and confusion matrices on XZT will
  probably show severe overfit-to-rare-class behaviour.
- Ground-truth labels are in the `Image_Classification` feature on
  `Image` (one per row, per tk-001).

---
