# Experiment Design Decisions

Accumulated rationale for experiment design choices in this project.
Each entry captures what was decided and why.

---

### Reassigned cifar catalog alias to catalog 13

Old `cifar` alias pointed to catalog 11 (empty domain schema). Deleted it, but server
prevents reuse of deleted alias names. Created `cifar10` alias pointing to catalog 13
which has the actual CIFAR-10 domain tables (Image, Image_Class, etc.). Catalog 6
(`cifar10_10k`) was also a candidate but 13 had existing workflows/executions.

### Created FooBar test feature with mixed term + float columns

Created vocabulary `BarValue` (terms: Bar1, Bar2) and feature `FooBar` on Image table
with both a vocabulary term column (BarValue) and a float metadata column (FooValue).
Purpose was to test multi-column feature creation and multivalued feature workflows.
Added two executions (3WY2 seed=42, 3X4R seed=99) with 100 random values each to
exercise the multivalued feature path.

### Added select_by_execution to deriva-ml FeatureRecord

`fetch_table_features` supported `select_newest` and `select_by_workflow` for resolving
multivalued features, but workflow-level filtering is insufficient when multiple
executions share the same workflow. Added `FeatureRecord.select_by_execution(rid)` as
a static method returning a closure-based selector. Chose this over adding an
`execution` parameter to `fetch_table_features` directly because the selector pattern
is more composable and consistent with the existing API. Also added `execution`
parameter to the MCP `fetch_table_features` tool (deriva-ml v1.23.3, deriva-mcp v0.10.4).

### Added description guidance and git workflow to deriva-mcp skills (v0.10.6)

Added entity-specific description guidance to 9 skills (configure-experiment, write-hydra-config, generate-descriptions, create-dataset, create-feature, create-table, dataset-versioning, manage-vocabulary) so good description practices are available at the point of action. Added critical rule #5 to configure-experiment requiring goal-oriented experiment descriptions rather than just technical parameters. Updated feature selection documentation in create-feature to include the new `execution` parameter. Added branch-based git workflow (branch → develop → PR → merge → bump version) to the new-model skill, with guidance that PRs are valuable even for solo developers and that Claude Code can handle PR creation/merge via the GitHub CLI.

### Default bag fetch concurrency set to 1 in deriva-ml

bdbag's HTTP fetcher shares a single `requests.Session` across threads in its
`ThreadPoolExecutor`. `requests.Session` is not thread-safe — with concurrency=8,
cookie jar corruption causes ~60% of Hatrac fetches to fail silently with HTTP 401.
Diagnosed by comparing serial fetch (0 failures) vs concurrent fetch (244/686 failures)
on eye-ai dataset 4-411G. Changed default `fetch_concurrency` from 8 to 1 in
`download_dataset_bag`, `cache`, and `_materialize_dataset_bag`. Chose to fix the
default rather than patch bdbag because the upstream bug is in shared mutable state
across threads, and per-thread sessions would require a bdbag fork.

### Fixed duplicate association table denormalization bug (deriva-ml v1.26.3)

Bag-based denormalization returned empty DataFrames on eye-ai dataset 4-411G because
the schema has two association tables linking Dataset to Image: `Image_Dataset` (343 rows)
and `Dataset_Image` (0 rows). `_schema_to_paths()` discovered paths through both;
`_prepare_wide_table()` merged them into a single join graph. The INNER JOIN through the
empty table produced 0 rows. Fixed by deduplicating paths in `_prepare_wide_table()` that
reach the same `(element, endpoint)` via different association tables, using
`is_association(pure=False)` for structural detection rather than naming conventions.
Added `Image_Dataset_Legacy` to the demo schema as a regression test. Verified fix
produces 347 rows x 18 columns for `['Subject', 'Clinical_Records']` on 4-411G.

### E2E platform-test baseline run on catalog 46 (execution CMY)

Hypothesis: post-bugfix sweep (PRs landed across deriva-py, deriva-ml,
deriva-ml-mcp, deriva-ml-skills, template) lets cifar10_quick train end-to-end
against catalog 46 with RID-aware prediction recording. Picked cifar10_quick +
cifar10_small_labeled_split for the smallest reproducible loop (200 train /
50 test, 3 epochs, batch 128). Run finished in ~6s on CPU. Test accuracy 94%
(47/50, 3 ship→bird misclassifications, all at confidence > 0.98). 50 per-image
classification predictions recorded as Image_Classification feature values with
RIDs surfaced via the new (sample, target, rid) tuple shape (deriva-ml #169).
3 Execution_Assets uploaded (weights, training log, predictions CSV). This
became the canonical baseline for Phases 3, 6, 9, and 10 of the e2e test.

### Confidence bucket thresholds 0.70 / 0.95 (feature Prediction_Confidence_Bucket)

Created `Prediction_Confidence_Bucket` feature on Image (term column referencing
new `Confidence_Bucket` vocabulary: Low / Medium / High) for human-readable
triage of model predictions. Use case: a coarse bucket on top of the continuous
Confidence score in Image_Classification helps surface high-confidence
misclassifications — the most valuable triage signal in this run, given CMY's
hard convergence. Thresholds Low (<0.70) / Medium ([0.70, 0.95)) / High (≥0.95)
chosen so Low flags review candidates, High is reliable for downstream use.
Bucketing CMY's 50 test predictions produced 45 High / 5 Medium / 0 Low, and
crucially all 3 ship→bird misclassifications bucket as High (confidence
0.98–0.99) — exactly the high-confidence-error signal the feature was designed
to surface. Population happened inside execution DER. Rejected scalar-only
feature in favor of term-based: terms are stable across schema changes and
queryable as first-class taxonomy.

### Phase 7 subset filter substituted bird-only for cat+dog+frog (dataset E6R v0.2.0)

E2E test spec called for a 3-class subset (cat+dog+frog) of dataset 85J to
exercise the curated-subset path. Discovery: 85J's 250-image partition contains
only 2 of 10 classes (225 bird, 25 ship) — load-cifar10's small-partition
sampling biases to whichever classes the source data orders first. Substituted
bird-only filter (225 images) because that path exercises the same
create_dataset + add_dataset_members code the spec was probing — only the
filter classes differ. Rejected dropping the subset test (loses signal);
rejected switching source dataset (the complete 500-image dataset 850 has 4
classes — airplane/bird/ship/truck — but is larger and slower to test).
Dataset E6R released to 0.2.0 so it can be pinned in an experiment config;
registered as `cifar10_phase7_bird_subset_localhost`. Empty E5M dataset remains
as a Phase 7 artifact (created during the zero-match cat/dog/frog dry-run;
classifier-blocked from deletion, journaled rather than removed).

### Two e2e bugs were misdiagnosed in the journal; verify before fixing

Two findings from this session — B18 (`lookup_asset` "stale executions") and
B19 (ROC notebook 98% vs 94%) — had hypotheses recorded in the journal at
discovery time that turned out to be wrong. B18 was filed as a `get_ml()`
cache-staleness bug; the actual cause (caught during PR #43) was a blanket
`try/except Exception` in `tools/asset.py` swallowing transient ermrest errors
into `executions = []`. B19 was filed as a `gt_lookup` filter bug in
`roc_analysis.ipynb` cell 9; the actual cause (caught during PR #16) was that
`_initialize_execution` wrote two assets with the same `Filename` to the same
path, silently overwriting the first — a deriva-ml bug fixed in PR #179
(RID-key the dest_dir). Lesson: a journal hypothesis written at discovery
time can persist through issue body, review, and even the start of the fix.
Verify with a reproduction before settling on a fix scope; if the agent
investigating finds a different cause, trust the reproduction. Both PRs
landed correct fixes only because the implementing agent reproduced before
patching.
