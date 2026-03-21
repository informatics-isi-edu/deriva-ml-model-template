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
