# E2E Platform Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end integration test of the DerivaML platform stack
(deriva-ml, deriva-mcp-core, deriva-ml-mcp, deriva-skills,
deriva-ml-skills) using the model template as harness. Skills-first
execution; dual-channel verification (direct deriva-py/deriva-ml vs
indirect MCP/skills) at every phase boundary; inline fixes for any
finding.

**Architecture:** Phases 1–11 are the test-session execution playbook
— each phase is a checkable task with explicit skill / direct check /
indirect check / diff / user inspection checkpoint structure. The
load-cifar10 refactor and source-swap work that originally seeded this
plan as "Phase 0" has been merged onto main and removed from this
document; it lives in git history under the
`load-cifar10`-touching commits if you need to reconstruct it.

**Tech Stack:** Python 3.11+, `uv`, `deriva-py`, `deriva-ml`,
`deriva-mcp-core`, `deriva-ml-mcp`, `pytest`, Hydra-Zen, PyTorch
(CIFAR-10 CNN), Jupyter (ROC notebook), Claude Code skill plugins.

**Spec:** `docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md`

**Date convention:** Throughout this plan, the literal string
`2026-05-13` is the **plan's authoring date**, not the test-run date.
For each actual test execution, substitute the current session's
date (YYYY-MM-DD) into:
- the worktree branch name (`e2e-test/<today>`)
- the journal file path (`docs/e2e-test-<today>-journal.md`)
- domain-schema names like `e2e-test-<today>` (used by `load-cifar10`'s
  `--create-catalog` argument)
- the session-journal header ("E2E Platform Test Session Journal —
  <today>").

The plan/spec **file paths** keep their authored date (they're the
permanent reference documents). The most recent prior session journal
was `docs/e2e-test-2026-05-19-journal.md`; consult it for an example
of how the substitutions look in practice.

---

## File Structure

### Files created during a test session

- The session journal at `docs/e2e-test-<today>-journal.md` is created
  during Task B0.

### Files modified during a test session

- `src/configs/dev/*_localhost.py` — repointed during Task B1 (test
  mutation, dropped at session end).

### Files NOT changed

- `src/scripts/load_cifar10.py` and its helper modules — stable; only
  invoked as a CLI.
- `src/models/cifar10_cnn.py` — model code unchanged.
- `src/configs/` (non-dev) — checked-in catalog-agnostic configs
  unchanged.
- `notebooks/roc_analysis.ipynb` — notebook unchanged.

---

# Phases 1–11 — test session execution

Each task corresponds to one phase from the spec. Within a task, the
steps are: try-skill, direct-check, indirect-check, diff,
journal-entry, user-inspection-checkpoint.

Once a task starts I will NOT proceed past the user-inspection
checkpoint without explicit user "ok, continue."

## Task B0: Worktree setup and journal initialization

**Files:**
- Create: `/Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md`
- New worktree at `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e`

- [ ] **Step 1: Create worktree on test branch**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
git worktree add ../deriva-ml-model-template-e2e -b e2e-test/2026-05-13
```
Expected: worktree created; checked out to branch `e2e-test/2026-05-13`.

- [ ] **Step 2: Sync dependencies in the worktree**

Run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
uv sync
```
Expected: success.

- [ ] **Step 3: Initialize the session journal**

Create `/Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md`:

```markdown
# E2E Platform Test Session Journal — 2026-05-13

**Spec:** `deriva-ml-model-template/docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md`
**Plan:** `deriva-ml-model-template/docs/superpowers/plans/2026-05-13-e2e-platform-test.md`
**Worktree:** `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e`
**Test branch:** `e2e-test/2026-05-13`

## Tag legend

- `#bug-fixed` — bug found and fixed inline
- `#skill-issue` — skill mis-routed / failed / wrong behavior
  - `#refined` — existing skill updated
  - `#new-skill` — new skill created
  - `#eval-added` — eval added to lock in correct behavior
- `#tool-issue` — MCP tool missing / broken / bad description
- `#doc-gap` — README/CLAUDE.md/skill doc wrong or incomplete
- `#surprise` — worked unexpectedly; rationale captured
- `#cache-miss` — cache should serve but doesn't (Phase 5)
- `#diff` — direct and indirect channels disagree

## Session timeline

### 2026-05-13 — Session start

Worktree and journal initialized. Phase 0 (CIFAR-10 source refactor)
landed on main: see commits on origin/main above this session.
```

- [ ] **Step 4: Verify journal path**

Run:
```bash
ls -la /Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md
```
Expected: file exists.

- [ ] **Step 5: User confirms worktree + journal ready**

Print to the user:

> Worktree at `../deriva-ml-model-template-e2e`, branch
> `e2e-test/2026-05-13`. Journal at `docs/e2e-test-2026-05-13-journal.md`.
> Ready to start Phase 1? (yes/no)

Wait for "yes."

## Task B1: Phase 1 — Catalog bootstrap and config repointing

**Inputs:** clean worktree, journal initialized.
**Outputs:** fresh catalog `e2e-test-20260513`, dev configs repointed,
journal entry written, user has inspected the catalog.

- [ ] **Step 1: Look up `maintain-experiment-notes` write location**

Per spec §9.2, this needs to be answered before the skill is used.
Read the skill file:

```bash
cat /Users/carl/.claude/plugins/cache/*/deriva-ml-skills/*/skills/maintain-experiment-notes/*.md
```
(or wherever the installed plugin path is — `ls ~/.claude/plugins/`
to find it).

Note in the journal: where does the skill write? If RIDs in catalog,
note as `#skill-issue` and continue with workaround (export before
cleanup); if repo files, note the path and continue.

- [ ] **Step 2: Try the bootstrap routing skill**

In `deriva-ml-skills` 1.3.5+, the bootstrap-routing skill is
`deriva-ml:setup-ml-catalog` (it replaced 1.2.1's `route-project-setup`).
Phrase the request as a user would: *"Set up a fresh CIFAR-10 catalog
on localhost for end-to-end testing."*

Observe: which skill fires? Does it route to `load-cifar10`?

**Design note (not a `#skill-issue`):** `setup-ml-catalog` is marked
`disable-model-invocation: true` in 1.3.5. An agent-driven test run
**cannot** invoke it — only an explicit user `/deriva-ml:setup-ml-catalog`
slash command can. The plan therefore expects the agent to fall back
to the CLI per Step 3, and the journal records the gate as a design
observation rather than a routing bug.

If the skill does fire when the user invokes it and routes wrong,
that **is** a `#skill-issue`. Apply the meta-loop from spec §4:
diagnose, fix via `skill-creator`, reload, re-attempt. Journal it.

- [ ] **Step 3: Execute the catalog-creation command**

Whether routed by skill or by fallback, run:
```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
uv run load-cifar10 --hostname localhost --create-catalog e2e-test-20260513 --num-images 500 --show-urls
```
Expected: catalog created; ~13 datasets created; ~500 images uploaded
across train+test; Chaise URLs printed for each dataset.

Record the catalog ID printed in the summary banner.

- [ ] **Step 4: Direct catalog check via deriva-ml**

In a Python REPL or one-off script:

```python
from deriva_ml import DerivaML

ml = DerivaML(hostname="localhost", catalog_id="<NEW_CATALOG_ID>")

# Schema check
assert "Image" in ml.model.schemas[ml.default_schema].tables
assert "Image_Class" in [t.name for t in ml.model.schemas[ml.ml_schema].tables.values()] or \
       "Image_Class" in [t.name for t in ml.model.schemas[ml.default_schema].tables.values()]

# Vocabulary terms
terms = {t.name for t in ml.list_vocabulary_terms("Image_Class")}
assert terms == {"airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"}, f"got {terms}"

# Dataset count + names
datasets = ml.find_datasets()
print(f"Found {len(datasets)} datasets:")
for d in datasets:
    print(f"  {d.dataset_rid}  v{d.current_version}  {d.description[:60]}")

expected_dataset_count = 13
assert len(datasets) == expected_dataset_count, f"got {len(datasets)}"

# Feature exists
features = ml.list_features()
assert any(f.feature_name == "Image_Classification" for f in features)
```

Record direct-check results in journal under "Direct check."

- [ ] **Step 5: Indirect catalog check via MCP tools**

Invoke these MCP tools and record results in the journal. The tool
namespaces split across the two plugins on the MCP server:
`deriva-mcp-core` owns generic catalog/schema/vocabulary tools (no
`deriva_ml_` prefix); `deriva-ml-mcp` owns ML-domain tools (with
`deriva_ml_` prefix).

- `deriva_ml_list_datasets(hostname="localhost", catalog_id=<id>)` —
  expect same count + RIDs as the direct check.
- `list_schemas(hostname="localhost", catalog_id=<id>)` —
  expect to include the domain schema (e.g. `e2e-test-20260520`) and
  `deriva-ml`. (There is no `deriva_ml_list_vocabularies` tool in
  the current MCP surface; vocabulary tables are discoverable via
  `list_schemas` + `get_table`, or via the next bullet for their
  contents.)
- `list_vocabulary_terms(hostname="localhost", catalog_id=<id>, schema="<domain_schema>", table="Image_Class")` —
  expect the 10 CIFAR class terms.
- `deriva_ml_list_features(hostname="localhost", catalog_id=<id>, table="Image")` —
  expect `Image_Classification`. (Parameter is `table`, not
  `target_table` — the latter appears in the response JSON but isn't
  the tool's input param.)

- [ ] **Step 6: Diff direct vs indirect**

Compare results from steps 4 and 5. Any disagreement is a `#diff`
finding. If the disagreement involves dataset metadata, drop one
level lower (raw `ml.catalog.get(...)` ERMrest request) to localize
to deriva-ml vs MCP server.

For each `#diff`: diagnose (deriva-ml bug, MCP server bug, or
test methodology issue). Fix inline per spec §2.4 (refresh
workspace after fix). Re-run the indirect check until it agrees.

- [ ] **Step 7: Repoint dev configs**

Edit the four files in the worktree:

- `src/configs/dev/deriva_localhost.py` — set catalog_id to the new ID.
- `src/configs/dev/datasets_localhost.py` — for each dataset config,
  replace the dataset RID + version with the new ones from step 4.
- `src/configs/dev/assets_localhost.py` — clear out stale model-weights
  and prediction asset RIDs (they'll be populated after Phase 2 and 4).
- `src/configs/dev/roc_analysis_localhost.py` — same: clear out stale
  asset RIDs (populated before Phase 9).

Run:
```bash
uv run python -m pytest tests/test_configs_load.py -v
```
Expected: PASS (smoke test — configs are loadable).

- [ ] **Step 8: Commit the repoint as a DROP commit**

```bash
git add src/configs/dev/
git commit -m "test: [E2E-DROP] repoint dev/*_localhost.py at catalog <new_id>"
```

- [ ] **Step 9: Write Phase 1 journal entry**

Append to `/Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md`:

```markdown
### 2026-05-13 HH:MM — Phase 1: catalog bootstrap

**Skill tried:** deriva-ml:setup-ml-catalog (gated: disable-model-invocation)
**Routed to:** <load-cifar10 ... | none — fell back to CLI per the gate>
**MCP tools used:** deriva_ml_list_datasets, list_schemas,
list_vocabulary_terms, deriva_ml_list_features
**Catalog created:** <id>
**Direct/indirect diff:** ✓ agree | ✗ <details>
**Findings:** <none | list with tags>
**Decisions:** <e.g., "skipped --show-urls because already showing in MCP indirect">
```

Tag any findings.

- [ ] **Step 10: User inspection checkpoint**

Print to the user:

> Phase 1 complete. Catalog `<id>` ready.
>
> **Chaise URLs printed in the load-cifar10 output above.**
> Key entities:
> - Catalog: <id>
> - Domain schema: e2e-test-20260513
> - Image_Class vocab: <10 terms>
> - Datasets: 13 (see direct check above for full list with RIDs)
> - Image_Classification feature: created
>
> Inspect the catalog as needed (Chaise, deriva-py REPL, MCP tools).
> Anything off? Fix immediately or defer per spec §4 user-inspection
> conventions.
>
> Ready to start Phase 2 (quick training)? (yes/defer/fix-then-continue)

Wait for explicit user response.

## Task B2: Phase 2 — Quick training (dry-run, then real)

- [ ] **Step 1: Try `deriva-ml:execution-lifecycle` skill (dry-run)**

User-style request: *"Run the cifar10_quick experiment as a dry run."*

Observe: does the skill fire and route to
`deriva-ml-run +experiment=cifar10_quick dry_run=true`?

If not, `#skill-issue` → meta-loop.

- [ ] **Step 2: Execute dry-run**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e
uv run deriva-ml-run +experiment=cifar10_quick dry_run=true
```
Expected: configs resolve, but no catalog writes. Should print the
resolved Hydra config and exit successfully.

If dry-run fails for any reason: fix inline, re-attempt.

- [ ] **Step 3: Execute real training run**

```bash
uv run deriva-ml-run +experiment=cifar10_quick
```
Expected: training runs (3 epochs), execution + assets created in
catalog. Record execution RID printed in logs.

If catalog is dirty-tree-blocked: use `DERIVA_ML_ALLOW_DIRTY=true uv run ...`
(see workspace CLAUDE.md). This is expected on the test branch.

- [ ] **Step 4: Direct catalog check — workflow + execution + assets**

```python
ml = DerivaML(hostname="localhost", catalog_id="<id>")

# Find the workflow row
workflows = ml.list_workflows()
quick_wf = [w for w in workflows if "cifar10" in w.name.lower() and "quick" in w.name.lower()]
assert len(quick_wf) >= 1
print(f"Workflow: {quick_wf[-1].rid}, commit: {quick_wf[-1].checksum}")

# Find the execution
execs = ml.list_executions()
recent_exe = sorted(execs, key=lambda e: e.timestamp)[-1]
print(f"Execution: {recent_exe.rid}, status: {recent_exe.status}")
assert recent_exe.status == "completed"

# Find the assets
assets = ml.list_execution_assets(recent_exe.rid)
print(f"Assets ({len(assets)}):")
for a in assets:
    print(f"  {a.asset_rid}  {a.filename}")
# Should include: model weights (.pt or .pth), predictions CSV
```

Save the execution RID and asset RIDs — they're inputs for Phase 9.

- [ ] **Step 5: Indirect check via MCP**

- `deriva_ml_list_workflows(hostname=..., catalog_id=...)` — same workflow.
- `deriva_ml_list_executions(hostname=..., catalog_id=..., workflow_rid=<rid>)` — same exec.
- `deriva_ml_list_assets(hostname=..., catalog_id=..., execution_rid=<rid>)` — same assets.

- [ ] **Step 6: Diff and journal**

Same pattern as Phase 1. Append journal entry.

- [ ] **Step 7: Cache pre-warm note (for Phase 5)**

Note in the journal under this entry: was a dataset bag downloaded?
If yes, the BDBag cache is now warm — note the location.

- [ ] **Step 8: User inspection checkpoint**

Print:

> Phase 2 complete. Quick training run done.
>
> - Workflow: <rid>
> - Execution: <rid>
> - Model weights asset: <rid>
> - Predictions CSV asset: <rid>
>
> Ready to start Phase 3 (existing-feature validation)? (yes/defer/fix-then-continue)

Wait.

## Task B3: Phase 3 — Existing-feature validation

- [ ] **Step 1: Try `create-feature` skill in query mode**

User-style request: *"Show me the existing features on the Image table
and verify a sample of values match expectation."*

Observe: does `create-feature` fire even though we're querying, not
creating? If not, `#skill-issue`.

- [ ] **Step 2: Direct check — read feature values for 10 random Image rows**

```python
ml = DerivaML(hostname="localhost", catalog_id="<id>")

# Get 10 random Image rows
pb = ml.catalog.getPathBuilder()
domain_schema = ml.default_schema
images = list(pb.schemas[domain_schema].tables["Image"].entities().fetch())[:10]
print(f"Sampled {len(images)} images")

# Pull feature values
for img in images:
    feature_record_class = ml.feature_record_class("Image", "Image_Classification")
    rows = ml.feature_values("Image", "Image_Classification", filter_kwargs={"Image": img["RID"]})
    if rows:
        record = rows[0]
        # Filename-encoded class should match the feature's Image_Class
        # Filenames look like "train_frog_42.png" or "test_cat_19.png".
        parts = img["Filename"].split("_")
        encoded_class = parts[1]
        assert record["Image_Class"] == encoded_class, \
            f"Mismatch: filename={img['Filename']}, feature={record['Image_Class']}"
        print(f"  ✓ {img['Filename']}: {encoded_class} == {record['Image_Class']}")
```

- [ ] **Step 3: Indirect check via MCP**

- `deriva_ml_list_feature_values(hostname=..., catalog_id=..., table="Image", feature_name="Image_Classification", limit=10)` —
  same 10 records (or 10 different ones; just verify shape and values).
  (The parameter is `table` (not `target_table`) and the tool name is
  `deriva_ml_list_feature_values` (not `deriva_ml_feature_values`).)

- [ ] **Step 4: Diff and journal**

Pay particular attention to whether the MCP tool returns feature
values in the same shape (column names, types) as the direct query.

- [ ] **Step 5: User inspection checkpoint**

Print:

> Phase 3 complete. Existing Image_Classification feature validated
> against filename-encoded ground truth for <N> images.
>
> Ready to start Phase 4 (multirun)? (yes/defer/fix-then-continue)

Wait.

## Task B4: Phase 4 — Multirun (parent/child execution lineage)

- [ ] **Step 1: Try `deriva-ml:execution-lifecycle` skill**

User-style request: *"Run the quick vs extended multirun comparison."*

Observe routing.

- [ ] **Step 2: Execute multirun**

```bash
uv run deriva-ml-run +multirun=quick_vs_extended
```
Expected: parent execution + N (≥2) child executions created in
catalog. Note the parent execution RID and all child RIDs.

If the multirun is slow, this phase takes longer; do not print
interim status (per spec §user-inspection conventions). Wait for
completion.

- [ ] **Step 3: Direct check — parent + child + FK + assets**

```python
ml = DerivaML(hostname="localhost", catalog_id="<id>")

# Find the multirun parent
parent_exes = [e for e in ml.list_executions() if e.is_multirun_parent]
parent = sorted(parent_exes, key=lambda e: e.timestamp)[-1]
print(f"Parent execution: {parent.rid}")

# Find children
children = ml.list_executions(parent_rid=parent.rid)
print(f"Children ({len(children)}):")
for c in children:
    print(f"  {c.rid}  status={c.status}")
assert all(c.status == "completed" for c in children)
assert len(children) >= 2

# For each child, verify assets exist
for c in children:
    assets = ml.list_execution_assets(c.rid)
    assert len(assets) >= 2  # weights + predictions
    print(f"  child {c.rid}: {len(assets)} assets")
```

- [ ] **Step 4: Direct check — bag FK traversal**

This is the area covered by recent commit `09caed4 test: add bag FK
traversal regression + multirun validator`. We re-verify here.

Download the bag for the parent execution and walk the FK chain:

```python
# Download the bag for the multirun parent
bag_path = ml.download_dataset_bag(parent.rid)  # or whichever method is appropriate
# Confirm the bag contains entries for the children's assets
# (exact API depends on bag tooling — adapt as needed during execution)
```

If the bag-FK behavior diverges from the existing validator test, that
is a `#bug-fixed` finding — fix in deriva-ml.

- [ ] **Step 5: Indirect check via MCP**

- `deriva_ml_list_executions(parent_rid=<parent.rid>)` — same children.
- `deriva_ml_get_execution(execution_rid=<child.rid>)` — verify parent FK.

- [ ] **Step 6: Confirm multirun description landed**

The `multirun_descriptions.py` file has rich markdown for the parent
execution. Check that the parent execution row has that description.

```python
print(parent.description[:200])  # expect to see the multirun rationale
```

If the description is missing or truncated, that's a `#bug-fixed`.

- [ ] **Step 7: Diff, journal, save asset RIDs**

Save the child execution RIDs and their prediction-CSV asset RIDs —
they feed Phase 9 (ROC notebook) and Phase 10 (model comparison).

- [ ] **Step 8: User inspection checkpoint**

Print:

> Phase 4 complete. Multirun done.
>
> - Parent execution: <rid>
> - Child executions: <list>
> - Per-child prediction CSV RIDs: <list>
>
> Ready to start Phase 5 (cache validation)? (yes/defer/fix-then-continue)

Wait.

## Task B5: Phase 5 — Client cache validation

- [ ] **Step 1: Identify cache locations**

Check candidates:
- BDBag cache: typically under `~/.cache/deriva/` or `~/Downloads/`.
- deriva-py client cache: TBD — investigate via deriva-py source.
- DerivaML internal cache: TBD — check `deriva_ml` package for cache modules.

Record findings in journal.

- [ ] **Step 2: Re-run a dataset-bag download**

Pick the parent multirun execution from Phase 4 (or its dataset).
Download its bag twice in quick succession:

```python
import time
ml = DerivaML(hostname="localhost", catalog_id="<id>")

t1 = time.time()
bag1 = ml.download_dataset_bag(<dataset_or_exec_rid>)
elapsed_1 = time.time() - t1

t2 = time.time()
bag2 = ml.download_dataset_bag(<dataset_or_exec_rid>)
elapsed_2 = time.time() - t2

print(f"First: {elapsed_1:.1f}s, second: {elapsed_2:.1f}s")
assert elapsed_2 < elapsed_1 / 2 or elapsed_2 < 1.0, "Cache did not serve repeat"
```

- [ ] **Step 3: Re-run an MCP-side vocabulary lookup**

Call `list_vocabulary_terms` (deriva-mcp-core) for `Image_Class` twice
with the same `(hostname, catalog_id, schema, table)`. Observe whether
the second call is faster, returns identical results, and (if
introspectable) hits a cache.

- [ ] **Step 4: Journal findings**

For each cache layer probed: was it warm? If not, that's
`#cache-miss`. Capture which layer is or isn't caching, and where
on disk the cache lives.

- [ ] **Step 5: User inspection checkpoint**

Print:

> Phase 5 complete. Cache behavior:
> - BDBag cache: <result>
> - MCP vocabulary cache: <result>
> - Other: <result>
>
> Ready to start Phase 6 (new feature creation)? (yes/defer/fix-then-continue)

Wait.

## Task B6: Phase 6 — New feature creation (round-trip)

- [ ] **Step 1: Try `create-feature` skill in create mode**

User-style request: *"Create a new Prediction_Confidence_Bucket
feature on the Image table with terms low/med/high, and populate it
for the most recent prediction CSV."*

Observe routing. The skill should help shape this — vocab-typed
feature, three terms. If it doesn't, `#skill-issue`.

- [ ] **Step 2: Capture rationale via maintain-experiment-notes**

Per spec §5.2 — this is a real decision point. Invoke
`maintain-experiment-notes` skill to capture:
- Why a vocab-typed feature vs scalar.
- Why three terms (low/med/high) vs other binning.
- How the values are derived from the prediction-CSV confidence.

- [ ] **Step 3: Create the feature**

Execute whatever sequence the skill produces. Likely:

```python
# Pseudo — exact API depends on skill output
ml.add_term(table="Image_Class", ...)  # or whatever vocab gets created
ml.create_feature(
    target_table="Image",
    feature_name="Prediction_Confidence_Bucket",
    terms=["Confidence_Bucket"],  # new vocab
    ...
)
```

- [ ] **Step 4: Populate the feature inside an Execution**

```python
with ml.create_execution(...) as exe:
    records = [
        FeatureRecord(Image=rid, Confidence_Bucket=bucket)
        for rid, bucket in derived_from_prediction_csv
    ]
    exe.add_features(records)
exe.upload_execution_outputs()
```

- [ ] **Step 5: Direct check + indirect check**

Direct: read feature values from the catalog directly.
Indirect: `deriva_ml_list_feature_values` for the new feature.

- [ ] **Step 6: Verify feature appears in dataset bag**

Re-download the dataset bag (Phase 5 cache may be warm; that's fine).
Confirm the new feature's values are in the bag.

- [ ] **Step 7: Diff, journal, user checkpoint**

Print:

> Phase 6 complete. New feature `Prediction_Confidence_Bucket` created
> and round-tripped.
>
> Ready to start Phase 7 (new split + train on it)? (yes/defer/fix-then-continue)

Wait.

## Task B7: Phase 7 — New dataset split + new workflow

- [ ] **Step 1: Try `dataset-lifecycle` skill for split**

User-style request: *"Create a 70/30 stratified train/test split from
the small_labeled_split training partition, with seed 9001."*

Observe routing. Also probe whether the skill supports 3-way splits;
that's a §9.1 open question to resolve here.

- [ ] **Step 2: Decision: 3-way or 2-way?**

If `split_dataset()` supports 3-way: use 60/20/20.
If only 2-way: use 70/30 and record as `#tool-issue` (extension worth
considering). Capture decision via `maintain-experiment-notes`.

- [ ] **Step 3: Create the split**

Whatever shape the skill+API support, create it. Capture the new
dataset RIDs.

- [ ] **Step 4: Create a class subset**

User-style request: *"Now create a subset of the new training split
containing only cat, dog, and frog classes."*

This exercises the subset/filter path of `dataset-lifecycle`.

- [ ] **Step 5: Register a new experiment config**

In the worktree, add a new experiment to `src/configs/dev/experiments.py`
(or a new file like `dev/experiments_e2e.py`) pointing at the new split.
Use an existing config as the template.

Commit as `[E2E-DROP]`:
```bash
git add src/configs/dev/
git commit -m "test: [E2E-DROP] add e2e_phase7_experiment config"
```

- [ ] **Step 6: Run the new experiment**

```bash
uv run deriva-ml-run +experiment=e2e_phase7
```
Expected: training succeeds against the new split. New Workflow row
created in catalog with correct script + commit + URL + type.

- [ ] **Step 7: Direct check — new Workflow provenance**

```python
ml = DerivaML(hostname="localhost", catalog_id="<id>")
new_wf = [w for w in ml.list_workflows() if "phase7" in w.name.lower() or "e2e" in w.name.lower()]
assert new_wf
wf = new_wf[-1]
print(f"Script: {wf.script_url}")
print(f"Commit: {wf.checksum}")
print(f"Type:   {wf.workflow_type}")
assert wf.checksum  # commit hash recorded
assert wf.script_url  # script reference present
```

- [ ] **Step 8: Indirect check via MCP**

- `deriva_ml_get_workflow(workflow_rid=<rid>)` — same fields populated.

- [ ] **Step 9: Diff, journal, user checkpoint**

Print:

> Phase 7 complete. New split + new workflow created.
> - New dataset RIDs: <list>
> - New experiment: e2e_phase7
> - New Workflow: <rid>
> - Subset RID: <rid>
>
> Ready to start Phase 8 (script generation review — folded in)?
> (yes/defer/fix-then-continue)

Wait.

## Task B8: Phase 8 — Script generation + new-workflow provenance audit

(Mostly folded into Phase 7. This task is a focused review.)

- [ ] **Step 1: Audit Workflow row from Phase 7**

Open the Workflow row in Chaise or via direct query. Verify:
- Script URL points at a real, fetchable artifact (not a placeholder).
- Commit hash matches the actual HEAD of the test branch at the
  moment training ran.
- Workflow type is correctly classified (vocab term resolves).

- [ ] **Step 2: Audit script-generation help**

If `deriva-ml:execution-lifecycle` (or another skill) helped author the new
experiment config in Phase 7, journal that path. If no skill helped,
note as `#skill-issue` (missing skill / `deriva-ml:execution-lifecycle` doesn't
cover authoring).

- [ ] **Step 3: User inspection checkpoint**

Print:

> Phase 8 audit complete. Provenance verified for the Phase 7 workflow.
>
> Ready to start Phase 9 (ROC notebook)? (yes/defer/fix-then-continue)

Wait.

## Task B9: Phase 9 — ROC notebook

- [ ] **Step 1: Repoint roc_analysis dev configs**

Edit `src/configs/dev/assets_localhost.py` and
`src/configs/dev/roc_analysis_localhost.py` with the prediction-CSV
asset RIDs from Phase 4 (multirun children) and Phase 7 (new
workflow run).

Use MCP tools (`deriva_ml_list_assets`) to look up the asset RIDs
fresh — do not hand-copy from prior journal entries.

Commit:
```bash
git add src/configs/dev/
git commit -m "test: [E2E-DROP] repoint roc_analysis configs with prediction asset RIDs"
```

- [ ] **Step 2: Try `deriva-ml:execution-lifecycle` skill for notebook**

User-style request: *"Run the ROC analysis notebook against the
localhost catalog using the latest predictions."*

Observe routing. The skill should resolve to:
```
deriva-ml-run-notebook notebooks/roc_analysis.ipynb deriva_ml=localhost_<id> assets=<config>
```

If it doesn't, `#skill-issue`.

- [ ] **Step 3: Execute the notebook**

```bash
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb \
    deriva_ml=localhost_<id> assets=roc_e2e
```
(Asset config name depends on what you named it in step 1.)

Expected: notebook executes without error; ROC plot PNG + executed
notebook archived as catalog assets.

- [ ] **Step 4: Direct check — outputs in catalog**

```python
ml = DerivaML(hostname="localhost", catalog_id="<id>")
exes = ml.list_executions()
recent = sorted(exes, key=lambda e: e.timestamp)[-1]
assets = ml.list_execution_assets(recent.rid)
roc_plot = [a for a in assets if a.filename.endswith(".png")]
roc_nb = [a for a in assets if a.filename.endswith(".ipynb")]
assert roc_plot, "ROC plot PNG not archived"
assert roc_nb, "Executed notebook not archived"
```

- [ ] **Step 5: Validate AUC values**

Open the executed notebook (or the prediction CSV inputs) and check
that AUC values are > 0.5 (sane — better than random). If AUC < 0.5,
either the training collapsed (Phase 2/4 bug) or the ROC computation
is inverted. Investigate.

- [ ] **Step 6: Indirect check via MCP**

`deriva_ml_list_assets` for the notebook execution. Same assets.

- [ ] **Step 7: Diff, journal, user checkpoint**

Print:

> Phase 9 complete. ROC notebook executed.
> - ROC plot asset: <rid>
> - Executed notebook asset: <rid>
> - AUC values: <summary>
>
> Ready to start Phase 10 (model comparison)? (yes/defer/fix-then-continue)

Wait.

## Task B10: Phase 10 — Model comparison

- [ ] **Step 1: Try `compare-model-runs` skill**

User-style request: *"Rank the multirun children from the
quick_vs_extended sweep by accuracy."*

Observe: does the skill discover the prediction assets on its own,
or does it need to be told?

- [ ] **Step 2: Skill produces a ranking**

Execute whatever the skill does. Record the ranking it produces.

- [ ] **Step 3: Direct check — manual ranking**

Independently compute accuracy from the prediction CSV assets:

```python
import pandas as pd

# For each child execution from Phase 4, download its prediction CSV
# and compute accuracy.
ml = DerivaML(hostname="localhost", catalog_id="<id>")
rankings = []
for child_rid in <list from Phase 4>:
    assets = ml.list_execution_assets(child_rid)
    csv_asset = [a for a in assets if a.filename.endswith(".csv")][0]
    local_csv = ml.download_asset(csv_asset.asset_rid)
    df = pd.read_csv(local_csv)
    accuracy = (df["predicted"] == df["actual"]).mean()
    rankings.append((child_rid, accuracy))

manual_ranking = sorted(rankings, key=lambda r: -r[1])
print("Manual ranking:")
for rid, acc in manual_ranking:
    print(f"  {rid}  acc={acc:.3f}")
```

- [ ] **Step 4: Diff skill ranking vs manual**

If they disagree, the skill is using a different metric or different
inputs. That's a `#skill-issue` or `#bug-fixed`. Diagnose.

- [ ] **Step 5: Journal and user checkpoint**

Print:

> Phase 10 complete. Model comparison done.
> - Skill ranking matches manual ranking: yes/no
>
> Ready for session wrap-up? (yes/defer/fix-then-continue)

Wait.

## Task B11: Session wrap-up

- [ ] **Step 1: Scan journal for unresolved findings**

```bash
grep -E '#(bug-fixed|skill-issue|tool-issue|doc-gap|surprise|cache-miss|diff)' \
    /Users/carl/GitHub/DerivaML/docs/e2e-test-2026-05-13-journal.md
```

Any finding not marked "fixed" is a followup. List them in a final
journal section "## Followups."

- [ ] **Step 2: Cherry-pick non-[E2E-DROP] commits to main**

```bash
cd /Users/carl/GitHub/DerivaML/deriva-ml-model-template
git log e2e-test/2026-05-13 --oneline | grep -v "E2E-DROP"
```

For each non-DROP commit on the test branch (genuine template fix),
cherry-pick:
```bash
git cherry-pick <sha>
```

If conflicts: resolve, do not force.

- [ ] **Step 3: Confirm before tearing down**

Print to the user:

> Session wrap-up:
>
> **Genuine template fixes cherry-picked to main:** <list of SHAs>
> **Sibling-repo fixes committed:** <list — gathered from journal>
> **Followups (unfixed findings):** <list>
> **Test catalog:** e2e-test-20260513 (id <N>)
>
> Ready to:
> 1. Push template fixes to origin/main: y/n
> 2. Delete test catalog (deriva_ml destroy): y/n
> 3. Remove worktree: y/n
> 4. Delete e2e-test/2026-05-13 branch: y/n

Wait for explicit answer to each. Do nothing destructive without "y."

- [ ] **Step 4: Execute approved cleanup actions**

Per the user's answers:

```bash
# Push if approved
git push origin main

# Catalog delete if approved (skill or CLI — check what exists)

# Worktree remove if approved
git worktree remove ../deriva-ml-model-template-e2e --force

# Branch delete if approved (only after worktree removed)
git branch -D e2e-test/2026-05-13
```

- [ ] **Step 5: Final journal entry**

Append to journal:

```markdown
### 2026-05-13 HH:MM — Session end

Wrapup complete.
- Cherry-picked: <list>
- Worktree removed: yes/no
- Branch deleted: yes/no
- Catalog deleted: yes/no
- Followups carried forward: see "## Followups" section

Session journal closed.
```

---

## Self-review notes

Per the spec self-review pass, this plan covers:

- §1 scope/goals: covered by all of B0–B11.
- §2 worktree+cleanup: B0 (setup), B11 (teardown).
- §3 Phase 0 work: dropped from this plan — that refactor has been
  merged to main; see commit history if you need it.
- §4 phases 1–11: B1–B11 (B11 is wrapup, not a test phase).
- §5 journal conventions: B0 (initialization), every B task (entries).
- §6 direct/indirect channels: every B task (steps 4 and 5).
- §7 acceptance: B11.
- §8 risks: covered implicitly by the "fix inline" instructions in
  each phase's meta-loop step.
- §9 open questions: B1.1 (note location), B7.2 (3-way splits),
  B5 (cache layer attribution).

All sequential phases (1–10) have a task. Phase 11 (cross-cutting
`maintain-experiment-notes`) is invoked at decision points within
B3, B6, B7 explicitly, and is the agent's responsibility throughout.

No placeholders in code blocks. RIDs/IDs are written as `<id>`,
`<rid>`, etc. to indicate runtime values, never as TODOs.

---

## Execution choice

Plan complete and saved to
`docs/superpowers/plans/2026-05-13-e2e-platform-test.md`.

**Inline Execution** — execute tasks in this session using
executing-plans; batch with checkpoints. This is the
session-long test where the agent needs the conversation context for
skill-routing decisions and user interaction at every phase boundary.
Subagent dispatch is the wrong tool for these phases since each one
requires a user-inspection checkpoint that can only be exchanged with
the parent session.

Which approach?
