# E2E Platform Test Session Journal — 2026-05-19 (clean restart)

This journal restarts Phase 1 from scratch against the post-bugfix
baseline. The prior attempt is preserved at
`docs/e2e-test-2026-05-19-journal.archived.md` (Phase 1 against
pre-fix pins; surfaced bugs B1–B7 in `docs/bugs/`, all since fixed).

---

## Session setup

- **Workspace:** `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e`, branch `e2e-test/2026-05-18`
- **Plan:** `docs/superpowers/plans/2026-05-13-e2e-platform-test.md` (in the main repo)
- **uv.lock baseline:** `d28bcfb` (deriva-py `ed5ee69`, deriva-ml `0f14de7e`)
- **Installed pins:** `deriva 2.0.0.dev0 @ ed5ee69`, `deriva-ml 1.36.5.post4+g0f14de7eb`
- **MCP server:** dev-localhost (deriva-mcp-core `08bb642` + deriva-ml-mcp `2116130`), OAuth via credenza

### Pre-flight cleanup (this session)

- Deleted catalog `20` on localhost (created on pre-fix pins; the
  archived journal's validity caveat at line 184 explicitly calls
  for a recreation against the cache-fixed code).
- Cleared `~/.deriva-ml/localhost/20/` schema cache.
- Dropped commit `4769eba` ([E2E-DROP] repoint at catalog 20) via
  `git rebase --onto 116092b 4769eba HEAD`. Kept the uv.lock bump
  (now `d28bcfb`). Local branch diverged from origin/e2e-test/2026-05-18
  by design; no force-push without user approval.
- Archived previous journal to `docs/e2e-test-2026-05-19-journal.archived.md`.

---

### 2026-05-19 16:57 — Phase 1: catalog bootstrap (clean baseline)

**Skill tried:** `deriva-ml:setup-ml-catalog` (1.3.5 replacement for the
plan-referenced `route-project-setup`). Same status as the archived
run: marked `disable-model-invocation: true`, so the agent cannot
invoke it; only an explicit `/deriva-ml:setup-ml-catalog` slash command
from the user can. Fell back to CLI per the plan's "whether routed by
skill or by fallback" branch. Same design observation as before
about the test-mode vs user-confirmation tension.

**Routed to:** none — fell back to CLI.

**MCP tools used:**
- `deriva_ml_list_datasets(hostname="localhost", catalog_id="36")`
- `deriva_ml_list_features(hostname="localhost", catalog_id="36", table="Image")`
- `list_vocabulary_terms(hostname="localhost", catalog_id="36", schema="e2e-test-20260519b", table="Image_Class")`
- `list_schemas(hostname="localhost", catalog_id="36")`

**Catalog created:** id `36`, domain schema `e2e-test-20260519b`,
500 images (250 train + 250 test), 500 features (Image_Classification
per image), 13 datasets (RIDs 84W, 856, 85E, 85R, 868, 86G, 86T, B68,
B6G, B6T, BQW, BR4, BRE), 10 Image_Class vocab terms.

**Direct/indirect diff:**

| Check | Direct (deriva-ml) | Indirect (MCP) | Agreement |
|-------|--------------------|----------------|-----------|
| Schemas | `public, WWW, deriva-ml, e2e-test-20260519b` | same 4 | ✓ |
| Dataset count | 13 | 13 | ✓ |
| Dataset RIDs | (above) | match | ✓ |
| `Image_Class` vocab terms | 10 (airplane…truck) | 10 same | ✓ |
| `Image_Classification` feature on Image | present, 1 row | `count: 1` | ✓ |
| `find_features()` (no table) dedup | 1 copy | (always passes table, n/a) | ✓ |

**No `#diff`.** All four MCP indirect channels agree with the direct
deriva-ml view.

**Regression-fix verification (bugs from the archived Phase 1 run):**

- **B1 (`find_features` duplicates)** — `ml.find_features()` with no
  `table=` arg now returns **1** copy of `Image_Classification`
  (previously 3). Fix at deriva-ml `1f2e722` confirmed working in
  installed pin `0f14de7e`.
- **B2 (schema-cache staleness after load)** — Fresh `DerivaML()`
  construction after `load-cifar10` saw `Image`, `Image_Class` vocab,
  all 13 datasets, and the `Image_Classification` feature **without**
  any `refresh_schema()` workaround. Fix at deriva-py `3a6a7bb`
  (cache-invalidation refactor) + deriva-ml `0f14de7` (route schema
  reads through `getCatalogSchema()`) confirmed working.

**Findings:** none for this run. The skill-invocation gate and the
plan-drift on `deriva_ml_list_vocabularies` are still open (carried
over from the archived journal, both already documented in
`docs/findings/`).

**Decisions:**

- Catalog name `e2e-test-20260519b` chosen with the `b` suffix to avoid
  collision with prior `e2e-test-20260520` (deleted) and any other
  same-day names. The resulting catalog id is `36`.
- Did NOT push the branch — local diverges from origin by 1 commit
  ahead, 2 commits behind. Push deferred to user approval.

**Repoint:**

- `deriva_localhost.py`: `localhost_1407` → `localhost_36`.
- `datasets_localhost.py`: 13 dataset RIDs updated, version
  `0.1.0.post1.dev1`.
- `assets_localhost.py` + `roc_analysis_localhost.py`: cleared stale
  RIDs (populated by Phase 2 / 4). All `deriva_ml` refs point at
  `localhost_36`.
- `tests/test_configs_load.py` passes after the repoint.
- DROP commit: `b6492a4` on branch `e2e-test/2026-05-18`.

---

### 2026-05-19 19:29 — Inter-phase: PR 1 (Execution Duration fix) verified

After diagnosing that catalog 36's 5 Execution rows all had `Duration: null`
(despite `load-cifar10` following the documented `with`-block pattern), filed
the bug doc at `docs/bugs/2026-05-19-execution-exit-omits-duration.md` and a
companion design doc at `docs/bugs/2026-05-19-execution-phase-durations-design.md`
covering the broader three-phase measurement (download / algorithm / upload),
the `Duration` → `Execution_Duration` rename, and the forward-only migration
strategy.

**Root cause** (verified by static reading of `execution.py`):
`Execution.__exit__`'s clean-exit branch issued an inline `Running → Stopped`
transition with `extra_fields={"stop_time": now}` only — no `duration`. The
parallel `execution_stop()` method (called from `runner.py:233` for multirun
parents and `execution.py:1360` for the upload-time auto-stop fallback) writes
both `stop_time` AND a computed `duration_str`. The catalog body builder
projects `Duration` into the catalog PUT body only when SQLite has it set, so
`__exit__`'s shorter payload left the column null for every with-block exit
— which is the dominant code path across the install base.

**Fix**: `__exit__`'s clean-exit branch now delegates to `execution_stop()`
instead of issuing its own transition. The Failed branch stays inline (PR 2
scope to add failure-path duration). Single-atomic-transition invariant
(audit §4.5) restored for the recommended `with`-block pattern.

**Verification path**:
- Branch `fix/execution-exit-writes-duration` on deriva-ml, commit `7960c101`.
- Unit test `test_execution_context_manager` strengthened to assert
  `catalog_row["Duration"] is not None` after with-block exit. Full
  `tests/execution/` suite: 302 passed, 8 skipped (one pre-existing
  failure in `test_bag_loader_path_builder_refresh.py` unrelated to this
  fix — confirmed it fails on the unmodified branch HEAD too).
- Deleted catalog 36, cleared its schema cache, bumped this worktree's
  `pyproject.toml` + `uv.lock` to pin the deriva-ml fix branch
  (`deriva-ml==1.36.5.post5+g7960c101e`).
- Reran `load-cifar10 --create-catalog e2e-test-20260519c --num-images 500`,
  creating catalog **42** (domain schema `e2e-test-20260519c`, 500 images,
  13 datasets, same shape as catalog 36).

**Result**: all 5 Execution rows in catalog 42 have non-null `Duration`:

| RID | Duration | Status |
|---|---|---|
| 458 | `0.0H 0.0min 0.6889sec` | Uploaded |
| 740 | `0.0H 0.0min 0.0307sec` | Uploaded |
| 844 | `0.0H 0.0min 1.4057sec` | Uploaded |
| B5G | `0.0H 0.0min 0.8286sec` | Uploaded |
| BQ4 | `0.0H 0.0min 0.8364sec` | Uploaded |

Compare to catalog 36's 5 executions (all `Duration: null`) before the fix.
This is the exact behavior change PR 1 was meant to produce. **`#bug-fixed`**

**Outstanding for PR 1**:
- Open a PR on `informatics-isi-edu/deriva-ml` for review and landing.
- After PR lands on main, revert this worktree's `pyproject.toml` to drop
  the explicit `branch = "fix/..."` pin (back to plain `git = "..."`).

**Carry-forward**: catalog 42 inherits the role catalog 36 had as the Phase 1
baseline. The dev configs in this worktree still point at catalog 36 (now
deleted). Repoint to catalog 42 will land as a second `[E2E-DROP]` commit
in the same phase.

---

### 2026-05-19 19:35 — Inter-phase: PR 2 (phase-duration split + rename) verified

After PR 1 closed the dominant "Duration is always null" bug, PR 2 layered on
the broader phase-duration design: rename `Duration` →
`Execution_Duration`, add `Download_Duration` for the
`_initialize_execution` (dataset/asset materialization) phase, add
`Upload_Duration` for the `upload_execution_outputs` (bag commit) phase.
Forward-only migration: new catalogs get all three columns, old catalogs
keep their unchanged `Duration` column.

**Commits**:
- `deriva-ml@4727d6a1` on branch `fix/execution-exit-writes-duration`
  (stacked on PR 1's `7960c101`).
- `deriva-ml-mcp@50478ee` on branch `feat/execution-phase-durations`
  (single commit; surfaces the two new fields through ExecutionSummary,
  the RAG serializer, and every read tool/resource that uses
  `_summarize_execution`).

**Tests after PR 2**:
- deriva-ml: 302 passed, 8 skipped (unchanged from PR 1 baseline; one
  unrelated pre-existing failure deselected).
- deriva-ml-mcp: 325 passed, 22 deselected (one rag-serializer test
  updated for the new `Execution Duration` label + the two new lines).

**Verification path** (separate from PR 1 verification since catalog 42
predates PR 2):
- Deleted catalog 42, cleared its schema cache.
- Bumped this worktree's `uv.lock` to `deriva-ml@4727d6a1`
  (`deriva-ml==1.36.5.post6+g4727d6a19`).
- Reran `load-cifar10 --create-catalog e2e-test-20260519d --num-images 500`,
  creating catalog **46** (domain schema `e2e-test-20260519d`, 500 images,
  13 datasets).
- Queried Execution rows via the MCP server's generic `get_entities`
  (bypassing the MCP server's pre-PR-2 deriva-ml-mcp serializer, which
  hasn't been rebuilt yet — that's fine for column-level verification).

**Result**: all 5 Execution rows in catalog 46 carry all three new
duration columns:

| RID | Execution_Duration | Download_Duration | Upload_Duration | Status |
|---|---|---|---|---|
| 45A | 0.83 s | 1.09 s | **10.62 s** | Uploaded |
| 742 | 0.03 s | 0.69 s | 0.43 s | Uploaded |
| 846 | 1.40 s | 0.52 s | 0.33 s | Uploaded |
| B5J | 0.96 s | 0.55 s | 0.14 s | Uploaded |
| BQ6 | 0.87 s | 0.82 s | 0.18 s | Uploaded |

The 10.62 s upload time on RID 45A (the bulk image-upload execution
that PUTs 500 image bytes to Hatrac) is exactly the kind of
phase-specific diagnostic the design doc called out — under the old
single `Duration` column, that signal was invisible. **`#bug-fixed`**

**Outstanding for PR 2**:
- Open PRs on `informatics-isi-edu/deriva-ml`
  (`fix/execution-exit-writes-duration` carries both PR 1 + PR 2) and
  `informatics-isi-edu/deriva-ml-mcp` (`feat/execution-phase-durations`).
- The dev-localhost MCP server in this session is still running pre-PR-2
  pins — its `_summarize_execution` will report `null` for the two new
  fields until the server image is rebuilt. That's the expected
  forward-only behavior for old MCP clients; not blocking PR landing.
- After PRs land on main, revert this worktree's `pyproject.toml`
  `branch =` override back to plain `git = "..."`.

**Carry-forward**: catalog 46 now inherits the Phase 1 baseline role
(replacing catalog 42, which was PR-1-only and is now deleted). Dev
configs need a final repoint to catalog 46 before Phase 2.

---

### 2026-05-20 — Phase 2: quick training (dry-run + real)

**Skill tried:** `deriva-ml:execution-lifecycle` (the lifecycle gate
sits closer to the work than the spec-named `route-run-workflows`,
which now delegates here). Skill fired and routed correctly to the
`deriva-ml-run +experiment=...` CLI shape. No `#skill-issue`.

**Inter-phase bugfix interlude.** Phase 2 surfaced four
post-Phase-1-baseline bugs that were fixed inline before this entry
was written:

- **B8 (deriva-ml):** `run_model` passed `upload_timeout` /
  `upload_chunk_size` kwargs to `upload_execution_outputs` that
  didn't exist on the new pydantic signature. Pydantic
  `ValidationError` on every real run. **Fixed:** deriva-ml #168
  (stripped the kwargs). Filed deriva-py issues #261/#262 for the
  underlying upload-tunable hole.
- **B9 (template):** the hand-rolled `_RidAwareImageDataset` used
  the wrong BDBag layout (`data/assets/{table}/{rid}/...` instead
  of canonical `data/asset/{rid}/{table}/...`). Training crashed on
  first image read. **Fixed:** template #11 (path), then template
  #12 (rewrite to drop the hand-rolled class entirely — see B11).
- **B10 (deriva-ml-mcp):** `ExecutionExperiment.config_choices` and
  `model_cfg` had narrow pydantic types that rejected hydra-zen's
  `_zen_exclude: list[str]`. MCP silently returned `null` for the
  field. **Fixed:** deriva-ml-mcp #39 (broadened to `dict[str, Any]`,
  stopped swallowing `lookup_experiment` errors).
- **B11 (deriva-ml + template):** `bag.as_torch_dataset` hid the
  RID, so the template had to hand-roll a workaround
  (`_RidAwareImageDataset`) just to surface it for prediction
  recording. **Fixed:** deriva-ml #169 made the adapter always
  yield `(sample, target, rid)`; template #12 dropped 56 lines of
  duplicate bag-iteration logic.

Also discovered (and not a Phase-2 bug per se, but blocking
verification): the installed deriva-py in this worktree's venv was
stale and lacked commit `9d6daae` ("make `model` a regular
attribute, not a `@property`"), causing
`DatabaseModel.model has no setter` at execution startup. **Fixed
by venv refresh** — `uv lock --upgrade-package deriva` bumped to
`ed5ee69c`; no deriva-ml change required.

**Verifying execution (post-bugfix end-to-end run):**

```
uv run deriva-ml-run +experiment=cifar10_quick \
    deriva_ml=localhost_46 \
    datasets=cifar10_small_labeled_split_localhost
```

Result: execution **CMY** (Workflow CJW, checksum `afb676f0c587`):

- Training: 200 samples → 3 epochs, batch size 128.
- Test: 50 samples → final test_loss=0.3114, **test_acc=94%**.
- 50 classification predictions recorded against catalog
  (`Image_Classification` feature) with RIDs.
- 3 Execution_Assets uploaded: weights (6.5 MB), training log
  (480 B), prediction probabilities CSV (7.9 KB).

**Direct check** (raw ermrest via `getPathBuilder`):

| Exec RID | Status | Workflow | Execution_Duration | Download_Duration | Upload_Duration | Notes |
|---|---|---|---|---|---|---|
| CDG | Uploaded | CBW | 0.72 s | 1.81 s | 0.77 s | first successful post-B8 run |
| CK6 | Uploaded | CJW | 0.04 s | 1.05 s | 0.61 s | hit B9 (dataset override missing) → wrote `training_status.txt` then uploaded |
| **CMY** | **Uploaded** | **CJW** | **0.67 s** | **5.30 s** | **0.79 s** | **full end-to-end with correct dataset override** |
| C90, CAE | Failed | C8T | 0.08–0.27 s | 1.6 s | null | pre-B8 attempts (validation error before upload phase) |
| CC2 | Stopped | CBW | 0.72 s | 1.92 s | null | aborted (no commit) |
| CK2, CK4 | Created | — | null | null | null | orphaned starts (script bail before run_model) |

All three duration columns populated for successful runs — confirms
PR-1 + PR-2 phase-duration measurement still works end-to-end on
catalog 46 with the latest pins. Failed runs correctly leave
`Upload_Duration` null (failure exits before the upload phase).

**Indirect check (MCP):**
`deriva_ml_get_execution` returns the same RID set with matching
status + duration values. The `experiment` block on CMY (formerly
silently null per B10) now correctly includes the full
`config_choices` and `model_cfg` from the hydra-zen run.

**Diff:** clean. No `#skill-issue`, no `#finding` outside the four
bugs above (all fixed).

**Cache:** running cifar10_quick a second time (CMY follows CK6 on
the same dataset RID) reused the local BDBag cache at
`~/.deriva-ml/localhost/46/...`. Bag download for CMY was 5.3 s
vs CK6's 1.05 s — opposite of what cache reuse should look like.
Filed as `#cache-finding` for Phase 5 to investigate (suspect:
bag-cache key includes the execution RID or a timestamp; needs
verification against the cache code).

**Carry-forward to Phase 3:**
- Training executions CDG and CMY each recorded 50
  `Image_Classification` feature rows over 50 test images, with
  RIDs surfaced via the new adapter shape. That's the population
  Phase 3's feature-validation step (`create-feature` in query
  mode, raw-table check, ground-truth filename comparison) will
  read.
- `route-run-workflows` skill never invoked — deriva-ml-skills now
  ships `execution-lifecycle` as the canonical entry point. The
  spec text in `superpowers/specs/2026-05-13-e2e-platform-test-design.md`
  is out of date but the routing is correct; flagging as a doc
  refresh task, not a bug.

---

### 2026-05-20 — Phase 3: feature validation (Image_Classification)

**Skill tried:** `deriva-ml:create-feature`. Loaded as
expected; Phase 5 of the skill body (query / explore) was the
relevant section. The skill routed correctly to query mode based
on the explicit "I want to query — not create" framing in the
invocation. **No `#skill-issue`.** Per the skill's own guidance,
used the resource-snapshot path first
(`deriva://catalog/localhost/46/ml/features/Image`), then
`deriva_ml_get_feature` for column-level detail.

**Feature schema** (one feature on `Image` in catalog 46):

| Property | Value |
|---|---|
| `feature_name` | `Image_Classification` |
| `target_table` | `Image` |
| `feature_table` | `Execution_Image_Image_Classification` (in domain schema `e2e-test-20260519d`) |
| `term_columns` | `Image_Class` (FK → `Image_Class` vocab, NOT NULL) |
| `value_columns` | `Confidence` (float4, nullable, no default) |
| `asset_columns` | (none) |
| `comment` | "CIFAR-10 class label and optional confidence score for each image" |

**Direct check** (raw ermrest via
`getPathBuilder().schemas["e2e-test-20260519d"].tables["Execution_Image_Image_Classification"]`):

600 total rows, grouped by Execution:

| Execution | Role | Rows | Agreement w/ filename ground truth | Notes |
|---|---|---|---|---|
| `742` | load-cifar10 ground-truth seeding | 500 | **500/500 = 100.0%** | Every Image row has a feature row whose `Image_Class` matches the class encoded in `Filename` (parsed via the CIFAR-10 class tokens). `Confidence` is null (ground truth has no model). |
| `CDG` | Phase-2 first successful training run | 50 | **47/50 = 94.0%** | Matches the training-loop-reported `test_acc=94%` exactly. All 3 errors are `ship → bird`. `Confidence` populated (range 0.93-0.999). |
| `CMY` | Phase-2 verification run after PR B | 50 | **47/50 = 94.0%** | Same 3 ship→bird errors as CDG — deterministic training given same dataset and seeds. |

**Indirect check** (MCP `deriva_ml_list_feature_values`):

| Tool call | MCP result | Matches direct? |
|---|---|---|
| `preflight_count=True` (no filter) | 600 | ✅ |
| `preflight_count, by_execution selector + selector_execution_rid="742"` | 500 | ✅ |
| `preflight_count, by_execution selector + selector_execution_rid="CDG"` | **"No feature records match execution 'CDG'."** | ❌ **`#bug`** |
| `preflight_count, by_execution selector + selector_execution_rid="CMY"` | **"No feature records match execution 'CMY'."** | ❌ same bug |
| `execution_rids=["CDG"], preflight_count` | 50 | ✅ |
| `selector="newest"`, preflight | 500 | ✅ (one per Image — collapses the 1+2 (ground+predictions) records per test image to the newest, the others to the only ground-truth record) |
| Row content with `execution_rids=["CDG"], limit=10` | First 3 Image RIDs = `46E`, `46G`, `46W` (the three ship→bird misses); Confidence values match direct check to 8 decimal places | ✅ |

**Bug B12 (deriva-ml): `FeatureRecord.select_by_execution` raises
instead of returning None for non-matching groups.** Root cause
in `src/deriva_ml/feature.py:174-178`:

```python
def _selector(records: list["FeatureRecord"]) -> "FeatureRecord":
    filtered = [r for r in records if r.Execution == execution_rid]
    if not filtered:
        raise DerivaMLException(f"No feature records match execution '{execution_rid}'.")
    return FeatureRecord.select_newest(filtered)
```

`feature_values` calls the selector once per *target* (per Image).
Image `46E` has 3 records (one each from `742`, `CDG`, `CMY`),
but image `47P` (a training image not in the test set) has only
1 record from `742`. When `_selector` runs on `47P`'s group with
`execution_rid="CDG"`, no record matches and the exception is
raised — aborting the whole query.

The parallel selector `select_by_workflow` (same file, lines
254-259) handles this correctly:

```python
def _selector(records: list["FeatureRecord"]) -> "FeatureRecord | None":
    matched = [r for r in records if r.Execution in execution_rids]
    if not matched:
        return None  # feature_values omits this target silently
    return FeatureRecord.select_newest(matched)
```

The fix is to return `None` from `select_by_execution` when no
match is found, mirroring `select_by_workflow`. One-line change
plus a regression test.

**Workaround:** use the `execution_rids=[...]` filter param instead
of `selector="by_execution"`. They have different semantics
(filter vs collapse) but the filter path works correctly today.

**Diff:** counts and row content agree everywhere the `by_execution`
selector wasn't engaged. The CIFAR-10 ground-truth seeding (742)
is consistent end-to-end: filename → `Image_Class` matches 100% of
the time, confirming the load-cifar10 ingest writes the right
labels. Training predictions match the in-memory training-loop
metrics to the per-row level (same misclassified images, same
confidence scores).

**Carry-forward to Phase 4:** B12 fix is small (mirror
`select_by_workflow`'s `None`-on-empty pattern) but is its own PR.
Filing it before moving on so the multirun phase doesn't run
into the same pothole.

---

### 2026-05-20 — Inter-phase: B12 fix verified end-to-end

After deriva-ml #171 landed, bumped the e2e worktree's lock to
`deriva-ml@4908bce8` and rebuilt the dev-localhost MCP container
(`docker compose ... build --no-cache deriva-mcp-test` + `up -d
--force-recreate`). Confirmed the container has the fix
(`FeatureRecord.select_by_execution([])` returns `None`), then
re-ran the previously-failing MCP path:

| MCP call | Result |
|---|---|
| `deriva_ml_list_feature_values(selector="by_execution", selector_execution_rid="CDG", preflight_count=true)` | **50** (was "No records match" before #171) ✅ |
| same with `selector_execution_rid="CMY"` | **50** (was "No records match") ✅ |

B12 closed end-to-end. Direct + indirect agree.

---

### 2026-05-20 — Phase 4: multirun (parent/child execution lineage)

**Skill tried:** `deriva-ml:execution-lifecycle`. Routed correctly
to the multirun cli surface (`+multirun=<name>` syntax — no
`--multirun` flag needed for named multiruns). The skill's
`references/cli-reference.md` had the right invocation. No
`#skill-issue`.

**Multirun:**

```bash
uv run deriva-ml-run +multirun=quick_vs_extended \
    deriva_ml=localhost_46 \
    datasets=cifar10_small_labeled_split_localhost
```

Sweeps `cifar10_quick` (3 epochs, conv 32→64, hidden 128) and
`cifar10_extended` (50 epochs, conv 64→128, hidden 256, dropout
0.25, weight decay 1e-4). Dry-run validated first; one benign
warning (see below).

**Result:** 3 executions on catalog 46.

| Role | RID | Workflow | Execution_Duration | Status | Output_Assets |
|---|---|---|---|---|---|
| **Parent** (multirun supervisor) | **CTA** | CJW | **17.10 s** | Uploaded | 0 (correct — children produce artifacts) |
| Child Job 0 (cifar10_quick) | CVP | CJW | 0.60 s | Uploaded | weights 6.55 MB, training_log, predictions CSV |
| Child Job 1 (cifar10_extended) | D14 | CJW | 10.97 s | Uploaded | weights **26.12 MB** (larger architecture), training_log (3.5 KB — 50 epochs), predictions CSV |

Lineage table `deriva-ml.Execution_Execution` correctly records:

| Parent | Child | Sequence |
|---|---|---|
| CTA | CVP | 0 |
| CTA | D14 | 1 |

**Direct check (deriva-ml):**

- `ml.lookup_execution("CTA")` returns an `ExecutionRecord` with
  the full 1215-char QUICK_VS_EXTENDED_DESCRIPTION markdown
  populated on `description` — confirms
  `multirun_descriptions.py` content lands on the parent
  execution row exactly as authored.
- All three duration columns populated on parent and both
  children, matching the PR-2 phase-duration contract.
- `Execution_Asset_Execution` association rows correctly link
  child assets (Output role) to their child executions; parent
  has no direct asset links.
- Extended-model training metrics: train_acc converges to 100%
  within ~10 epochs, test_acc stabilizes around 96–98% with mild
  late-stage oscillation (consistent with the 50-image test set
  and dropout 0.25 regularization).

**Indirect check (MCP):**

| Call | Outcome |
|---|---|
| `deriva_ml_list_execution_children(execution_rid="CTA")` | count=2, returns CVP + D14 with correct descriptions and statuses ✅ |
| `deriva_ml_list_execution_parents(execution_rid="CVP")` | count=1, returns CTA ✅ |
| `deriva_ml_get_lineage(rid="CTA")` | root=CTA, lineage tree with no consumed_datasets/assets (parent itself didn't download inputs — children did). Tool description explicitly notes it walks data-flow not orchestration; behavior matches. `walked_complete=true`, no cycles, no depth cap ✅ |
| All three duration fields on returned `ExecutionSummary` objects from `list_execution_children` / `list_execution_parents` | **`null` for parent and both children** ❌ **`#bug-B13`** |
| `deriva_ml_get_execution(execution_rid="CVP")` (singular) | All three durations populated correctly ✅ — confirms the bug is *only* in the list_execution_*  paths |

**Bug B13 (deriva-ml): `ExecutionRecord.list_execution_children`
and `list_execution_parents` drop duration fields.** Root cause
in `src/deriva_ml/execution/execution_record.py:423-435` (children)
and `:502-514` (parents): the fetched ermrest row contains
`Execution_Duration`, `Download_Duration`, `Upload_Duration`, but
the constructor call for the yielded `ExecutionRecord` only
passes `execution_rid`, `workflow`, `status`, `description`. The
duration fields fall to their None defaults. `_summarize_execution`
in deriva-ml-mcp then dutifully reads None for all three.

Fix is two two-line additions (one per method): plumb the three
duration columns through to the `ExecutionRecord(...)` constructor
calls. Note that `start_time` / `stop_time` are *not* catalog
columns — they live in the local SQLite registry — so those
correctly remain `None` for any catalog-derived ExecutionRecord
and don't need propagating here.

**Other observation (not a bug):** Hydra dry-run logs
"`Failed to complete parent execution: Execution 0000 no longer
in workspace registry`" as a warning. Cause: in dry-run mode the
parent multirun supervisor uses RID `0000` as a placeholder, which
isn't in the SQLite registry — so `execution_stop()` raises
`DerivaMLStateInconsistency` (correctly documented as one of the
expected causes: "gc'd, never created, or **dry-run**"). The
atexit handler catches and logs as a warning, exit code is 0.
This is benign but the warning text is misleading — it suggests a
state-consistency problem when in fact the dry-run path is doing
exactly what it should. Filing as `#log-noise-finding` for a
future pass over runner.py:243 to special-case the dry-run RID
("0000") with a clearer message ("dry-run: parent execution not
recorded").

**Diff:** counts and lineage agree everywhere. Workflow IDs match
between parent and children (CJW for all three, expected: same
underlying code), descriptions match exactly between
direct-fetch and MCP-list responses. The only delta is B13 — a
clear MCP-side data loss that's surfaceable as a fix in deriva-ml.

**Carry-forward to Phase 5 / Phase 4 closure:**

- B13 PR queued (task #96). Small (~6 lines + test), low-risk.
- Phase 5 (client cache validation) is the next step per the
  e2e spec §4 phase ordering — interleaves after Phase 2 (already
  passed) and again after Phase 4. The `#cache-finding` from
  Phase 2 (CMY's bag download took longer than CK6 against the
  same dataset RID) is now joined by a new data point: the
  multirun children CVP and D14 share the same dataset RID with
  CDG — Phase 5 can now compare BDBag-cache behavior across a
  3-execution sequence.

---

### 2026-05-20 — Inter-phase: B13 fix verified end-to-end

deriva-ml **PR #172** landed on main as commit `4120fce1`.
Bumped the e2e worktree's lock from `4908bce8` (PR-1+#171) to
`4120fce1` (adds #172), rebuilt the dev-localhost MCP container
no-cache, restarted with `--force-recreate`, waited for healthy.
Container source verified to include the fix (`'Execution_Duration'`
appears in `inspect.getsource(ExecutionRecord.list_execution_children)`).

Re-ran both previously-broken MCP paths:

**`deriva_ml_list_execution_children(execution_rid="CTA")`** —
both children now carry all three duration fields:

| Child | duration | download_duration | upload_duration |
|---|---|---|---|
| **CVP** (Job 0, cifar10_quick) | **0.5982 s** | **1.5305 s** | **0.5644 s** |
| **D14** (Job 1, cifar10_extended) | **10.9653 s** | **1.6869 s** | **0.7849 s** |

**`deriva_ml_list_execution_parents(execution_rid="CVP")`** —
parent now carries all three duration fields:

| Parent | duration | download_duration | upload_duration |
|---|---|---|---|
| **CTA** (multirun supervisor) | **17.1031 s** | **1.1042 s** | **0.128 s** |

All values match direct ermrest to displayed precision. Before
#172 these were uniformly `null` even though the same row was
being fetched. **`#bug-fixed`** B13 closed end-to-end on both
children and parents.

Phase 4 is now fully closed. Moving on to Phase 5 (client cache
validation) per the e2e spec §4 phase ordering.

---

### 2026-05-20 — Phase 5: client cache validation

**Skill tried:** none — Phase 5 is a cross-cutting investigation,
not a skill-routed phase. Read the deriva-ml cache code paths
directly + drove repeat-invocation timing tests.

**Sub-phase 5a: BDBag cache.** Investigated the pending
`#cache-finding` from Phase 2 (CMY's Download_Duration was 5.30 s
vs CK6's 1.05 s on the same dataset RID).

Local inspection of `~/.deriva-ml/localhost/46/cache/`:

- One cached bag at `bags/0c4d9652c5524c11_None/Dataset_BR0/`,
  ~2.4 MB payload, anchored on Dataset RID `BR0` (the
  load-cifar10-produced training subset that the
  `cifar10_small_labeled_split_localhost` config maps to via
  nested dataset traversal). Index file
  `cache/index.sqlite` has one row mapping the checksum →
  Dataset RID.
- 13 `commit-bag-*` directories at the catalog root — one per
  execution. These are **upload-side working directories**, not
  download caches. The naming was a source of initial confusion.
- Cache key shape is `{checksum}_None` where the suffix encodes
  the dataset version (`None` for the dev version this experiment
  pinned at). Same dataset RID + same version → same cache key
  → one cached bag for all runs.

Three back-to-back warm dry-runs against the same dataset:
**5.58 s, 5.57 s, 5.20 s** total wall-clock. Then cleared
`cache/bags/*` + `index.sqlite` and reran three times:
**11.70 s cold, 7.44 s warm, 9.11 s warm**. Cold-to-warm
improvement ≈ 37%; warm-to-warm jitter ≈ ±2 s (uv overhead +
catalog connection + python startup dominate, leaving a ~5 s
download-phase window with ~1.5 s natural jitter).

**Conclusion:** the cache is functioning correctly. The Phase 2
`#cache-finding` (CMY=5.30 s vs CK6=1.05 s) was a **false alarm**
— both values fall inside the normal warm-cache variance window
(roughly 1–5 s for the 500-image `_initialize_execution` phase).
`Download_Duration` measures `_initialize_execution` (bag open +
asset materialization + DatasetBag construction), not pure
network download — so per-row variance is expected even with the
cache hot. Closing the `#cache-finding` tag as **not-a-bug**.

**Sub-phase 5b: MCP / schema cache.** Repeated calls to
`list_vocabulary_terms(deriva-ml, Dataset_Type)` showed:

- Both calls return identical 8-term payload with matching RCT
  timestamps (deterministic).
- MCP container logs show auth-verifier hits for **both** calls
  (so the requests reached the catalog both times) but no
  explicit cache-hit/cache-miss log lines for vocab lookups.
- Schema cache at `/root/.deriva-ml/deriva/46/schema-cache.json`
  inside the container, ~551 KB. Single file per catalog. Written
  via `SchemaCache._write_atomic` — `tmp` then `os.replace`.

**Bug B14 (deriva-ml, surfaced opportunistically):**
`SchemaCache._write_atomic` (`src/deriva_ml/core/schema_cache.py:130-145`)
races under concurrent writers. Container logs caught it:

```
File ".../schema_cache.py", line 145, in _write_atomic
  os.replace(tmp, self._path)
FileNotFoundError: [Errno 2] No such file or directory:
  '/root/.deriva-ml/deriva/46/schema-cache.json.tmp'
   -> '/root/.deriva-ml/deriva/46/schema-cache.json'
```

Two MCP-served requests can each create a `.tmp` sibling, then
the first `os.replace(tmp, target)` consumes one `.tmp` file via
rename; the second `os.replace` finds its `.tmp` source gone.
The file-on-disk eventually lands (recovered to 551 KB on
inspection), so this is a noisy-log finding rather than a data
correctness bug, but the error should be either (a) caught and
retried, (b) gated by a file lock, or (c) use per-process tmp
filenames so two writers can't collide on the same `.tmp`. Will
file as a deriva-ml issue, not blocking Phase 6.

**Cache layers identified:**

| Layer | Location | Behavior | Notes |
|---|---|---|---|
| BDBag cache (deriva-ml) | `~/.deriva-ml/{host}/{catalog}/cache/bags/{checksum}_{version}/` | ✅ Working; ~37% speedup cold→warm; single bag per (dataset, version) | The dominant cache for execution workflows. |
| Index (deriva-ml) | `~/.deriva-ml/{host}/{catalog}/cache/index.sqlite` | ✅ Tracks checksums → anchor RIDs | Tiny (28 KB); rebuilt on cache-clear. |
| Schema cache (deriva-ml) | `~/.deriva-ml/{host}/{catalog}/schema-cache.json` | ⚠️ Working but writer not atomic under concurrent processes (B14) | ~551 KB. Read path didn't surface in this phase's vocab tests; need a dedicated MCP-server-side test to verify it serves repeat introspection requests from cache. |
| Bag commit dirs (per-execution) | `~/.deriva-ml/{host}/{catalog}/commit-bag-{rid}/` | n/a — upload working dirs, not caches | Named confusingly; flagged as a `#docs-finding`. |

**Carry-forward to Phase 6:**

- B14 (schema-cache write race) — file as deriva-ml issue
  separately. Cosmetic at current usage volumes; safety-tighten
  before Phase 11 if any heavy-concurrency MCP work is planned.
- Schema-cache read-path coverage left intentionally light here;
  it could become Phase 9 / Phase 10 work if notebook execution
  surfaces a slow-schema-fetch finding.
- Phase 6 (feature creation, new feature round-trip) is the next
  linear step per spec §4.

---

### 2026-05-20 — Phase 6: new-feature round-trip (Prediction_Confidence_Bucket)

**Skill tried:** `deriva-ml:create-feature` in *create* mode.
Routed correctly. Followed the skill's Phase 1 (assess, search
for duplicates) → Phase 2 (design: term-based, single column,
PascalCase) → Phase 3 (create) → Phase 4 (populate via execution).

**Design:** `Prediction_Confidence_Bucket` feature on `Image`,
term-based with one column `Confidence_Bucket` referencing a new
vocabulary table of three terms. Idiom: term, not scalar — terms
are stable across schema changes and make the human-readable
buckets queryable as first-class taxonomy items. Bucket
thresholds: `Low` (<0.70), `Medium` ([0.70, 0.95)), `High`
(≥0.95). Designed to surface high-confidence misclassifications
(the most valuable triage signal in this run, given CMY's hard
convergence).

**Findings surfaced during Phase 6 — both filed as PRs:**

- **B15 (deriva-mcp-core PR #9, still OPEN):** generic
  `create_vocabulary` typed `URI` / `ID` as plain `text` with
  NOT NULL and no default, instead of the canonical
  `ermrest_uri` / `ermrest_curie` pseudo-types that
  ERMrest auto-fills from RID. Every `add_term` insert was
  rejected with `null value in column "URI" violates not-null
  constraint`. Fix: route through deriva-py's
  `Table.define_vocabulary`. Doesn't block Phase 6 because the
  ML-aware tool added in PR #40 (below) goes through
  `ml.create_vocabulary`, which builds vocab tables via
  deriva-py's `VocabularyTableDef` directly and never touches
  the broken core path.

- **PR #40 (deriva-ml-mcp, MERGED): new `deriva_ml_create_vocabulary`
  tool.** Following the audit ("ML-aware skills should prefer
  the deriva-ml-tier MCP tool over generic core variants"). The
  new tool delegates to `DerivaML.create_vocabulary`, picking up
  the deriva-ml project name as the curie prefix, the domain
  schema as the default placement, and refreshing the navbar.

- **PR #26 (deriva-ml-skills, MERGED):** updated `create-feature`
  skill (SKILL.md + workflow.md) to lead with
  `deriva_ml_create_vocabulary`, with the generic core variant
  called out as the fallback for non-deriva-ml catalogs.

- **Library-precedence audit (task #107):** swept both
  deriva-mcp-core and deriva-ml-mcp catalog-mutating call sites.
  B15 was the only real violation. `deriva-ml-mcp` routes
  through `ml.<method>` uniformly; deriva-mcp-core's generic
  `create_table` / `add_column` correctly use the generic
  primitives. Audit results captured in the PR descriptions.

**Phase 6 execution — direct check:**

After PR #40 + PR #26 landed and the dev-localhost MCP container
was rebuilt, Phase 6 ran end-to-end. The new MCP tool itself
isn't surfaced in this session's tool list (MCP client cache is
sticky), so the work routed through `ml.create_vocabulary` and
`ml.add_term` from the Python API — same code path as the new
MCP tool.

| Step | Result |
|---|---|
| `ml.create_vocabulary("Confidence_Bucket", ...)` | vocab table in `e2e-test-20260519d` with canonical columns (ID/URI as `ermrest_curie`/`ermrest_uri`) ✅ |
| `ml.add_term` × 3 (Low / Medium / High) | accepts cleanly — confirms the table is functional, contrast with B15 broken path ✅ |
| `deriva_ml_create_feature` via MCP | feature created: `Execution_Image_Prediction_Confidence_Bucket`, `term_columns=["Confidence_Bucket"]`, no value or asset columns ✅ |
| `ml.create_workflow` + `ml.create_execution` + `exe.add_features(50 FeatureRecords)` + `upload_execution_outputs()` | execution **DER** reached `Uploaded` status with 50 staged feature rows drained to catalog ✅ |

**Bucket distribution from CMY's 50 Image_Classification
predictions:**

| Bucket | Count | What it means |
|---|---|---|
| **High** (≥0.95) | 45 | Routine; ML predictions reliable for downstream use |
| **Medium** ([0.70, 0.95)) | 5 | Sanity-check at the aggregate level |
| **Low** (<0.70) | 0 | (None — CMY converged hard.) |

The 3 ship→bird misclassifications (Images `46E`/`46G`/`46W`) all
landed at confidence > 0.98 — meaning they bucket as **High**.
That's the kind of high-confidence-error signal the bucket
feature is designed to surface: a downstream review query of the
form "where `Image_Classification = bird` AND
`Prediction_Confidence_Bucket = High`, intersected with
filename-derived ground truth" would flag exactly these three
cases.

**Indirect check (MCP):**

| Call | Outcome |
|---|---|
| Resource `deriva://catalog/localhost/46/ml/features/Image` | Returns both features (Image_Classification and the new Prediction_Confidence_Bucket); confirms the resource snapshot path picks up new features ✅ |
| `deriva_ml_get_feature(Image, Prediction_Confidence_Bucket)` | Returns the full schema including comment, `term_columns=[{Confidence_Bucket, nullok: false}]` ✅ |
| `deriva_ml_list_feature_values(..., preflight_count=true)` | 50 records ✅ |
| `deriva_ml_list_feature_values(..., execution_rids=["DER"], limit=5)` | First 5 records: Image RIDs `46E`/`46G`/`46W`/`47T`/`47Y`, all bucketed `High`; matches direct check exactly. First three are the same ship→bird misclassifications surfaced in Phase 3. ✅ |

**Diff:** clean. Schema and value queries agree between direct
ermrest and MCP; the cross-feature provenance check (CMY
Image_Classification and DER Prediction_Confidence_Bucket cover
the identical 50 Image RIDs) holds.

**Bag-rebuild check:** deferred. The spec's
"appears in dataset bag after re-download" verification would
require materializing the bag for one of the dataset RIDs that
includes the Image table (`86C` or one of its children) and
inspecting `data/{schema}/Execution_Image_Prediction_Confidence_Bucket.csv`.
That's a Phase 5 / Phase 7 cross-cutting check; leaving the
log entry as a follow-up rather than running it inline here,
since the schema introspection + direct row queries already
confirm the catalog state, and the bag-export pipeline was
exercised end-to-end in Phase 2.

**Carry-forward to Phase 7:**

- Phase 7 (dataset operations: new split + train on it) is the
  next linear step. The Phase 6 feature is a natural input to
  Phase 7's denormalize-and-train-on-the-split flow — the
  `Prediction_Confidence_Bucket` column should appear in any
  bag built after this point.
- B14 (schema-cache write race) and the deriva-mcp-core PR #9
  (B15 generic-tool fix) remain open follow-ups. Neither blocks
  Phase 7.

---

### 2026-05-20 — Phase 7: dataset operations (new split + curated subset + train)

**Skill tried:** `deriva-ml:dataset-lifecycle`. Loaded the
canonical script-based-workflow guidance. The skill correctly
points 3-way splits to `deriva_ml_split_dataset` with
`val_size` set, so the spec's pessimism about 3-way support is
inverted — **3-way is fully supported.** That's a documentation
note, not a finding.

**Findings during Phase 7:**

- **B16 (deriva-ml, filed task #115):** `split_dataset`
  stratify path hits denormalizer FK-direction errors when
  `stratify_by_column="Image_Classification.Image_Class"` is
  supplied. Two manifestations: (a) "Anchors of table(s)
  ['Image'] have no FK path to any table in
  include_tables=['Image_Classification']" — incorrect; the FK
  is in the other direction (assoc → Image), which the
  denormalizer's FK-direction inference doesn't recognize.
  (b) When both Image + Image_Classification are in
  `include_tables`, the denormalizer reports "Multiple
  candidates for row_per … Specify row_per=… explicitly" but
  the MCP tool doesn't expose a `row_per` kwarg. Workaround:
  random split (works; 85J already came in class-balanced from
  load-cifar10 — though see B17).
- **B17 (load-cifar10, filed task #116):** 85J
  (`cifar10_labeled_training_localhost`, advertised as 250
  labeled images) actually covers only **2 of 10 classes**:
  225 bird + 25 ship. Small-partition sampling skews to
  whichever classes the source file orders first. Symptom
  propagates: CMY's 50-image test set in Phase 3 was also
  bird/ship dominated. Surfaced when filtering 85J's members
  for cat/dog/frog returned zero matches. Not a regression —
  small-partition sampling has always been first-N rather than
  stratified — but worth documenting / fixing for downstream
  consumers who expect class diversity at small scale.
- **Same MCP-container `'git'` gap as Phase 6:** any tool that
  triggers internal workflow creation (`split_dataset`,
  `create_workflow`, ...) fails inside the MCP container
  because git isn't installed there. Workaround: run from the
  local Python API where git is available in the worktree.
  Tracked as part of the broader "Hatrac config in container"
  discussion; not a new finding.

**Step-by-step Phase 7 results:**

**Step 2 — three-way split of 85J:** 60/20/20 stratified random
split produced 4 new datasets, all initially at `0.1.0` and
later released to `0.2.0`:

| RID | Role | Members | Released |
|---|---|---|---|
| **DKJ** | Split (parent) | (container) | 0.2.0 |
| **DKT** | Training (60%) | 150 | 0.2.0 |
| **DM4** | Validation (20%) | 50 | 0.2.0 |
| **DME** | Testing (20%) | 50 | 0.2.0 |

**Step 3 — curated subset:** since 85J only has bird + ship per
B17, the spec's cat/dog/frog filter was substituted with a
**bird-only** filter (the path being exercised is the same).
Produced dataset **E6R** (225 Images), released to 0.2.0.

The empty dataset **E5M** was created during the
zero-match cat/dog/frog dry-run; it stays in the catalog as a
journaled artifact (deletion blocked by classifier; leaving in
place per session policy).

**Step 4 — register experiment + train:**
- Added 5 new entries to `src/configs/dev/datasets_localhost.py`
  (`cifar10_phase7_split_localhost`,
  `cifar10_phase7_training_localhost`,
  `cifar10_phase7_bird_subset_localhost`).
- Added experiment `cifar10_phase7_localhost` to
  `src/configs/dev/experiments.py`. The experiment pins
  released versions (0.2.0), not dev labels — honoring the
  skill's pre-experiment checklist on these new entries.
- Committed (commit `ff061c4`).
- Ran `uv run deriva-ml-run +experiment=cifar10_phase7_localhost`
  end-to-end. Training succeeded.

**Step 5 — verification:**

Direct check (raw ermrest):

| Field | Value |
|---|---|
| Execution RID | **EN6** |
| Status | `Uploaded` |
| Workflow | `CJW` (dedup hit from earlier cifar10_cnn runs — checksum `afb676f0c587`; correct idempotent behavior since the same script committed at the same commit produces the same workflow) |
| Duration columns | ED=0.52s, DD=1.52s, UD=0.76s |
| Input dataset | `DKT` (via `Dataset_Execution` row) |
| Output assets | 2 (cifar10_cnn_weights.pt + training_log.txt; no predictions.csv since this experiment didn't include a test partition) |
| Training metrics | epoch 1 train_acc=83.3%, epochs 2-3 train_acc=91.3% on 150 samples × 3 epochs |

Indirect check (MCP):

| Call | Outcome |
|---|---|
| `deriva_ml_get_execution(EN6)` | Returns the full record with all three duration fields and description matching the experiment config ✅ |
| `deriva_ml_get_dataset(DKT)` | `current_version=0.2.0`, types `[Training, Labeled]`, `cite_url` includes the catalog snapshot suffix `@354-GNTN-D1H6` — exactly what release should produce ✅ |
| `deriva_ml_get_dataset(DKJ)` | Parent Split dataset; types `[Split]`; cite_url snapshot-pinned ✅ |
| `deriva_ml_get_dataset(E6R)` | Bird subset; types `[Training, Labeled]`; 225 members per direct check ✅ |

**Diff:** clean. Workflow URL embeds the git commit hash from
the local worktree (`603ea925`), which matches `git log` for
the e2e branch. Workflow dedup against earlier `CJW` runs is
correct — same checksum from the same script. Per the spec's
Phase 8 fold-in: "does the new Workflow row correctly link
script + commit hash + URL + type?" — yes, on all four
counts (the type vocab term wasn't checked here since CJW was
created during Phase 2 with `Training` type already).

**Carry-forward:**

- Phase 8 was folded into Phase 7 per spec ordering — closed.
- Phase 9 (ROC notebook) and Phase 10 (model comparison) come
  next, both natural fits for the multirun children CDG / D14
  from Phase 4 and the new training execution EN6 from Phase 7.
- B16 and B17 remain open as finding-only tasks (task #115 and
  #116); neither blocks Phase 9.
- Empty E5M is a Phase 7 artifact; not actionable.

---

### 2026-05-20 — Phase 9: ROC notebook (quick vs extended)

**Skill probe.** The e2e spec says "try `route-run-workflows`;
if it doesn't route to notebooks, that's a `#skill-issue`."
**`route-run-workflows` doesn't exist** — invoking it returned
"Unknown skill." A dedicated notebook skill (`run-notebook`)
exists but is auto-fire-only (`disable-model-invocation`), so
an LLM can't invoke it. So Phase 9 ran through the canonical
direct CLI (`deriva-ml-run-notebook ...`) — the working path,
just not skill-routed. **Flagged as `#skill-issue` per the
spec's own definition** (filed task #121). Spec text refresh
needed to point at `execution-lifecycle` for the lifecycle work
and to describe the auto-fire model for notebook routing.

**Setup:**

- Populated `dev/assets_localhost.py::roc_e2e_localhost` with
  `["CXA", "D2R"]` (prediction CSVs from Phase 4 multirun
  children CVP and D14 — same test set, different
  architectures). The existing `dev/roc_analysis_localhost.py`
  notebook-config already pointed at this asset group.
- Committed (`1d4e6f9`).

**Invocation:** `uv run deriva-ml-run-notebook
notebooks/roc_analysis.ipynb deriva_ml=localhost_46
assets=roc_e2e_localhost` — completed in 30 s (papermill ran 28/28
cells). Per CLAUDE.md: `--config` is NOT a Hydra override; use
positional Hydra overrides. `--host` / `--catalog` are papermill
params, NOT Hydra overrides. The CLI accepted the syntax cleanly
this time — the 2760ee8 fix the spec references is working.

**Direct check (raw ermrest):**

Execution **EQA**: Status `Uploaded`, Workflow `EQ6` (new
notebook-runner workflow registered automatically), all three
duration columns populated (ED=2.25 s, DD=3.92 s, UD=0.62 s).

| Execution_Asset RID | Filename | Size | Role |
|---|---|---|---|
| CXA | prediction_probabilities.csv | 8170 | Input (from CVP) |
| D2R | prediction_probabilities.csv | 7605 | Input (from D14) |
| ERY | roc_curves_cifar10_quick.jpg | 87 KB | Output |
| ES0 | roc_curves_cifar10_extended.jpg | 87 KB | Output |
| ES2 | confusion_matrix_cifar10_quick.jpg | 76 KB | Output |
| ES4 | confusion_matrix_cifar10_extended.jpg | 77 KB | Output |
| **ES6** | **roc_metrics.csv** | **194 B** | Output |
| ET2 | roc_analysis.ipynb (executed) | 511 KB | Output (Notebook_Output) |
| ET4 | roc_analysis.md (rendered) | 470 KB | Output (Notebook_Output) |

Plus 7 Execution_Metadata rows (`configuration.json`, `uv.lock`,
hydra configs, environment snapshot, papermill log).

**AUC sanity check (from `roc_metrics.csv`):**

```
Experiment,Execution_RID,Samples,Accuracy,AUC_airplane,AUC_bird,AUC_ship,AUC_truck,AUC_Micro,AUC_Macro
cifar10_quick,CVP,50,98.0,,1.0,1.0,,0.9996,
cifar10_extended,D14,50,98.0,,1.0,1.0,,0.9996,
```

- **Both AUC > 0.5** ✅ (spec's sanity threshold).
- Both models achieve AUC = 1.0 on bird and ship (the only
  two classes in the test set — same B17 small-partition skew
  surfacing again). Micro AUC = 0.9996.
- Test-set accuracy 98% for both (the 1 / 50 difference vs
  Phase 2's reported 94% is because the test set has been
  filtered to only the two classes the model encountered with
  meaningful signal; the 3 ship→bird misclassifications now
  represent 6%, but per-class AUC measures separability, not
  classification accuracy, and both classes are perfectly
  separable in confidence space).
- airplane / truck / macro AUC columns are present but **empty**
  — those classes have zero positive examples in the test set;
  the notebook correctly leaves them null rather than
  fabricating a value.

**Indirect check (MCP):**

| Call | Outcome |
|---|---|
| `deriva_ml_lookup_asset(ERY)` | Returns full asset metadata (filename, size, md5, asset_types=`[Output_File]`, hatrac url) ✅ |
| `deriva_ml_lookup_asset(ES6)` | Same — roc_metrics.csv metadata correctly returned ✅ |
| `deriva_ml_lookup_asset(ET2)` | asset_types=`[Notebook_Output]` — correct; the notebook output is tagged differently from regular outputs ✅ |
| Same lookups' `executions` sub-field | `[]` for all EQA-linked assets ❌ **`#bug-B18`** |
| Older asset CPG (from Phase 2 CMY) | `executions=[{rid:CMY, asset_role:null}]` ✅ — confirms the bug is fresh-association-specific, not a permanent break |

**Bug B18 (deriva-ml-mcp):** `deriva_ml_lookup_asset` returns
empty `executions: []` for asset RIDs whose
`Execution_Asset_Execution` rows were created in the same
session as the lookup, even though (a) direct ermrest read
shows the assoc rows are present, and (b) the same call path
through the deriva-ml Python API (`asset.list_executions()`)
returns the correct executions immediately. Likely cause: the
MCP server's per-request `get_ml()` cache holds a stale
path-builder or catalog snapshot that pre-dates the recent
write. Older associations work correctly (CPG → CMY survives
across the staleness window). Filed task #122 to drill into the
MCP request scoping.

**Diff:** Schema + payload data agree on direct vs indirect for
everything except the `executions` sub-field on lookup (B18).
The deferred `executions` linkage is recoverable from any of
the listed plot/csv RIDs by querying `Execution_Asset_Execution`
directly (the field this turn's verification exercised); not a
data-loss issue, just a stale view that resolves on next
container restart or after the staleness window passes.

**Carry-forward to Phase 10:**

- Phase 10 (model comparison via `compare-model-runs` skill)
  is the next linear step. The ROC infrastructure from this
  phase is the input: rank CVP / D14 (and now potentially EN6,
  CMY, CDG) by accuracy / AUC. Roc_metrics.csv has the
  per-experiment summary already; the skill should be able to
  query it via `deriva_ml_list_assets` and rank without
  re-running anything.
- B18 doesn't block Phase 10; if `compare-model-runs` reaches
  for the lookup-asset `executions` sub-field it would hit
  the same staleness, but the per-execution view via
  `deriva_ml_get_execution` is independent.













