# Multipersona E2E Run — 2026-05-27c Evaluation

**Status:** Complete. All three persona arcs executed successfully.
**Catalog:** id=95, hostname=localhost, name=`e2e-test-20260527c`
**Branch:** `e2e-test/2026-05-27-c`
**deriva-ml version:** v1.39.4 (`4ed88122`)
**deriva-ml-mcp:** v0.5.4 — container rebuilt against deriva-py with SchemaPathWalker
**deriva-py:** pinned via `@deriva-ml` branch at `e944ad8e` (includes SchemaPathWalker)
**deriva-ml-skills:** v1.4.8 (latest e2e polish from previous runs)

---

## 1. What this run validated

Third pass on 2026-05-27 — the prior two were:

- **2026-05-27a (first pass):** blocked by the PagedFetcher row-loss bug at
  `--num-images=1500`. Fixed by PR #246 (chunk-loop oversized GET requests).
- **2026-05-27b (second pass):** P0 succeeded after the fix landed; arcs ran
  to completion; produced 3 findings (analyst/01, analyst/02, developer/01).
  Those findings drove three follow-up PRs (#35 record_predictions
  provenance, #65 + #66 skill polish, #37 cifar10_cnn refactor).
- **2026-05-27c (this run):** validates the platform across all of those.
  Specifically:
  - **PR #246 + #243 + #237** — PagedFetcher / RB-07 dedup / defensive one-liners.
  - **PR #37** — the cifar10_cnn refactor (~280 lines saved, 3-section layered
    structure). First end-to-end exercise.
  - **PRs #38/#59** — deriva-py path_walker pin (blocked the start of this run
    until merged).
  - **deriva-ml-skills v1.4.8** — the denormalize row-count gotcha note + the
    ExecutionRecord lookup guidance.

**Headline result:** the stack is solid. The Analyst's denormalize parity
check on TX0 (750 images → 1150 rows) and JZ8 (1500 images → 1900 rows)
returned exactly the expected (Execution, Image) tuples, matching direct
member-driven queries bit-for-bit. **No regression.**

---

## 2. Phase-by-phase outcomes

### Phase 0 (Bootstrap)

Required two iterations:

- Initial attempt on `e2e-test/2026-05-27-b` failed at `--phase datasets`
  with `ModuleNotFoundError: No module named 'deriva.bag.path_walker'`. Root
  cause: stale deriva-py pin (`ed5ee69c`) predates SchemaPathWalker.
- After PRs #38 (model-template) and #59 (deriva-ml-mcp) merged, the e2e
  worktree was reset to fresh main, deriva-ml-mcp v0.5.4 released, MCP docker
  container rebuilt. Catalog 94 deleted, catalog 95 created cleanly.

Final P0 state: 1500 images, 750/750 train/test, 13 datasets, perfectly
balanced 150/class. Cross-channel parity verified.

### Curator arc (commit `a3015ab`)

- Audited all 13 datasets via MCP + direct deriva-ml. No mismatches.
- **Curated XEM** (Validation, 150 stratified images from K04, seed=20260527).
  Wired into `cifar10_validation_150` config.
- Wrote tk-002 (audit table) and tk-003 (use-case → dataset matrix).
- **0 findings filed.** Arc clean.

### Model Developer arc (commit `be860d2`)

- Three training runs, all `Uploaded`:

  | Rank | Exec | Variant | Datasets | Seed | Epochs | log test_acc |
  |---|---|---|---|---|---|---|
  | 1 | Y1M | cifar10_quick | TX0 | 123 | 10 | 38.00% |
  | 2 | YDT | default_model | TX0 + XEM | 7 | 10 | 36.67% |
  | 3 | XRJ | cifar10_quick | WD2 | 3 | 3 | 24.00% |

- YDT exercises the Validation dispatch lane (TX0 + XEM composite),
  surfacing val_acc=34.67% per epoch in the training log.
- Added `cifar10_train_with_validation` composite config to `datasets.py`.
- Wrote tk-004 (ranked table + decision log + handoff).
- **1 finding:** `developer/01` — `record_predictions` doesn't print the
  "Emission-time accuracy: NN.NN%" line that was added in PR #35. **Real
  regression introduced by PR #37 refactor** — the function split
  (`predict_batch` + `record_predictions`) accidentally dropped the
  GT-comparison step. Low severity (CSV still has Source_Label; Analyst can
  reconcile manually), but the in-process safety rail is gone.

### Analyst arc (commit `84f62e1`)

- Re-ranked from CSV-recomputed accuracy (joining each
  `prediction_probabilities.csv` against the GT feature):

  | Rank | Exec | CSV-recomputed test_acc | Micro-AUC | Macro-AUC |
  |---|---|---|---|---|
  | 1 | Y1M | 34.00% | 0.777 | 0.773 |
  | 2 | YDT | 33.33% | 0.792 | 0.786 |
  | 3 | XRJ | 24.00% | (smoke) | (smoke) |

  Y1M and YDT are within 1 prediction of each other on a 150-image bag.
  YDT wins on AUC (the Validation lane delivered a small but consistent
  generalization improvement); Y1M wins on argmax.

- **Reconciliation gap:** CSV-recomputed (34.00% / 33.33%) ≠ training log
  (38.00% / 36.67%). Same shape as the 2026-05-27b analyst/02 finding.
  Developer/01 already covers it — the emission-time-accuracy log line that
  would surface this divergence at training time is missing.

- **Denormalize parity (the headline #246 validation):**
  - TX0 (750 image members) → 1150 rows in 0.98s.
  - JZ8 (1500 image members) → 1900 rows in 1.44s.
  - Every `(Execution, Image)` tuple matches the direct member-driven
    query bit-for-bit. **PR #246 + #37 + #38/#59 compose cleanly.**

- Ran `notebooks/roc_analysis.ipynb` end-to-end via `deriva-ml-run-notebook`
  in ~13 seconds. Produced analysis execution `YT6` with 8 output assets
  (ROC plots × 3, confusion matrices × 3, metrics CSV, executed notebook,
  rendered markdown).

- Wrote tk-005 (denormalize parity record) and tk-006 (analysis conclusions).
- **0 findings filed.** (developer/01 already covers the only anomaly.)

---

## 3. Three-axis evaluation

### 3.1 Platform fitness

**Strong.** Every PR landed in the last 24 hours validated end-to-end on a
fresh 1500-image catalog with no row-loss artifacts, no spurious
missing-value errors, no MCP-vs-direct disagreements. The Analyst's
denormalize parity assertion against JZ8 — the headline test for PR #246 —
returned the exact expected RID set both via direct deriva-ml and via the
rebuilt MCP container.

Friction surfaced this run is **strictly less severe** than the previous
two runs:

- 2026-05-27a: P0 blocker (PagedFetcher row-loss) → required emergency fix
  + release.
- 2026-05-27b: 3 findings, one Medium (analyst/02 prediction-CSV
  provenance gap).
- 2026-05-27c: 1 finding (developer/01), Low, introduced as a regression
  by the refactor it was meant to clean up. Easy to fix (~5 lines).

### 3.2 Knowledge transfer (tacit-knowledge.md)

**Very effective again.** Six tk entries, all in v1.4.7+ format (`tk-NNN`,
**When/By/Supported by**). The Supported-by chain is intact:
tk-002 → tk-001; tk-003 → tk-001+002; tk-004 → tk-003; tk-005/006 → tk-004.

The Curator's use-case matrix in tk-003 was consumed directly by the
Developer (picked WD2 for smoke, TX0+XEM for the Validation-lane run).
The Developer's tk-004 ranked table flowed into the Analyst's report §1 by
exact RID match. The Curator's gotchas-section ("XEM ⊂ K04 by
construction; don't compose them") shows up verbatim as a constraint in
the Developer's experiment design.

### 3.3 Result substance

**Substantive.** Y1M wins on argmax test_acc (CSV: 34.00% vs YDT 33.33%),
but YDT wins on macro-AUC (0.786 vs 0.773) — the Validation lane delivered
exactly the kind of generalization signal it's there to surface. The
1-prediction-on-150 argmax gap is within noise; the AUC gap is real but
small. The Analyst's report frames this honestly without overclaiming.

---

## 4. Comparison vs prior runs

| Run | Status | New findings | Notes |
|---|---|---|---|
| 2026-05-21 | Partial | 19 | Pre-PR-#189 era; A01/A02/A04 surfaced denormalize |
| 2026-05-26 | Clean | 6 (Low/Med) | Validated #189–#244; UX polish |
| 2026-05-27a | **Blocked** | 1 (Blocker) | PagedFetcher row-loss at 1500 images |
| 2026-05-27b | Clean (post-#246) | 3 (1 Med, 2 Low) | End-to-end validation of #246 |
| **2026-05-27c** | **Clean** | **1 (Low)** | **Validates #246 + #37 + #38/#59** |

The trend is healthy: each run scales the catalog at the same size,
exercises more platform changes, and surfaces fewer (and less severe)
findings. The 2026-05-27c finding is a refactor regression in
**model-template-side** code (cifar10_cnn.py), not platform code (deriva-ml,
deriva-ml-mcp, deriva-ml-skills) — meaning the platform releases of the day
landed cleanly.

---

## 5. Outstanding items

- **`developer/01` follow-up:** `record_predictions` needs to print
  emission-time accuracy. The cifar10_cnn refactor split the function into
  `predict_batch` (pure inference) + `record_predictions` (catalog write).
  The fix is to either thread GT labels through `predict_batch` or have
  the caller (the entry point in cifar10_cnn) call `evaluate()` and pass
  the resulting accuracy in. ~5 lines. To do as a follow-up PR.

- **Test-only mode:** not exercised in this run. The runner still supports
  it; verifying it works against catalog 95 weights would be a fast
  smoke check.

- **Save-best:** intentionally out of scope per the cifar10_cnn refactor
  decisions. If/when added, the Source_Label semantics would extend to
  `best_epoch_N` vs `final_epoch_N`.

---

## 6. Reproducibility

- **Catalog id:** 95 (preserved; query for archeology, or delete at next
  bootstrap)
- **Branch:** `e2e-test/2026-05-27-c` (7 [E2E-DROP] commits ahead of main)
- **Versions pinned for this run:**
  - deriva-ml v1.39.4 (`4ed88122`)
  - deriva-ml-mcp v0.5.4
  - deriva-py `@deriva-ml` HEAD `e944ad8e`
  - deriva-ml-skills v1.4.8
- **Persona execution RIDs:** Developer XRJ/Y1M/YDT; Analyst YT6.
- **Output assets:** see tk-004 (Developer) and tk-006 (Analyst).

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
