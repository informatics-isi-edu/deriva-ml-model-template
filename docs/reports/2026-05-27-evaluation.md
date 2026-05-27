# Multipersona E2E Run — 2026-05-27 Evaluation

**Status:** Complete. All three persona arcs executed cleanly post-PR #246 fix.
**Catalog:** id=93, hostname=localhost, name=`e2e-test-20260527`
**Branch:** `e2e-test/2026-05-27`
**deriva-ml version:** v1.39.4 (`4ed88122`) — includes #246 PagedFetcher row-completeness fix
**deriva-ml-mcp version:** v0.5.3 — container rebuilt against v1.39.4

---

## 1. What this run validated

This rerun was triggered by the user request "rerun the multi-persona
test using the same conditions as before" on the morning of 2026-05-27.
The first attempt (P0 step 7 at `--num-images=1500`) failed with a
spurious 50% missing-values error in the labeled stratified split.
Root-cause investigation pinned the bug at `PagedFetcher._fetch_rid_batch_with_fallback`
(lines 320-347 of `paged_fetcher.py`): the shrink-on-URL-too-long path
returned the first prefix that fit and silently dropped the suffix.

The fix landed as PR #246 (chunk-loop replacement; 153 lines added with
two regression tests). The rerun documented in this report exercises
the fix end-to-end across all three persona arcs at the 1500-image
catalog scale that originally triggered the bug.

**Headline result:** the fix holds. Cross-channel parity is perfect.
All three personas completed their arcs without hitting the original
defect or any regression.

---

## 2. Phase-by-phase outcomes

### Phase 0 (Bootstrap)

- Catalog 92 was attempted first; failed at `--phase datasets` due to
  the now-fixed bug. Catalog deleted, 92 abandoned.
- Catalog 93 created fresh post-fix. 1500 images, 750/750 train/test,
  1500 features, perfectly balanced 150/class across all 10 CIFAR-10
  classes. All 13 datasets created successfully.
- Cross-channel parity (MCP vs direct deriva-ml) verified at the end
  of P0; no disagreement.

### Curator arc (commit `a1d033d`)

- Audited all 13 datasets via MCP, cross-verified via direct API.
- Identified gap: bootstrap shipped no Validation-typed dataset, so
  the cifar10_cnn runner's Validation dispatch lane (D01 from
  2026-05-26 work) had no data to consume.
- **Curated variant created:** XEM (`Validation`, `Labeled`, 100
  stratified images from K04, seed=2026). Wrote
  `scripts/curator_create_validation.py`.
- Wrote tk-002 (audit) and tk-003 (handoff table with use-case →
  dataset RID mapping including the K04↔XEM overlap caveat).
- 0 findings filed. Arc clean.

### Model Developer arc (commits `4f53b8e`, `7b2b304`)

- Three executions trained, all `Uploaded`:

  | Rank | Exec | Variant | Dataset | Seed | Epochs | test_acc |
  |---|---|---|---|---|---|---|
  | 1 | XYG | `default_model` | TX0 | 123 | 10 | 42.00% (log) |
  | 2 | YAP | `cifar10_regularized` | TX0 + XEM | 2026 | 10 | 37.33% (log) |
  | 3 | XNE | `cifar10_quick` | WD2 | 42 | 3 | 24.00% (log) |

- All three exercised commit_output_assets (9 Execution_Asset rows
  total: weights + log + predictions per run).
- YAP specifically exercised the **Validation dispatch lane**
  (TX0+XEM composite dataset config) — this is the first run that
  has produced a real consumer for the D01 work from 2026-05-26.
- Wrote tk-004 (ranked executions + decision log + handoff).
- 1 finding: `developer/01` — `lookup_execution()` returns the
  read-only `ExecutionRecord` model rather than the live `Execution`
  handle with `.execution_assets()`. Naming/API confusion, low severity.

### Analyst arc (commit `6730d9a`)

- Ranked the three executions on Analyst-recomputed metrics (test
  accuracy from prediction CSVs, micro-AUC, macro-AUC):

  | Rank | Exec | test_acc (CSV) | Micro-AUC | Macro-AUC |
  |---|---|---|---|---|
  | 1 | **YAP** | 36.00% | 0.789 | 0.789 |
  | 2 | XYG | 34.67% | 0.781 | 0.779 |
  | 3 | XNE | 24.00% | 0.677 | 0.712 |

  Ranking reversed from the Developer's log-based view — see
  `analyst/02` finding.
- **Denormalize parity (the headline #246 validation):** ran
  `get_denormalized_as_dataframe` on JZ8 (1500 Image members) with
  `include_tables=["Image", "Execution_Image_Image_Classification"]`.
  Result: exact 1500 Image RID set, perfectly balanced 150/class on
  the ground-truth subset, zero missing, zero extra. The fetcher
  fix works at the scale that originally tripped it.
- Ran `notebooks/roc_analysis.ipynb` end-to-end via
  `deriva-ml-run-notebook`. Produced analysis execution YQ2 with 13
  linked Execution_Asset rows (3 ROC curves, 3 confusion matrices,
  1 ROC overlay YSC, 1 metrics CSV YSM, executed notebook YTW, plus
  markdown summary YTY).
- Cross-channel verified: MCP `deriva_ml_list_executions` and direct
  path-builder agree on all asset RIDs and the YQ2 → Y0E/YCP/XQC →
  XYG/YAP/XNE lineage.
- Wrote tk-005 (denormalize parity record) and tk-006 (analysis
  conclusions).
- 2 findings: `analyst/01` (denormalize raw row-count surprise —
  polish), `analyst/02` (committed prediction CSV doesn't reflect
  final-epoch model state — medium provenance gap).

---

## 3. Three-axis evaluation

### 3.1 Platform fitness

**Strong.** The 1500-image scale exercised the previously-broken
fetcher path on every persona arc and the row-completeness invariant
held everywhere we checked:

- Curator: MCP and direct agree on all 13 dataset member counts, all
  feature value totals.
- Developer: three training runs completed without hitting
  truncated-input artifacts (any pre-fix denormalize would have
  silently corrupted the labeled-split feature joins).
- Analyst: the explicit denormalize parity test on JZ8 returned the
  exact expected row set, and roc_analysis.ipynb successfully read
  per-image predictions CSV-aligned against ground truth.

The only platform friction surfaced was `developer/01` (low-severity
API naming polish) and `analyst/01` (low-severity denormalize-result
ergonomics — feature rows from non-GT executions get included in raw
row count). Neither is data-integrity.

### 3.2 Knowledge transfer (tacit-knowledge.md as a handoff artifact)

**Very effective.** Six tk entries, all in the post-PR-#65 format
(`tk-NNN`, **When**, **By**, **Supported by**, body). Each persona
read prior entries before acting:

- Developer's tk-004 cites tk-003's use-case table explicitly when
  picking TX0 + XEM for the regularized run.
- Analyst's tk-006 cites tk-004's ranked execution table when setting
  up the ROC comparison.
- The Supported-by chain is unbroken: tk-002 → tk-001;
  tk-003 → tk-001 + tk-002; tk-004 → tk-003; tk-005/006 → tk-004.

The Curator's handoff table format (use case → dataset RID → caveats)
is being reused naturally — both Developer and Analyst pulled
RID + version pairs directly from it.

### 3.3 Result substance

**Substantive answer produced.** The Analyst's report
(`docs/reports/2026-05-27-analysis.md`) ranks the three runs on
multiple metrics, identifies YAP (regularized) as the winner across
every Analyst-measurable axis, flags a real provenance gap (`analyst/02`
— prediction CSVs don't reflect final-epoch model state), and ties the
denormalize parity result to the #246 fix as end-to-end validation.

The YAP win on macro-AUC despite the Developer's log saying XYG was
better is the kind of finding the multipersona structure exists to
surface: the Developer reports training-loop accuracy; the Analyst
verifies against the artifact actually committed to the catalog. The
runner is emitting a save-best checkpoint's predictions without
labelling the source epoch — that's a small gap in the cifar10_cnn
runner, not in the platform.

---

## 4. Comparison vs prior runs

| Run | num-images | Failure mode | Personas | New findings | Notes |
|---|---|---|---|---|---|
| 2026-05-21 | 500 | Several denormalize / A01–A04 | 3 | 19 | Pre-PR-#189 era |
| 2026-05-26 | 500 | Clean | 3 | 6 (Low/Med) | Validated #189–#244 work |
| **2026-05-27** | **1500** | **Clean post-#246** | **3** | **3 (1 Med, 2 Low)** | **End-to-end validation of #246 fetcher fix** |

The pattern is healthy: each run scales the catalog larger, exercising
more of the platform; the #246 fix unblocks the 1500-image scale; new
findings shift from data-correctness to UX/polish (the right direction).

---

## 5. Open items / next-run inputs

- **`analyst/02` follow-up:** cifar10_cnn runner should emit
  predictions from the final epoch (or label the prediction CSV with
  its source epoch). Medium priority — affects what downstream
  consumers think they're getting.
- **`analyst/01` follow-up:** `dataset-lifecycle` skill could surface
  "denormalize raw row count includes feature rows from all
  executions; filter by `Confidence IS NULL` for GT-only rows." Low
  priority documentation polish.
- **`developer/01` follow-up:** rename `lookup_execution()` to
  `lookup_execution_record()`, or have it return the live `Execution`
  when the caller asks for asset enumeration. Low priority API polish.

---

## 6. Reproducibility

- **Catalog id:** 93 (preserved post-run; can be re-queried for
  archaeology, or deleted at the next bootstrap)
- **Branch:** `e2e-test/2026-05-27` (7 [E2E-DROP] commits ahead of
  model-template `main`)
- **deriva-ml SHA:** `4ed88122` (v1.39.4)
- **PR shipped during the run:** [deriva-ml#246](https://github.com/informatics-isi-edu/deriva-ml/pull/246)
- **Persona execution RIDs:**
  Developer: XNE, XYG, YAP. Analyst: YQ2.
- **Output asset RIDs:** see tk-004 (Developer) and tk-006 (Analyst)
  for the full mapping.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
