# Multipersona E2E Run — 2026-06-01 Evaluation

**Catalog:** id=2, hostname=`localhost`, schema `e2e-test-20260601`
**Branch:** `e2e-test/2026-06-01`
**Sibling versions:** deriva-ml v1.41.1 (@5377f999); deriva-ml-mcp surface via `dev-localhost` MCP server (authenticated)
**Evaluator channels:** `deriva_ml_*` MCP tools + read-only `deriva-ml` Python 1.41.1 (independent recomputation from raw asset bytes)

---

## Headline

**This run is clean.** Every load-bearing claim the three personas made about
catalog 2 holds up under direct inspection — and not just at the
"does the RID exist" level: I independently recomputed the analysis from the
raw prediction CSVs against PK6 ground truth and reproduced the report's
leaderboard (accuracy 20/32/20, macro-AUC 0.739/0.751/0.638, micro-AUC
0.643/0.757/0.630) **to the digit**, plus the collapse signatures (Smoke→horse
34/100, Fast-LR→deer 28/100), the Regularized run's confusion-pair table
(frog→cat 5, cat→dog 4, automobile→truck 4, ship→airplane 3, bird→deer 3, in
that exact order), and the calibration gaps. The Curator→Modeler→Analyst chain
is coherent end to end: the no-cross-families leakage constraint that the
Curator identified was honoured structurally by the Modeler (all three runs
consumed only the `Split` parent PJM) and confirmed by the Analyst's lineage
walk — which I re-ran and which holds. `tacit-knowledge.md` is genuine
decision-rationale, not state replication. The four persona findings are all
real, correctly scoped, and correctly severity'd (three Low/Medium platform-ergonomics
notes plus one harness-environmental note that should not be scored against the
platform). The single thing the user should act on is small: the recorded
analysis execution REJ carries a **stale, actively-misleading description**
("ROC curve analysis (default: quick vs extended training)") that names a
comparison that never happened — the Analyst already filed this; I concur and
would prioritize the lower-touch config fix.

No Blocker, no High. The indirect channel (MCP tools) and the direct channel
(deriva-ml Python) agreed everywhere I compared them.

## Catalog ↔ claim agreement (the load-bearing thread)

I verified a broad sample of concrete claims by **both** channels. Every check
passed. Highlights, with both readings where they differ in shape:

### Data substrate (Curator claims)
- **1100 images, single-execution ground truth.** Direct Python:
  `Image_Classification` holds 1400 rows; 1100 are `Confidence IS NULL`, all from
  a **single** execution `CWC`; 300 are predictions (100 each from QK8/QWA/R5C,
  all `Confidence` populated). MCP `deriva_ml_list_feature_values` first page
  agrees row-for-row (CWC rows, `Confidence: null`). ✅
- **Perfect 10×110 class balance.** Direct Python: every one of the 10 classes
  has exactly 110 GT rows. ✅
- **Split set-algebra (the leakage finding).** Re-ran the Curator's repro with
  membership set-intersection:
  - `F38 (1100) = F3T (550) ⊎ F44 (550)`, disjoint, union == F38. ✅
  - Small split: `F4W ⊆ F3T`, `F56 ⊆ F44`, `F4W ∩ F56 = 0` (proper mirror). ✅
  - Labeled split: `NF8 ⊆ F3T`, `NFJ ⊆ F3T`, `NF8 ∩ NFJ = 0`, `NFJ ∩ F44 = 0`. ✅
  - Small labeled: `PJW ⊆ F3T`, `PK6 ⊆ F3T`, `PJW ∩ PK6 = 0`, `PK6 ∩ F44 = 0`. ✅

  This confirms the Curator's central, downstream-load-bearing claim: NFJ/PK6 are
  carved 100% from the **training** pool F3T, so they are valid hold-outs only
  *relative to their own sibling train sets* — exactly the trap the finding
  describes.

### Executions, workflows, lineage (Modeler + Analyst claims)
- **3 training executions + 1 analysis execution, all `Uploaded`.** MCP
  `deriva_ml_list_executions`: 8 total — loader/setup (47Y, CWC, F28, NE0 on
  workflow 46T), training QK8/QWA/R5C (all on workflow **QK2** "CIFAR-10 2-Layer
  CNN"), analysis REJ (on workflow **REE** "Roc Analysis"). All status `Uploaded`.
  Matches the report's §2/§7 exactly. ✅
- **Prediction assets QN6/QY8/R7A map to QK8/QWA/R5C.** `deriva_ml_get_lineage`
  on each: QN6→QK8, QY8→QWA, R7A→R5C; each consumed only dataset PJM; F3T appears
  only via the *ancestor* splitter NE0 — **never F44/F38 anywhere**. The Modeler's
  "no F3T/F38/F44 was ever an input" and the Analyst's "traces back through PJM →
  F3T cleanly, never the reserved test pool" both hold. ✅
- **REJ consumed the three prediction CSVs.** `deriva_ml_get_lineage(REJ)`:
  `consumed_assets` = QN6, QY8, R7A (`prediction_probabilities.csv`); parents =
  R5C, QK8, QWA → PJM → NE0 → F28. Full provenance chain intact. ✅
- **All 10 output assets exist, linked to REJ, filenames match §7 table.**
  `lookup_asset` on RGP/RGR/RGT/RGW/RGY/RH0/RH2/RH4/RJC/RJE: all present,
  `execution_rid == REJ`, `asset_table == Execution_Asset`, filenames match the
  report's table 1:1 (e.g. RGP=`roc_curves_cifar10_quick_QN6.jpg`,
  RH4=`roc_metrics.csv`, RJC=`roc_analysis.ipynb`, RJE=`roc_analysis.md`). ✅
- **Executed notebook is real.** RJC = 28 cells, 16/16 code cells executed
  (all carry outputs), references PK6 / QWA / REJ / macro-AUC. ✅

### The metrics, recomputed independently
This is the strongest check. I downloaded the three raw prediction CSVs
(QN6/QY8/R7A), joined to PK6 ground truth, and recomputed from scratch with
sklearn — *not* reading the metrics CSV:

| Run | acc (recomputed / RH4 / report) | macro-AUC | micro-AUC | classes used |
|---|---|---|---|---|
| Smoke QN6 | 20.0 / 20.0 / 20% | 0.7388 / 0.7388 / 0.739 | 0.6429 / 0.6429 / 0.643 | 7 / 7 / 7 |
| Reg. QY8 | 32.0 / 32.0 / 32% | 0.7506 / 0.7506 / 0.751 | 0.7573 / 0.7573 / 0.757 | 10 / 10 / 10 |
| Fast-LR R7A | 20.0 / 20.0 / 20% | 0.6376 / 0.6376 / 0.638 | 0.6302 / 0.6302 / 0.630 | 10 / 10 / 10 |

All three columns agree across **independent recomputation, the catalog's
metrics CSV (RH4), and the report prose.** QWA wins on macro-AUC. The
report's "classes it actually uses" diagnostic (7/10/10) is reproduced exactly.
Fast-LR's near-chance per-class AUCs (deer 0.4967, horse 0.4961, cat 0.5683)
match the report's "deer 0.497, horse 0.496, cat 0.568."

**Two cosmetic slips in the report prose** (neither changes any conclusion; see
`findings/evaluator/01`):
1. §5 states the Regularized calibration gap as **+0.08**; the true gap is
   0.0738 (components 0.667/0.594, which round to the stated 0.67/0.59). It
   rounds to +0.07, not +0.08.
2. §4 says QWA has "no class below cat's 0.657" — but RH4 shows QWA's **deer
   AUC = 0.6533**, which is *below* cat's 0.6567. Cat is not the per-class
   minimum; deer is. The broader point (QWA's per-class AUC is far more even
   than Fast-LR's near-0.5 classes) still holds.

**Verdict on §3.1: no discrepancy between the reporting channel and the catalog.**
The indirect (MCP) and direct (Python) channels agreed on every shared check.
The only inaccuracies are two sub-0.01 prose rounding/min-selection nits in the
human-readable report, not catalog-vs-claim disagreements.

## Coherence of the team's deliverables

The chain reads as one continuous, honest story.

- **Curator → Modeler.** The Curator's headline output was not a data dump — it
  was a *decision*: which split families are safe to cross and which are a
  leakage trap, with the machine-checkable signal (set-intersection) spelled
  out because the catalog's type system can't express it. The Modeler consumed
  this directly: the no-cross-families rule drove the choice to train on the
  `Split` parent PJM (which structurally guarantees same-family train/eval),
  and the Modeler's tacit entry cites the Curator finding by path. A reader who
  starts at the Modeler entry can walk back to the Curator's reasoning without
  having been in the room.
- **Modeler → Analyst.** The Modeler deliberately produced *three runs with
  distinct training dynamics* (smoke / regularized / fast-LR) on one shared eval
  set (PK6), and recorded both a per-image feature row (argmax + confidence) and
  a wider per-class probability CSV. That second artifact is exactly what made
  ROC/AUC computable without rerunning anything — the Analyst's report calls
  this out, and it is true: I computed AUC from those CSVs myself.
- **Analyst → reader.** The report sets out to answer "which run is best, where
  does each fail, and do the differences make sense" for a non-ML reader, and it
  does — with the genuinely insightful move (ranking on AUC, not top-1, to
  separate the two runs that tie at 20%) properly motivated and properly
  caveated ("these are plumbing-validation runs, not capability claims"). The
  figures it references exist on the catalog and contain what it says.

The **no-cross-families constraint held end to end** — I verified it structurally
(lineage: only PJM consumed) and by set-algebra (PK6 ∩ F44 = 0). A fresh reader
given only this worktree and the catalog could reconstruct what the team did and
why. That is the coherence bar, and the run clears it.

## tacit-knowledge.md quality

Read top to bottom as a fresh contributor: **this is good tacit knowledge**, not
state replication. Five entries, each capturing a *decision and its rationale*:

- **Curator characterization entry.** Strong. The "one gotcha that matters
  downstream — split source pools" section is precisely the gotcha-surfaced-by-work
  the rubric prizes: it explains *why* NFJ/PK6 are a trap (valid relative to their
  siblings, but inside F3T), gives the only machine-checkable signal, and ends
  with explicit forward guidance ("Never pair NF8/NFJ or PJW/PK6 against
  F3T/F38/F44"). It even corrects a subtlety in the README's steer. The brief
  per-dataset counts it includes are *in service of* the decision, not a state
  table for its own sake — acceptable.
- **Modeler three-runs entry.** Captures the *why this dataset* (the leakage
  rule), the *why three configs* (did variation produce variation — yes), and a
  reproducibility note (seed=42, clean-tree provenance, dirty-tree override used
  only for read-only smoke tests). Decision-rationale throughout.
- **Modeler Hydra-grammar entry.** A durable behavior-gotcha ("Hydra rejects
  free-text `description=` with parens/commas, it's a parse error not a runtime
  error") with the workaround and the idiomatic alternative. Exactly the kind of
  dead-end/gotcha entry the rubric calls highest-leverage.
- **Analyst AUC-ranking entry.** The most analytically valuable entry: it records
  *why* macro-AUC was the load-bearing metric (it separates the two runs tied at
  20%) and the durable interpretive finding (collapse-onto-one-class = non-learning;
  confuses-similar-pairs = learning). This is a teaching artifact, not narration.
- **Dual-purpose feature convention entry.** A genuine convention entry: how to
  read `Image_Classification` as GT (filter `Confidence IS NULL` or by loader
  exec) vs predictions, why a `newest` selector is *not* a safe GT substitute,
  and the explicit note that the raw count (1400) is a snapshot that rots so the
  *convention* is what's durable. Textbook.

Failure-modes checked for and **not** found: no PR-number citations, no
TODO-list framing, no handoff-as-narrative, no load-bearing `[inferred from
pattern]` claims. The file links to catalog entities (chaise/`deriva://`/`/id/`
URLs) rather than inlining their contents, per its own preamble. Minor note: a
couple of the catalog links use the `https://localhost/id/...` and
`https://localhost/chaise/...` forms rather than the `deriva://catalog/.../ml/...`
resource form the preamble recommends — purely cosmetic, the RIDs resolve.

## Platform fitness

The platform largely got out of the way. The four persona findings are all
legitimate; I re-classify none of them downward to non-findings. My
cross-persona read adds one finding of my own and one observation about a missed
nuance.

**Persona findings (all confirmed real):**
- `findings/curator/labeled-test-splits-drawn-from-training-pool.md` —
  **Medium, confirmed and arguably the most valuable finding in the run.** Not a
  data bug; a genuine *expressiveness gap*: `Dataset_Type` cannot distinguish a
  canonical held-out test set (F44) from a re-split-from-train eval set (NFJ/PK6),
  and there is no parent link from NFJ/PK6 back to F3T, so the only leakage signal
  is set-intersection that no consumer runs by default. I verified the repro
  exactly. Severity is right. **Strong candidate for a GitHub issue** (platform/
  vocabulary expressiveness or dataset-lineage feature).
- `findings/modeler/hydra-description-override-grammar.md` — **Low, confirmed.**
  The QWA/R5C execution descriptions in the catalog are visibly the *sanitized*
  (paren/comma-free) form, corroborating both the failure and the workaround.
  Correctly scoped as a Hydra-passthrough ergonomic, not a deriva-ml bug.
- `findings/analyst/roc-execution-description-stale-when-assets-overridden.md` —
  **Low, confirmed verbatim.** REJ's recorded description is exactly
  `"ROC curve analysis (default: quick vs extended training) [overrides:
  assets=roc_modeler_e2e_three_way]"`, and workflow REE carries the same stale
  prose. The override suffix carries the truth; the prose lead is false. This is
  the highest-value *fix* in the run because it is a one-line config edit that
  removes an actively-misleading provenance record. **Fix inline.**
- `findings/curator/mcp-resource-read-tool-unavailable.md` — **harness-environmental,
  do not score against the platform.** `ReadMcpResourceTool` is genuinely absent
  from spawned-agent harnesses (I hit the same wall — I read concepts/getting-started
  conventions via the skill text and routed all reads through `deriva_ml_*` tools
  and Python). The Curator's fallback was correct and the catalog work was
  unaffected. The finding's own "worth confirming whether the server should also
  expose orientation as a tool/prompt" is a fair forward question, but the
  *unavailability* is a Claude Code limitation, not a deriva-ml defect.

**Skill use (inferred from artifacts).** The right skills evidently fired: the
Curator's set-algebra characterization is what `dataset-lifecycle` +
`semantic-awareness` steer toward; the Modeler's clean-tree provenance discipline
and `+experiment=` preset usage match `execution-lifecycle`; the
`capture-tacit-knowledge` discipline is visible in all five tk entries (decision
framing, no state replication). No evidence of a persona reaching for raw
`insert_entities` on lifecycle tables or otherwise bypassing the deriva-ml
surface. The one skill-adjacent gap is that the personas (correctly) could not
follow the `using-deriva-mcp` cold-start as written because the resource-read
tool is absent — a harness gap, already filed.

**Missed friction (my cross-persona read).** Filed as
`findings/evaluator/01-report-prose-metric-nits.md` (Low): two small numeric
slips in the *report* prose (calibration gap +0.08 should be +0.07; "no class
below cat 0.657" is wrong — deer 0.653 is lower). Neither is a platform defect —
they are authoring nits in a human-readable deliverable — but a cold reader who
trusts the prose over the figures would carry a slightly-wrong number. Worth a
one-line correction; not worth a GitHub issue.

## Comparison vs prior runs

Prior evaluation reports are **not present in this worktree** — they live on the
`origin/archive/e2e-test-*` branches (I can see 8 such archived runs:
2026-05-19, -26, -27 (×4 variants), -28, -30). I am deliberately **not
fabricating** a quantitative trend against reports I cannot read. Qualitative
observation only: the existence of 8 prior archived runs indicates a mature,
repeatedly-exercised test harness, and *this* run produced zero Blocker/High
findings with full catalog-parity — which, for a run whose entire purpose is to
catch the indirect channel diverging from the direct channel, is the healthy
outcome. If the user wants a real trend line, the prior `docs/reports/*-evaluation.md`
on those archive branches would need to be checked out and compared on
finding-count and severity distribution.

## Recommended actions

Organized by likely disposition — the user decides.

**Fix inline (cheap, removes an actively-misleading artifact):**
- The stale REJ/ROC-config description
  (`findings/analyst/roc-execution-description-stale-when-assets-overridden.md`).
  Lowest-touch fix is option (a) from the finding: make the `roc_analysis` config
  `description=` generic ("ROC curve analysis of the selected prediction-probability
  assets") in `src/configs/roc_analysis.py` so it stops falsely claiming a
  "quick vs extended" comparison for *every* override group.
- The two report prose nits (`findings/evaluator/01`): change "+0.08" → "+0.07"
  in §5 and fix the "no class below cat's 0.657" sentence in §4 (deer 0.653 is
  the actual per-class min). One-line edits to the analysis report.

**Promote to GitHub issue:**
- The `Dataset_Type` expressiveness / missing-train-pool-lineage gap
  (`findings/curator/labeled-test-splits-drawn-from-training-pool.md`). This is
  the one finding with platform-design weight: a real, silent leakage trap that
  the type system cannot express and the lineage does not link. Either a
  `Dataset_Type` qualifier term or explicit parent nesting from re-split eval
  sets back to their source pool. Medium severity, but high value because it
  recurs for anyone who re-splits a training pool.

**Defer / document:**
- The Hydra free-text `description=` grammar friction
  (`findings/modeler/hydra-description-override-grammar.md`). Low severity, clean
  workaround exists. Lowest-risk resolution is the doc note the finding already
  suggests (and the template CLAUDE.md already has a related Hydra-passthrough
  caveat). A dedicated `--description` non-Hydra flag would be the nicer fix but
  is more than this warrants.

**Dismiss (with reason):**
- `findings/curator/mcp-resource-read-tool-unavailable.md` as a *platform* defect
  — it is a Claude Code spawned-harness limitation (`ReadMcpResourceTool` absent),
  not a deriva-ml or MCP-server bug. The forward question it raises (expose
  orientation as a callable tool/prompt too) can be noted, but the run was
  unaffected: the `deriva_ml_*` tools carry the same conventions and the catalog
  work was complete and correct.

---

*Evaluation by the cold evaluator. Catalog read-only; no mutations performed.
All metric figures above were recomputed independently from the catalog's raw
prediction CSVs (QN6/QY8/R7A) against PK6 ground truth, not copied from the
report or the metrics CSV.*
