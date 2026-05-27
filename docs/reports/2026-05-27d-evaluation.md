# Multipersona E2E Run — 2026-05-27d Evaluation

**Catalog:** id=96, hostname=`localhost`, name=`e2e-test-20260527d`
**Branch:** `e2e-test/2026-05-27-d`
**Run shape:** first run under the four-document framework (scenario +
bootstrap + orchestrator + evaluator), Curator → Modeler → Analyst.
**Sibling versions (latest releases as of run):** deriva-ml v1.39.x,
deriva-ml-mcp v0.5.x, deriva-skills v1.x, deriva-ml-skills v1.4.x.

---

## Headline

**The team produced a coherent, end-to-end correct story, and the
platform supported them without getting in the way.** Every Modeler
and Analyst claim verified against the catalog bit-for-bit: three
training executions (XDP/XPR/XZT) with status `Uploaded` and the
exact asset triples claimed (weights / log / predictions); analysis
execution Y90 with 13 linked assets (3 inputs + 10 outputs) all
matching the reported byte counts; the Analyst's calibration finding
("XPR is the operating point, not XZT") re-derived from the prediction
CSVs **to the byte** by independent re-computation (XDP 23.5%/17.8%,
XPR 50.8%/41.8%, XZT 88.6%/77.8%).

**The most useful thing the user can act on:** the new framework is
clearly producing better tacit knowledge than the prior runs did.
Compared to the immediately prior 2026-05-27c run, the
`tacit-knowledge.md` file has:

- **Zero PR-number citations** (vs 19 in 27c, 11 in 27)
- **353 lines vs 623 lines** (43% leaner)
- **25 table rows vs 44 / 64** (substantially less state-replication)
- **Zero TODO-list-style entries** (no "Analyst should next X" framing)
- **Substantive convention entries** (the WD2 within-family-pair rule
  in tk-002, the saturated-softmax-as-overfit-signature observation
  in tk-004)

The capture-tacit-knowledge skill is clearly steering the personas
off the failure modes prior runs exhibited. The two findings I filed
are both **Low** and both about polish, not correctness.

---

## Catalog ↔ claim agreement

Verified directly against catalog 96 via `deriva-ml` Python (PathBuilder
on `Execution`, `Execution_Asset`, `Execution_Asset_Execution`,
`Dataset_Image`, `Execution_Image_Image_Classification`).

| Claim source | Claim | Verified |
|---|---|---|
| tk-001 | 1500 images, 13 datasets, all loader execs status `Uploaded` | yes |
| tk-001 | JZ8 (cifar10_complete): 150/class, perfectly balanced | yes (filtered to FZC GT rows) |
| tk-001 | WDM (small_labeled_testing): 10/class | yes (filtered to FZC GT rows) |
| tk-001 | `*_split` parents (WD2/TX0/JZJ/K0M) carry zero direct Image members | yes (0/0/0/0) |
| tk-002 | TXJ ⊂ JZT (150/150 shared) | yes |
| tk-002 | WDM ⊂ JZT (100/100 shared) | yes |
| tk-002 | K0W ⊂ JZT (500/500 shared) | yes |
| tk-002 | WDA *not* ⊂ TX8 (different stratified seed) | yes (274/400 overlap) |
| tk-003 | XDP/XPR/XZT all status `Uploaded`, all workflow XDG, all input WD2 | yes |
| tk-003 | Asset triples per execution (weights/log/preds RIDs and sizes) | yes (every RID + byte count matches) |
| tk-003 | XPR description = "Simple model run" (generic default) | yes — friction confirmed |
| tk-003 | Workflow XDG commit `4b7f48bdd368…` | yes (`4b7f48bdd36826897b28713e2c3f4bb9a86b67b7`) |
| analyst report | Y90 has 13 linked assets, all `Uploaded` | yes |
| analyst report | Per-asset byte counts (YB4/YB6/YB8/YBA/YBC/YBE/YBG/YBJ/YCT/YCW) | yes (all 13 match) |
| analyst report | XDP 24%/XPR 38%/XZT 41% top-1 acc | yes (re-derived independently) |
| analyst report | Confidence calibration table (23.5/17.8, 50.8/41.8, 88.6/77.8) | yes (re-derived to the byte) |
| analyst report | `roc_metrics.csv` values | yes (matches `analysis-scratch/y90_outputs/roc_metrics.csv` exactly) |

**No discrepancies found between any claim and the catalog.** The
historically-highest-risk thread (§3.1 of the rubric) is clean.

One subtle observation that's not a discrepancy but is worth recording:
tk-001's claim "no duplicate labels (no need for `newest` selector)"
holds **only when filtered to loader execution FZC**. After the Modeler
ran, the feature table holds 1800 rows / 1500 distinct images — because
`Image_Classification` is the *same* feature that
`record_test_predictions` writes into, with `Confidence IS NULL`
distinguishing GT rows from prediction rows. The Analyst's rank script
filters correctly (`gt[gt["Confidence"].isna()]`), so the analysis was
sound — but a reader of tk-001 who took the no-duplicate-labels claim
at face value would be off. Filed as **evaluator/01** (Low / Skill
issue).

---

## Coherence of the team's deliverables

**Strong, all the way through.** Curator → Modeler → Analyst reads as a
continuous story even to a cold reader.

**Curator (tk-001, tk-002).** The audit characterised what the
downstream personas would actually need: class balance, GT
completeness, leakage map. The leakage map (tk-002) is the
high-leverage entry — it documents which dataset pairs are safe to
train/test on and which silently leak training data into evaluation.
The Modeler consumed it directly: the choice of WD2 for all three runs
("within-family pair, safe per tk-002") is the right call and is
called out explicitly in tk-003. **No curated dataset variants were
created in this run** — the Curator concluded the existing 13 datasets
were sufficient. That's a legitimate outcome per the scenario §2.1
("the exploration itself is the work"); it does mean this run didn't
exercise the Validation-lane path the way 27c did, but that's a
coverage gap, not a finding.

**Modeler (tk-003).** Three differentiated training runs on a single
held-out test partition — exactly what the Analyst needs for a
controlled comparison. The rationale for "WD2 for all three, vary
only model_config" is explicit ("controlled experiment the catalog
supports"). The asset RIDs are committed straight into
`src/configs/assets.py` (`roc_quick_vs_extended` → `["XFM","XRP","Y1R"]`)
so the Analyst's `deriva-ml-run-notebook` invocation Just Works. The
overfit anticipation ("XZT shows textbook overfit signature, training/
test loss diverging hard around epoch 15") is the kind of forward
signal that pays off in the Analyst arc.

**Analyst (tk-004, `docs/reports/2026-05-27d-analyst-report.md`).** The
report is **structured for a non-ML reader** (TL;DR + sections on
inputs, verification, ranking, confusion matrices, calibration, recs).
The headline finding ("XPR is the operating point, not XZT — XZT's 41%
top-1 is bought with saturated softmax confidence") is exactly the
flavour of insight the scenario §2.3 asks for ("understand what the
models are doing — what they got right, what they got wrong, where the
confusion lies"). The calibration table is a non-trivial substantive
contribution beyond "rank by accuracy": it inverts the recommendation
relative to what `roc_metrics.csv` alone would suggest, and the report
flags that inversion clearly ("the numbers alone are misleading; the
story is in the calibration").

Cross-channel verification is exemplary: the Analyst re-derived all
emission-time numbers from scratch via direct deriva-ml + the
prediction CSV (`analysis-scratch/rank_runs.py`), and confirmed the 13
Y90 asset rows via PathBuilder (`analysis-scratch/verify_y90.py`). Both
scripts are re-runnable, both are committed, and I re-ran the accuracy
+ calibration re-derivation independently and got identical results.

**Handoff chain holds.** tk-003's "for the Analyst" section flows
directly into tk-004's analysis; tk-001's class-balance verification
flows into tk-004's "can read accuracy straight from prediction-vs-GT
join without weighting"; tk-002's leakage map shows up in the report's
§7 recommendations ("the Modeler's choice of WD2 was correct for this
exercise — within-family pair").

A fresh contributor handed only the worktree and the catalog could
reconstruct what the team did and why. That's the §3.2 test passing.

---

## tacit-knowledge.md quality

Read top to bottom as a fresh contributor would. **This is the best
tacit-knowledge.md I've evaluated across the run history.**

### What works

- **Each entry captures rationale, not state.** tk-001 says "no
  rebalancing or filtering work to do up front" (the implication for
  the Modeler) instead of just listing per-class counts as a table.
  tk-002's leakage map is conceptually *rules* ("WDA→WDM safe;
  JZT→WDM unsafe") even though it uses a table to express them —
  borderline state-y but functional.
- **Dead-end / forward-signal entries.** tk-003's "XZT is *not*
  automatically the best choice" is exactly the kind of forward signal
  the next persona needs. tk-004's "the probability outputs of XZT
  are not usable as confidence signals downstream" is the kind of
  load-bearing conclusion that's hard to recover from the catalog
  alone.
- **Convention entries.** tk-003's "to compose a run with
  `model_config=` + `datasets=` overrides without `+experiment=`, you
  lose the description handle — pass `description='…'`" is exactly
  what the skill calls "convention entries: especially valuable when
  the convention isn't otherwise documented."
- **Supported-by chain is intact and used.** tk-002 → tk-001;
  tk-003 → tk-001 + tk-002; tk-004 → tk-001 + tk-002 + tk-003. The
  chain expresses real dependency, not formal padding.
- **Zero PR-number citations.** The historically common
  transient-coordinate failure mode (e.g. "PR #246 fixed the row-loss
  bug") is completely absent. The closest thing to a citation is
  tk-003's explicit workflow-commit reference (`4b7f48bdd368…`), which
  is durable provenance, not a transient PR coordinate.
- **No TODO-list framing.** tk-003 and tk-004 both have "For the
  [next persona]" sections, but they're brief footers naming the
  inputs and links — they don't try to *direct* what the next persona
  does (the scenario explicitly says personas should drive their own
  work).

### Minor weaknesses

- **tk-001's "no duplicate labels" claim is a snapshot fact framed as
  a convention.** It's true at audit-time and false after the Modeler
  runs. The convention worth recording — "GT and predictions share a
  feature table; filter by `Confidence IS NULL` or by execution to
  separate them" — isn't there. Filed as **evaluator/01**.
- **tk-003's results table is borderline state-replication.** The
  train-acc / test-acc / weights-size columns ARE recoverable from
  the catalog (training logs are committed assets). What's *not*
  recoverable and IS worth recording is the asset-RID handoff
  (XFG/XFJ/XFM etc.), the rationale ("variation took"), and the
  forward signal ("XZT shows overfit signature"). The table mixes
  both shapes. Not severe enough to file; calling out for the next
  iteration.

### Failure-mode comparison vs prior runs

| Failure mode | 2026-05-27 (catalog 93) | 2026-05-27c (catalog 95) | **2026-05-27d (this run)** |
|---|---|---|---|
| PR-number citations | 11 | 19 | **0** |
| State-replication tables | many | many | a few, narrower |
| TODO-list framing | present | reduced | **absent** |
| Handoff-as-narrative blocks | yes | reduced | brief footers only |
| Lines of tacit-knowledge.md | comparable | 623 | **353** |
| Load-bearing `[inferred from pattern]` claims | unknown | unknown | **none found** |

**This is the convergence the new four-document framework was meant to
produce.** Pulling the evaluator out into its own rubric, and tightening
the `capture-tacit-knowledge` skill against snapshot-replication, did
work.

---

## Platform fitness

The platform supported the team's work cleanly. No persona filed any
findings in-arc (the `findings/analyst/` directory exists but is
empty). The friction surfaced is all **documented, low-severity, and
cosmetic.**

### Persona-filed friction (in tk-003 and tk-004, "Friction noted, not fixed")

- **`description=` lost when overriding `model_config=` + `datasets=`
  without `+experiment=`.** Modeler in tk-003. Workaround documented.
  Not severe (provenance still in the training log).
- **`deriva-ml-run-notebook nb.ipynb roc_analysis` (positional trailing
  arg) raises Hydra `missing EQUAL at '<EOF>'`.** Both Modeler and
  Analyst hit this. The error message technically correct but doesn't
  point at the actual cause. Documented in CLAUDE.md.
- **Upload-cache mirrors only the second bag.** Analyst noted this in
  tk-004. Not a bug, but a documentation gap — local cache ≠ catalog
  state for analyses that emit two upload bags. Worked around by the
  Analyst's `fetch_outputs.py`.

### Friction not flagged but visible in artifacts

- **InsecureRequestWarning floods the executed notebook output (1152
  warning lines).** Modeler tk-003 calls it "cosmetic"; what's not
  obvious until you open the catalog-stored `roc_analysis.md` (YCW)
  is that ~50% of the 2300-line markdown export is HTTPS-warning
  chatter. This degrades the catalog as an archival store. Filed as
  **evaluator/02** (Low / Polish). Two fixes proposed.

### Skill use

The right skills fired:

- **Curator:** consulted MCP resources + direct deriva-ml for the
  audit; used `capture-tacit-knowledge` for tk-001/tk-002. No reaching
  for raw catalog-state queries where a resource would have served.
- **Modeler:** used `execution-lifecycle` + the `deriva-ml-run` CLI
  for training; followed `execution-lifecycle`'s explicit "offer to
  wire output asset RIDs into `assets.py`" pattern (PR #86 from the
  task history) — `src/configs/assets.py` was correctly updated with
  `roc_quick_vs_extended` + per-weight stores, with `with_description`
  + per-asset rationale. This is the wired-output-handoff pattern
  working end-to-end.
- **Analyst:** used `deriva-ml-run-notebook` for provenance-tracked
  notebook execution; used direct deriva-ml PathBuilder for
  cross-channel verification; used `capture-tacit-knowledge` for
  tk-004.

No "wrong tool" reaches I noticed.

### What the platform did NOT need from the team

The denormalize re-entry / row-completeness checks that the 2026-05-27
and 2026-05-27c runs exercised heavily — the headline value of PR #246
and the SchemaPathWalker work — were not directly exercised this run
because no one needed to denormalize a large dataset. This is a coverage
gap, not a regression. Worth noting for run planning: if the Analyst
chooses not to materialise a join (e.g. when the prediction CSV already
carries `Image_RID`), the PagedFetcher / denormalize path doesn't get
tested.

---

## Comparison vs prior runs

| Run | Status | Findings | Severity profile | Notable |
|---|---|---|---|---|
| 2026-05-21 | Partial | 19 | A01/A02/A04 surfaced denormalize | Pre-PR-#189 era |
| 2026-05-26 | Clean | 6 | Low/Med | UX polish |
| 2026-05-27a | Blocked | 1 | Blocker (PagedFetcher row-loss) | Drove PR #246 |
| 2026-05-27b | Clean | 3 | 1 Med + 2 Low | Post-#246 validation |
| 2026-05-27c | Clean | 1 | Low (refactor regression) | Validated #246 + #37 + #38/#59 |
| **2026-05-27d (this)** | **Clean** | **2 (both Low)** | **both filed by evaluator, both polish** | **First run under new 4-doc framework** |

**Trend:** runs are converging on "boring." The platform is no longer
the bottleneck; what surfaces is documentation-and-polish friction. The
2026-05-27d findings are both Low and both from the evaluator (no
persona-filed findings at all). The previous run (27c) had a
model-template refactor regression as its only finding. The trend line
is: **issues are migrating from platform to template to documentation,
and from blockers to polish.**

A separate axis: **tacit-knowledge.md quality.** This is the first run
where I'd hand `tacit-knowledge.md` to a new contributor and expect
them to find it useful rather than confusing. The PR-number citation
count dropping from 19 → 0 is the headline metric for the new framework.

---

## Recommended actions

| Action | Disposition | Why |
|---|---|---|
| **evaluator/01** (tk-001 ageing) | **Defer / skill refinement** | The Analyst caught it; the framing fix is a 1-line tweak to the `capture-tacit-knowledge` skill's convention-vs-snapshot example. Low priority but cheap. |
| **evaluator/02** (InsecureRequestWarning flood) | **Fix inline** (deriva-py) or **GitHub issue** | Surfaced across multiple runs; degrades the catalog-stored markdown exports. Trivial change at the deriva-py layer (`urllib3` warning filter when `verify=False` is set deliberately). |
| Modeler's `description=` friction (tk-003) | **Defer** | Workaround documented; not severe enough to chase as platform work right now. |
| `deriva-ml-run-notebook nb.ipynb roc_analysis` confusing error | **Defer / improve Hydra-parser error mapping** | Hits twice in two arcs; would be nice to translate "missing EQUAL" → "unexpected positional argument". Low priority. |
| `capture-tacit-knowledge` skill enhancement (snapshot vs convention) | **Skill iteration** | The convention-vs-snapshot distinction is the natural next refinement of the skill, building on the success of removing PR-number citations + TODO-list framing this run. |
| Validation-lane coverage gap | **Note for next run planning** | This run did not exercise the Validation-bag dispatch path (no Curator-created Validation slice). Next run's Curator prompt could nudge toward "consider adding a Validation slice" without prescribing it. |
| Denormalize coverage gap | **Note for next run planning** | The Analyst chose not to materialise a join. If we want to keep PR #246 / SchemaPathWalker exercised, the Analyst persona prompt could mention that scenario as an option. |

---

## Reproducibility

- **Catalog id:** 96 (preserve; query for archeology; delete at next
  bootstrap)
- **Branch:** `e2e-test/2026-05-27-d` (5 [E2E-DROP] commits ahead of
  main)
- **Persona execution RIDs:**
  - Loader (Phase 0): 46Y, FZC, JY8, TW0
  - Modeler: XDP, XPR, XZT (workflow XDG, commit `4b7f48bd…`)
  - Analyst: Y90 (workflow Y8W, commit `5816bca7…`)
- **Output assets (verified):**
  - Modeler triples: XFG/XFJ/XFM (XDP), XRJ/XRM/XRP (XPR), Y1M/Y1P/Y1R (XZT)
  - Analyst outputs: YB4, YB6, YB8, YBA, YBC, YBE, YBG, YBJ, YCT, YCW
- **Local copies for offline review:**
  `analysis-scratch/y90_outputs/` (all 10 Y90 outputs)
- **Re-runnable scripts:** `analysis-scratch/rank_runs.py`,
  `analysis-scratch/verify_y90.py`, `analysis-scratch/fetch_outputs.py`
- **Independent re-derivation result (this evaluator):** XDP 24.00% /
  conf 23.5%/17.8%; XPR 38.00% / 50.8%/41.8%; XZT 41.00% / 88.6%/77.8%.
  **Matches the Analyst's reported numbers exactly.**

---

🤖 Generated by the Evaluator persona (first run under the
four-document framework: scenario + bootstrap + orchestrator +
evaluator).
