# Multipersona E2E Run — 2026-05-27e Evaluation

**Catalog:** id=2, hostname=`localhost`, name=`e2e-test-20260527e`
**Branch:** `e2e-test/2026-05-27-e`
**Run shape:** second run under the four-document framework (scenario +
bootstrap + orchestrator + evaluator). First run to exercise the new
phase-3 wide-table deliverable (scenario PR #45) and the new
tk-entry conventions (`<a id="tk-NNN">` anchors and `ml.cite(rid)`
click-through links).
**Sibling versions (as exercised):** deriva-ml at commit
`08cd2561` (model-template main) + PR #248 (auto-derive notebook
config name, merged today into deriva-ml).

---

## Headline

**The team produced a coherent, end-to-end correct story, the catalog
holds up under inspection, and every headline number in the Analyst's
report re-derives bit-for-bit from the catalog.** The new phase-3
framing landed cleanly: the joined wide table
([`findings/analyst/wide_joined_K16.csv`](../../findings/analyst/wide_joined_K16.csv),
500 rows × 35 cols) is a real durable deliverable, and the standalone
derivation script
([`findings/analyst/rank_and_join.py`](../../findings/analyst/rank_and_join.py))
ran clean on the first try and produced the report's headline numbers
to all printed digits.

**The most important thing the user can act on:** the Analyst caught a
genuine regression in code that just shipped this morning. PR #248
(deriva-ml `feat(run_notebook): make config_name optional, auto-derive
from notebook filename`, commit `6ed68d08`) introduced a quality-of-life
feature whose default invocation path does not work under
`deriva-ml-run-notebook` — the only headless runner the project ships.
The Analyst routed around it by passing `config_name` explicitly, but
the workaround pollutes every notebook in the repo with a redundant
argument PR #248 was specifically designed to eliminate. I've upgraded
this from `analyst/01` (Analyst's in-arc finding) to
[`evaluator/01`](../../findings/evaluator/01-run-notebook-config-derivation-regression-confirmed.md)
as **High / Bug**. The fix is one line.

---

## Catalog ↔ claim agreement

Verified directly against catalog 2 via `deriva-ml` Python (PathBuilder
on `Execution`, `Execution_Asset`, `Execution_Asset_Execution`,
`Dataset_Image`, `Execution_Image_Image_Classification`,
`Workflow`).

| Claim source | Claim | Verified |
|---|---|---|
| tk-001 | 1500 Image_Classification GT rows, all from `Execution=FZC`, all `Confidence IS NULL` | yes (1500/1500) |
| tk-001 | 10 classes, 150 images each on `JZ8` (whole catalog) | yes |
| tk-001 | `JZT ∪ K04 = JZ8`, zero overlap, 750/750 partitions | yes (JZT=750, K04=750, JZ8=1500) |
| tk-001 | `K0W=500` Training (Family A), `K16=500` Testing (Family A) | yes |
| tk-002 | Family A test partition `K16 ⊆ K04` (real held-out) | yes (member-set check) |
| tk-004 | XZP/Z1R/103T all status `Uploaded`, all share workflow XDG | yes (all three on workflow XDG, commit `4b7f48bdd368…`) |
| tk-004 | Asset triple per execution: weights / log / predictions CSV | yes — XZP→(Y1G/Y1J/Y1M); Z1R→(Z3J/Z3M/Z3P); 103T→(105M/105P/105R) with sizes 6.55MB / 6.55MB / 26.12MB on the weights respectively |
| tk-004 | `description="Simple model run"` on Z1R and 103T (the friction noted at the end of tk-004) | yes — direct query confirms |
| tk-005 | Analysis execution 11AY status `Uploaded`, 13 linked assets (3 inputs Y1M/Z3P/105R + 10 outputs) | yes — bit-for-bit |
| analyst report §2 | Top-1 / Micro-AUC / Macro-AUC table | yes — re-derived independently via `rank_and_join.py`; matches to all printed digits |
| analyst report §3 | Per-class AUC table (10 classes × 3 models) | yes — every cell matches catalog-resident `roc_metrics.csv` (asset 118J) |
| analyst report §4 | 46 (9.2%) all-three correct; 96 (19.2%) all-three same; **207 (41.4%) all-three wrong** | yes — re-derived from wide table to the integer |
| analyst report §7 | Catalog-resident `roc_metrics.csv` (118J) matches `findings/analyst/roc_metrics_from_catalog_11AY.csv` | yes — `DataFrame.equals()` returns True |
| Image_Classification feature shape post-Modeler-arc | 3200 rows total: 1500 GT (FZC) + 1700 prediction rows across 5 training execs (XZP/Z1R/103T + XDP/XPR Family-B smoke) | yes — exact match |

**No discrepancies between any claim and the catalog.** The
historically-highest-risk thread (§3.1 of the rubric) is clean for
the second run in a row.

One observation worth noting (not a discrepancy): the `tacit-knowledge.md`
in-text URLs are snapshot-pinned **for Datasets only** (`@355-KW8K-DXSC`),
not for Executions or Assets. The skill says all RID citations should
be snapshot-pinned (default behavior of `ml.cite(rid)`), and `ml.cite()`
on this catalog does return snapshot-pinned URLs for every RID type
(checked directly: `JZ8`, `FZC`, `XZP`, `103T`, `Y1M`, `118J`, `11AY`
all return `@355-KZ0W-A4H4`). The team hand-wrote the URLs rather than
calling `ml.cite()` — see "Skill adoption" below.

### Cross-channel re-derivation

I re-ran `findings/analyst/rank_and_join.py` end-to-end and
independently re-derived the cross-model-agreement integers via pandas
on the wide table. Numbers matched exactly:

| Number | Report claim | Evaluator re-derived | Catalog-resident (118J) |
|---|---|---|---|
| XZP Top-1 | 25.2% | 25.20% | 25.20% |
| Z1R Top-1 | 36.0% | 36.00% | 36.00% |
| 103T Top-1 | 36.8% | 36.80% | 36.80% |
| XZP Micro-AUC | 0.722 | 0.7225 | 0.722522 |
| Z1R Micro-AUC | 0.795 | 0.7951 | 0.795121 |
| 103T Micro-AUC | 0.817 | 0.8172 | 0.817158 |
| 103T `airplane` AUC (key claim: regresses vs default) | 0.816 | 0.815600 | 0.815600 |
| Z1R `airplane` AUC (key claim: best airplane) | 0.847 | 0.847378 | 0.847378 |
| All-three correct (K16) | 46 / 9.2% | 46 / 9.2% | n/a (wide-table derived) |
| All-three wrong (K16) | **207 / 41.4%** | 207 / 41.4% | n/a |
| Same prediction across triplet | 96 / 19.2% | 96 / 19.2% | n/a |

---

## Coherence of the team's deliverables

**Strong, all the way through.** Curator → Modeler → Analyst reads as a
continuous story even to a cold reader. Each persona produced exactly
what the next needed, and the new phase-3 framing — wide joined table
as a first-class deliverable — landed cleanly.

**Curator (tk-001, tk-002, tk-003).** Three substrate entries. tk-001
characterised the GT layer's invariants (1500 rows, 10 classes ×
150 each, `Confidence IS NULL` on every loader-execution row).
tk-002 is the high-leverage entry: it documents that the two
small-variant Split families (K0M Family A; WD2/TX0 Family B) sample
test partitions from different sides of the canonical Toronto split,
and only Family A has a genuinely held-out test partition. tk-003
spells out the `Image_Classification` dual-purpose convention (loader
writes GT with `Confidence IS NULL`; training executions write
predictions with `Confidence` populated). The Modeler consumed all
three directly: the choice of K0M for the triplet was explicitly
driven by tk-002, and the Analyst's GT filter (`Execution=FZC AND
Confidence IS NULL`) was lifted verbatim from tk-003.

No curated dataset variants created. Same coverage gap as 27d — the
Validation-bag dispatch path is not exercised when the substrate is
already balanced and the Modeler doesn't ask for it.

**Modeler (tk-004).** Three differentiated training runs on a single
genuinely-held-out test partition (K0M / K16), varying capacity (3-ep
quick → 10-ep default → 20-ep large). This is exactly the controlled
experiment the Analyst needs for a clean ROC comparison.
Asset RIDs wired straight into `src/configs/assets.py` as
`modeler_familyA_triplet` (Y1M/Z3P/105R), with per-model weights
registered individually (Y1G / Z3J / 105M) under `modeler_*_weights`
group names. Descriptions are rich and rationale-bearing:

```python
asset_store(
    with_description(
        ["Y1M", "Z3P", "105R"],
        "Modeler arc 2026-05-27-e: three prediction CSVs on Family A K16 "
        "test partition. Y1M=cifar10_quick/3ep (XZP, 25.20%), ..."
    ),
    name="modeler_familyA_triplet",
)
```

The "load-bearing tk-003" reminder in tk-004 (predict-row count now
exceeds GT-row count, filter by execution RID before joining) is the
kind of forward-signal entry that pays off — and the Analyst's script
does exactly the right filter on the very next read.

The Family-B smoke run on WD2 (XDP, 24.00% test_acc) is left in the
catalog as evidence the experiment-preset path works, but
explicitly excluded from the triplet to keep the comparison clean.
That's a defensible curatorial decision recorded in the entry, not a
clutter.

**Analyst (tk-005, `docs/reports/2026-05-27-e-analysis.md`).** The
report is well-structured for a non-ML reader (TL;DR table, ranking,
per-class behaviour, confusion patterns, overfitting interpretation,
recommendations). Three substantive analytical conclusions worth
naming because they're stronger than "rank by accuracy":

1. **AUC discriminates better than Top-1.** 0.795 vs 0.817 is a
   bigger relative gap than 36.0% vs 36.8%, so a future analyst who
   reports Top-1 alone will under-call what capacity is buying.
2. **`airplane` regresses with capacity** (0.847 default → 0.816
   large), against the general trend. The interpretation
   ("capacity-vs-data signal — large model has room to overfit to
   sky/water/land background features that don't transfer K0W→K16")
   is the kind of insight that's not in the catalog and is exactly
   what the Analyst owes the reader.
3. **41.4% of K16 is missed by all three models simultaneously** —
   the *shared difficulty* of K16, not a model-capacity issue. This
   correctly frames the next modelling lever as augmentation /
   representation, not more parameters in the same architecture.

Cross-channel verification is exemplary: the Analyst re-derived all
numbers via `rank_and_join.py` (independent of the catalog-resident
`roc_metrics.csv` produced by execution 11AY), and the two agree to
all printed digits. I re-ran the script myself and got the same
result.

**The new wide-table deliverable works as intended.** The 500 × 35
table is a real durable artifact — Image_RID, True_Class, plus each
of three models' predicted class and all 10 per-class probabilities.
Every numerical claim in the report can be re-derived from it without
re-querying the catalog. This is the right shape for a phase-3
deliverable: it survives a catalog disappearing, it survives a
notebook re-execution, and it serves as the source of record for the
report's tables.

**Handoff chain holds.** tk-002 → tk-004 (Family A choice) → tk-005
(K16 substrate). tk-003 → tk-004 (predict-vs-GT filter discipline) →
tk-005 (GT filter on Image_Classification feature). The
`**Supported by:**` lines actually express dependency — they're not
formal padding.

A fresh contributor handed only the worktree and the catalog could
reconstruct what the team did and why. The §3.2 test passes.

---

## tacit-knowledge.md quality

Read top to bottom. **The file holds up as a record a fresh contributor
would actually find useful.** Both new tk-entry conventions (anchors
and click-through RID links) are visibly adopted. Density is up
modestly vs the 27d baseline but driven by genuinely longer reasoning
(tk-002's family-distinction logic, tk-005's per-class interpretation),
not by state-replication padding.

### Measured against the 27d baseline

| Metric | 27c | **27d** | **27e (this run)** | Trend |
|---|---|---|---|---|
| Lines | 623 | 353 | **391** | +38 vs 27d (10% growth) |
| PR-number citations | 19 | 0 | **1** (PR #46 — about Hydra description auto-composition, contextual not load-bearing) | mild regression vs 27d clean, still 19× better than 27c |
| TODO-list framing | reduced | absent | **absent** | ✓ |
| Handoff-as-narrative | reduced | brief footers | **integrated into entries** (no separate "handoff" footers — see "Implications for collaborators" sections instead) | ✓ |
| State-replication tables | several | a few, narrow | **two** (tk-002 family table, tk-004 triplet results table) — narrow and rationale-bearing | ✓ |
| Load-bearing `[inferred from pattern]` claims | unknown | none | **none found** | ✓ |
| `<a id="tk-NNN">` anchors adopted | n/a | yes | **yes** (5/5 entries) | ✓ |
| Click-through RID links adopted | n/a | partial | **partial** — Datasets are snapshot-pinned (`@355-KW8K-DXSC`); Executions and Assets are not | mild gap (see below) |

The single PR-number citation is in tk-004:

> The auto-composed `Execution.description` from PR #46 covers the
> `+experiment=` path (see XZP description above), but bare
> `model_config=/datasets=` overrides default to "Simple model run"
> (see Z1R, 103T).

This is borderline. It cites a PR rather than naming the durable
behaviour ("auto-composed descriptions only fire for the `+experiment=`
override path"). I don't think it's actively harmful — the PR
reference adds traceability if the behaviour ever needs to be looked
up — but it would be cleaner without. Worth noting as a minor regression
vs 27d's perfect 0, not severe enough to file.

### What works (strong patterns continuing from 27d)

- **Convention entries.** tk-003 (`Image_Classification` is
  dual-purpose) is exactly the highest-leverage shape: it names a
  catalog invariant a naive reader would miss, gives the durable
  filter (`Execution=FZC AND Confidence IS NULL`), and explains why
  the convention exists (provenance via single feature table).
- **Genuine forward signals.** tk-004's "tk-003 is now load-bearing —
  Image_Classification holds 1500 GT + 1700 predictions; filter by
  execution before joining" is the kind of reminder a next-Modeler
  doesn't have time to derive themselves. tk-005's
  "AUC-vs-Top-1 discriminates better" is forward signal for the next
  Analyst.
- **Weighed alternatives sections.** Every tk-entry has a
  rationale-bearing "Weighed alternatives" block — tk-002 considered
  curating a new "really held-out" labeled small split and declined
  with reason; tk-004 considered JZJ-scale, Family-B mixing, and
  seed variation and declined each with reason; tk-005 considered
  re-running on JZJ for absolute numbers, calibration plots, and
  macro-AUC primary and declined each with reason. This is the kind
  of decision record the file is for.
- **Supported-by chain is intact.** tk-005 → tk-004 → tk-003/tk-002 →
  tk-001. The chain expresses real dependency: each downstream entry
  consumes a *specific* invariant established upstream.

### What's worth refining (minor / skill-adoption)

- **Snapshot-pinned URLs are inconsistent.** Datasets get `@SNAPSHOT`
  suffixes (`https://localhost/id/2/JZ8@355-KW8K-DXSC`); Executions
  and Assets don't (`https://localhost/id/2/XZP`,
  `https://localhost/id/2/118J`). The skill's convention is "call
  `ml.cite(rid)`; default returns snapshot-pinned for all RID types,"
  and I verified `ml.cite()` against this catalog returns the same
  snapshot suffix for Executions, Assets, AND Datasets
  (`@355-KZ0W-A4H4`). The team appears to have hand-written the URLs
  for Executions/Assets rather than routing through `ml.cite()`, so
  the convention is adopted partially. Not severe — in practice,
  Executions and Assets are append-only and don't need
  snapshot-pinning for correctness, only for convention consistency.
  Worth a one-line nudge in the skill: "for consistency, use
  `ml.cite(rid)` for every RID type, not just Datasets."

- **tk-004's results table.** Five-column results table (Execution /
  Model config / Epochs / Channels / Test acc / Predictions CSV /
  Weights). The training-time test accuracy and the asset RIDs are
  recoverable from the catalog (training log is a committed asset);
  the rationale ("triplet exists to vary capacity on identical data")
  and forward signal ("103T shows textbook overfitting") are NOT
  recoverable and are the durable content. The table is borderline
  state-replication — narrow enough that it's defensible (the table
  is the *handle* a reader uses to find the components), but worth
  flagging that "Test acc 25.20%/36.00%/36.80%" is exactly the same
  number the Analyst then re-derives in §2 of the report. If those
  numbers diverged between catalog and tk-entry, the catalog would
  win — which is the rule. This isn't severe enough to file but is
  the next refinement of the skill's "convention vs snapshot"
  guidance.

### Failure-mode comparison vs prior runs

| Failure mode | 27c | 27d | **27e (this run)** |
|---|---|---|---|
| PR-number citations | 19 | 0 | **1** (contextual) |
| State-replication tables | many | a few | **two narrow** |
| TODO-list framing | reduced | absent | **absent** |
| Handoff-as-narrative blocks | reduced | brief | **integrated** |
| Lines of tacit-knowledge.md | 623 | 353 | **391** |
| Load-bearing `[inferred from pattern]` claims | ? | none | **none** |
| New conventions: anchors adopted | n/a | yes | **yes** |
| New conventions: click-through RIDs adopted | n/a | partial | **partial (same gap)** |

The new framework's wins are holding. The slight uptick in line count
is from longer reasoning per entry (which is good), not state padding.
The single PR-number reference is a regression vs 27d's perfect zero,
but the context — naming PR #46 to give traceability for a Hydra
description-auto-composition behaviour — is one of the few cases
where a PR number arguably adds value over its absence. The
click-through-RID adoption gap (Datasets snapshot-pinned, Executions
not) is unchanged from 27d and is a skill-iteration opportunity.

---

## Platform fitness

The platform did its job. The team produced the deliverables they
needed without routing around any blockers. **One real regression
surfaced** in code that shipped today; one persona-filed finding
documents the workaround. No other friction was load-bearing.

### Persona-filed friction

- **`analyst/01` — `run_notebook()` config-name auto-derivation
  fails under `deriva-ml-run-notebook`.** Confirmed and upgraded to
  [`evaluator/01`](../../findings/evaluator/01-run-notebook-config-derivation-regression-confirmed.md)
  (High / Bug). Real regression in PR #248 (deriva-ml commit
  `6ed68d08`, merged today). The auto-derive feature added to make
  `config_name=` optional silently fails under the only headless
  runner the project ships, because `pm.execute_notebook()` (unlike
  papermill's CLI) does not set `PAPERMILL_INPUT_PATH`. One-line
  fix: have `_derive_config_name_from_notebook` also consult
  `os.environ.get("DERIVA_ML_NOTEBOOK_PATH")`, which
  `run_notebook.py:618` already sets unconditionally.

### Friction in the team's writeups, not separately filed

- **`description=` is not a free-form Hydra override
  for `model_config=/datasets=` combinations.** Modeler tk-004
  documents this. The auto-composed `Execution.description` from
  PR #46 covers the `+experiment=` path; bare `model_config=` /
  `datasets=` overrides fall back to "Simple model run." Workaround
  is to define a small experiment preset rather than passing
  `description=` directly. This is the same friction noted in 27d's
  tk-003. **Documented, not blocking.**

### Friction NOT flagged but visible in artifacts

- **InsecureRequestWarning flooding persists.** Same as 27d. Catalog-resident
  `roc_analysis.md` (asset 11ET, 1.08MB) contains the executed
  notebook's full markdown export, which includes papermill's stdout —
  and that stdout is dominated by `InsecureRequestWarning` lines from
  the kernel-side `urllib3` warning filter. I filed this in 27d as
  evaluator/02. It's still there. No need to re-file in this run;
  the existing finding is still actionable.

### Skill adoption observed

The right skills fired:

- **Curator:** consulted MCP resources + direct deriva-ml Python for
  the substrate audit. Three tk-entries written, each with explicit
  `**Supported by:**` chaining where appropriate. No creation of
  unnecessary curatorial artifacts — the Curator correctly judged the
  substrate sufficient as-shipped.
- **Modeler:** `execution-lifecycle` skill drove the runs; wired the
  output asset RIDs into `src/configs/assets.py` with
  `with_description` and per-asset rationale strings. This is the
  proactive-offer pattern from execution-lifecycle (task `#86` in
  the historical log) firing correctly.
- **Analyst:** `deriva-ml-run-notebook` for provenance-tracked
  execution (worked around the PR #248 regression). Direct deriva-ml
  PathBuilder for cross-channel verification. `capture-tacit-knowledge`
  for tk-005. Used `feature_values()` + the prediction CSVs rather
  than reaching for `denormalize_dataset()` — defensibly, since the
  prediction CSVs already carried `Image_RID` and the join was
  straightforward in pandas.

**One mild skill-adoption gap.** The team hand-wrote the
`https://localhost/id/2/<RID>` URLs instead of calling
`ml.cite(<rid>)`. Datasets got snapshot suffixes (probably copied
from earlier-run examples that snapshot-pinned them); Executions and
Assets did not. `ml.cite()` is the documented entry point and returns
snapshot-pinned URLs for every RID type. This isn't severe — the
team's URLs do resolve in chaise — but the skill convention isn't
fully met. See "tacit-knowledge.md quality" above.

### Coverage gaps (not findings)

- **Validation-lane dispatch is still unexercised.** Same as 27d. The
  Curator concluded the substrate didn't need a Validation slice;
  the Modeler didn't request one. The `cifar10_cnn` runner's
  Validation-bag consumption path doesn't get tested when no one
  needs it. Note for next run planning, not a finding.
- **Denormalize path not directly exercised.** Same as 27d. The
  Analyst's join used `feature_values()` + prediction CSVs in
  pandas; `denormalize_dataset()` is not called anywhere in this
  arc. PR #246 (PagedFetcher row-completeness invariant) doesn't get
  hit. Note for next run planning.

---

## Comparison vs prior runs

| Run | Status | Findings | Severity profile | Notable |
|---|---|---|---|---|
| 2026-05-27 | Partial | many | mixed | Pre-framework |
| 2026-05-27b | Clean | 3 | 1 Med + 2 Low | Post-#246 validation |
| 2026-05-27c | Clean | 1 | Low | Validated #246 + #37 + #38/#59 |
| 2026-05-27d | Clean | 2 | Both Low | First run under new 4-doc framework |
| **2026-05-27e (this)** | **Clean** | **1 (High, in code that shipped today)** | **1 High** | First run under new 4-doc framework with new tk-conventions + new phase-3 framing |

**Trend:** Catalog ↔ claim agreement is now boringly clean for the
third run in a row. The platform itself isn't producing the issues
anymore — they come from new code shipping. In this run, the *only*
finding is in a feature that merged this morning, caught the same day
by the e2e run that's specifically supposed to exercise the
end-to-end ergonomic. **That's the framework working as intended.**

On `tacit-knowledge.md` quality: holding steady at the post-27d
baseline. The new anchor + click-through conventions adopted, with
one mild gap on snapshot-pinning Executions/Assets via `ml.cite()`.
One PR-number citation slipped back in vs 27d's perfect zero — minor
regression, contextually defensible, worth a one-line skill nudge.

On the new phase-3 deliverable: **the wide joined table works.** It's
a real durable artifact, the standalone derivation script is
re-runnable from a clean checkout, and the report can be rebuilt from
it without re-querying the catalog. The scenario PR #45 framing
landed cleanly on its first exercise.

---

## Recommended actions

| Action | Disposition | Why |
|---|---|---|
| **[`evaluator/01`](../../findings/evaluator/01-run-notebook-config-derivation-regression-confirmed.md)** — `run_notebook()` auto-derive regression under headless papermill | **Fix inline now** | One-line fix in `_derive_config_name_from_notebook` to also consult `DERIVA_ML_NOTEBOOK_PATH` (which is already set by `run_notebook.py:618`). Code shipped today; the workaround pollutes every notebook in the repo. After the fix, revert the Analyst's `run_notebook("roc_analysis", ...)` edit in `notebooks/roc_analysis.ipynb`. |
| Modeler `description=` friction (tk-004) | **Defer** | Documented across two runs now. Workaround (use experiment presets, not bare overrides) is clean. Hydra-side fix would be invasive. |
| InsecureRequestWarning flood in catalog-stored markdown exports | **Defer** (existing 27d/evaluator-02 finding) | Filed previously; no new info from this run. |
| `capture-tacit-knowledge` skill: nudge to use `ml.cite()` for **every** RID type, not just Datasets | **Skill iteration** | Mild gap visible in tk-001 / tk-004 / tk-005: hand-written URLs without snapshot suffix for Executions/Assets. Skill says "default `ml.cite()` is snapshot-pinned for all RIDs"; convention isn't fully adopted. One-line addendum to the snapshot-pinning section. |
| `capture-tacit-knowledge` skill: explicit "PR-number citations are not a footnote — name the durable behaviour" | **Skill iteration** | Single PR-number reference slipped back in (PR #46 in tk-004). The behaviour ("auto-composed descriptions only fire for `+experiment=` overrides") is what should be recorded; the PR is incidental. Minor skill refinement. |
| Curator's role: "consider creating a curated dataset variant" (Validation lane) | **Note for next run planning** | Two runs in a row, Curator concluded no work was needed. Skill is correctly leaving the call to the persona, but the Validation-bag dispatch path isn't getting tested. Could nudge in the scenario document without prescribing. |
| Denormalize path coverage | **Note for next run planning** | Two runs in a row, Analyst chose to join in pandas. PR #246 row-completeness work isn't exercised. Could mention in the Analyst persona prompt as an option. |

---

## Reproducibility

- **Catalog id:** 2 (preserve; query for archeology)
- **Branch:** `e2e-test/2026-05-27-e` (5 [E2E-DROP] commits ahead of main)
- **Persona execution RIDs:**
  - Loader (Phase 0): 46Y, FZC, JY8, TW0
  - Modeler triplet: XZP / Z1R / 103T (workflow XDG, commit `4b7f48bdd368…`)
  - Modeler Family-B smoke: XDP, XPR (workflow XDG)
  - Analyst: 11AY (workflow 115W, commit `08cd2561…` — current main HEAD)
- **Output assets (verified):**
  - Modeler triples: Y1G/Y1J/Y1M (XZP); Z3J/Z3M/Z3P (Z1R); 105M/105P/105R (103T)
  - Analyst outputs: 1184/1186/1188 (ROC curves), 118C/118E/118G (confusion matrices), 11D8 (cross-model plot), 118J (roc_metrics.csv), 11ER/11ET (executed notebook + markdown)
- **Local copies for offline review:**
  `findings/analyst/wide_joined_K16.csv` (500×35),
  `findings/analyst/ranking.csv`,
  `findings/analyst/per_class_recall.csv`,
  `findings/analyst/roc_metrics_from_catalog_11AY.csv`,
  `findings/analyst/rank_and_join.py` (re-runnable derivation script)
- **Independent re-derivation result (this evaluator):**
  XZP 25.20% / 0.7225 micro-AUC; Z1R 36.00% / 0.7951; 103T 36.80% / 0.8172.
  46 / 96 / 207 cross-model-agreement integers.
  **Every printed number in the Analyst's report matches.**

---

Generated by the Evaluator persona (second run under the
four-document framework: scenario + bootstrap + orchestrator +
evaluator; first run to exercise the new phase-3 wide-table
deliverable framing and the new tk-entry conventions).
