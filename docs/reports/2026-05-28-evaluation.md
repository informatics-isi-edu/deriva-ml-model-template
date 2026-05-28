# Multipersona E2E Run — 2026-05-28 Evaluation

**Catalog:** id=27, hostname=`localhost`, name=`e2e-test-20260528`
**Branch:** `e2e-test/2026-05-28`
**Sibling versions:** deriva-ml v1.40.2, deriva-ml-mcp v0.5.9,
deriva-mcp-core latest main, deriva-skills v1.2.4,
deriva-ml-skills v1.4.11.
**Run shape:** third run under the four-document framework
(scenario + bootstrap + orchestrator + evaluator); Curator →
Modeler → Analyst → Evaluator (cold).

---

## Headline

**The team produced a coherent, end-to-end correct story, the
catalog holds up under bit-for-bit inspection, and the platform
mostly stayed out of the way — but a single upstream defect in
`load-cifar10` (orphaned `Image_Classification` rows on retry)
quietly poisoned two downstream surfaces (`split_dataset` and the
`roc_analysis.ipynb` template) and the Curator's tk-001 / tk-002
characterisations are what saved the run from silently shipping
wrong numbers.** Every Modeler claim (three Toronto-pair runs
W76/XCE/YHP with the asset triples W92/W94/W96, XEA/XEC/XEE,
YKJ/YKM/YKP, all `Uploaded`) and every Analyst claim (joined wide
table 550×38, accuracies 24.0/37.8/41.1%, confidently-wrong
1/23/171) re-derives from the catalog independently. The most
important thing the user should act on is
**[`findings/evaluator/01`](../../findings/evaluator/01-loader-retry-leaves-orphaned-gt-feature-rows.md)**
— the orphaned-GT-rows loader bug is the single root cause of
three persona findings (curator/01, curator/02, analyst/02) and
will silently bite any future catalog where the loader has ever
been retried. The fix is one delete-then-insert step in the
loader's `--phase images` path.

---

## Catalog ↔ claim agreement

Verified directly against catalog 27 via the dev-localhost MCP
server and direct `deriva-ml` PathBuilder (the cross-channel
tie-breaker was not needed; both channels agreed on every spot
check).

| Claim source | Claim | Verified |
|---|---|---|
| curator/01 | `Execution=854` wrote 500 GT rows; `Execution=HSR` wrote 1100 | yes (`count_table(filters={"Execution": ...})` returns 500 and 1100 exactly) |
| curator/01 | 1600 GT feature rows across 1100 unique images | yes at Curator-arc time; today the table has 3250 rows because the three Modeler training runs added 1650 prediction rows (550 each × 3 executions) — consistent with tk-001's "after Modeler runs, GT and predictions interleave" |
| curator/02 | TCM∩TCY = 33; VAY∩VB8 = 24 | yes (direct PathBuilder set-intersection) |
| curator/02 | TCM=361 actual / TCY=105 actual / VAY=339 / VB8=95 | yes |
| curator (tk-002) | M16∩M1G = 0 (Toronto pair leakage-free) | yes (direct PathBuilder); also 55/class on both halves |
| modeler (tk-005) | W76 `Uploaded`, workflow W70, status='Uploaded'; XCE same workflow; YHP same | yes (`Execution` rows W76/XCE/YHP all `Uploaded`, all share Workflow=W70) |
| modeler asset table | (W76→W92/W94/W96), (XCE→XEA/XEC/XEE), (YHP→YKJ/YKM/YKP), with W96/XEE/YKP being `prediction_probabilities.csv` and YKJ being `cifar10_cnn_weights.pt` | yes (queried `Execution_Asset_Execution` linkage; sizes 83195/86566/91687 bytes for the three CSVs) |
| analyst | analysis execution **1012** has 7 linked assets (3 inputs W96/XEE/YKP + 4 outputs joined-wide-table.csv 243922 B, summary, per-class-confusion-long, per-class-recall) | yes — exact filenames and byte counts match the repo's `docs/reports/joined-wide-table.csv` (md5 `2d3d001d3b6a19e836fd0af50f4f5df4`) |
| analyst | notebook execution **ZW0** is `Uploaded` with ROC curves, confusion matrices, and the executed notebook | yes — 13 linked assets including `roc_curves_*.jpg`, `confusion_matrix_*.jpg`, `roc_metrics.csv`, `roc_analysis.ipynb`, `roc_analysis.md` |
| analyst report | accuracies W76=24.0%, XCE=37.8%, YHP=41.1% on n=550 M1G | yes — `(df[f"{m}_pred"] == df["True_Class"]).sum()` returns 132/208/226 = 0.2400/0.3782/0.4109 |
| analyst tk-007 | confidently-wrong (conf≥0.8 ∧ wrong) = 1 / 23 / 171; mean conf when wrong = 0.245 / 0.472 / 0.774 | yes — re-derived from `joined-wide-table.csv` matches exactly |
| analyst report | per-class recall table on n=55/class | yes — `per-class-recall.csv` rows match the analysis-report markdown table cell-for-cell |

**Cross-channel parity.** Spot-checked `count_table` (MCP) against
`PathBuilder.entities().fetch()` (deriva-ml Python) on the GT
counts and the dataset-image membership sets — both channels
returned identical values. No tie-breaker needed.

---

## Coherence of the team's deliverables

The chain Curator → Modeler → Analyst → external reader **holds
end-to-end** and is the strongest part of the run. The
Analyst's report is something a non-ML domain reader could
actually pick up and act on: every number traces to one CSV, the
ranking-by-accuracy / ranking-by-calibration tension is named
explicitly, and the caveats section is honest about
n=55/class noise and final-vs-best epoch.

Specific load-bearing handoffs:

- **Curator → Modeler:** tk-002 named the Toronto pair (M16/M1G)
  as the leakage-free pick before the Modeler picked a dataset.
  Modeler tk-004 cites tk-002 and adds the harness-shape reasoning
  ("one Training bag + one Testing bag from one execution so
  per-epoch test_acc is meaningful"). The Modeler did not
  rediscover the leakage trap — that's the kind of
  characterisation handoff the framework is supposed to enable.
- **Curator → Analyst:** tk-001 explicitly predicted the
  notebook-GT-execution trap before the Analyst hit it. The
  Analyst's findings/analyst/02 says so verbatim
  ("the Curator's `tacit-knowledge.md` [tk-001] explicitly
  documents this trap"). The Curator's characterisation work
  paid off twice: once for the Modeler, once for the Analyst.
- **Modeler → Analyst:** tk-005 + tk-006 named the
  final-vs-best-epoch convention that becomes load-bearing in
  the Analyst's calibration story (tk-007). The Analyst could
  cite "memorisation at epoch 17 per Modeler tk-005" without
  re-deriving the per-epoch arc from the training_log.txt.
- **Modeler → external reader (catalog):** `Execution.description`
  on W76/XCE/YHP carries the full experiment context including
  the `+experiment=cifar10_toronto_quick` override, so a reader
  who lands on the execution row in Chaise sees what was run
  without having to chase the workflow URL. PR #46's
  auto-composition is doing useful work here.

What's *not* perfectly coherent:

- The Analyst's report and the joined-table-summary CSV refer to
  the XCE experiment as `cifar10_default` in some places and
  `default_model` in others (`Execution.description` says
  "default_model", the per-class-recall.csv Label says
  "cifar10_default"). Not a finding — the labels disambiguate to
  the same execution via Execution_RID — but worth a future
  consistency pass.
- The analysis is split across two executions: 1012 (joined wide
  table + per-class derived CSVs) and ZW0 (ROC curves +
  confusion matrices on the n=250 truncated subset, with the
  GT-heuristic caveat). The Analyst flags this explicitly in the
  report's "deliverables" section, so a reader is not confused —
  but the right shape is one execution that covers both, and
  that's the lift `findings/evaluator/04` describes.

A cold reader handed only the worktree could reconstruct what
the team did and why. §3.2 coherence test passes.

---

## tacit-knowledge.md quality

Read top to bottom as a future contributor. **The file holds up
as a record a fresh reader would actually use.** 8 entries,
371 lines, each entry carries a real *why* rather than padding.

Comparison against the prior baseline:

| Metric | 27c | 27d | 27e | **27 (this run)** | Trend |
|---|---|---|---|---|---|
| Lines | 623 | 353 | 391 | **371** | flat vs 27e; well below 27c |
| tk-entries | — | — | 5 | **8** | +3 (more arc-derived content) |
| PR-number citations | 19 | 0 | 1 | **0** | clean; back to 27d's perfect record |
| TODO-list framing | reduced | absent | absent | **absent** | ✓ |
| Handoff-as-narrative blocks | present | brief | integrated | **integrated** (no "Handoff to next persona" headers; each entry has an "Implications for collaborators" footer that is *why*-shaped, not *what-next*-shaped) | ✓ |
| State-replication tables | several | 2 narrow | 2 narrow | **2 narrow** (tk-002 family overlap table, tk-005 per-execution hyperparameter table, tk-007 ranking table) — every table cites a finding or names a downstream consequence; not bare state replication | ✓ |
| `<a id="tk-NNN">` anchors | n/a | yes | 5/5 | **8/8** | ✓ |
| Click-through RID links | n/a | partial | partial | **adopted** for Datasets and Executions, snaptime-pinned where it matters (M16@355-RYPE-KKW8, YHP@355-RYPE-KKW8) | ✓ |
| Load-bearing `[inferred from pattern]` claims | unknown | none | none | **none** | ✓ |
| Dead-end entries / weighed-alternatives blocks | partial | yes | yes | **5 of 8 entries** (tk-002 considers building a clean labeled split and declines; tk-004 considers mutating template defaults and declines; tk-005 considers lower-lr/dropout/seed sweeps and declines each with reason; tk-007 considers macro-AUC ranking and declines with named-use-case reason; tk-008 considers per-class-recall ranking and declines with named-use-case reason) | ✓ strong |

**Strongest entries:** tk-007 (the accuracy-vs-calibration
ranking disagreement) and tk-008 (the bird↔deer scene-texture
confusion) are the kind of *interpretive judgments* a Modeler
or Analyst arriving cold would not derive on their own. tk-002
(the Toronto-pair recommendation with mechanism) is the
forward-looking convention that pays off for the next Modeler
arc. tk-001 (the dual-write Image_Classification convention) is
the load-bearing one that saved the Analyst from silently
running on 250/550 images.

**Minor things to watch on the next run:**

- The link text in tk-001's header reads
  "(feature Execution_Image_Image_Classification)" but resolves
  to `https://localhost/id/27/HSR@355-RWN7-R3D8` — an *execution*
  RID, not a feature-table link. The link works (Chaise will
  resolve to the execution context) but the label is misleading.
  Not severe; flagging in case the anchor template should
  distinguish entity-type in the link text.
- tk-008 starts "Inspected the symmetric off-diagonal mass" —
  this is procedural ("here's what I did") rather than tacit
  ("here's the invariant"). The *content* underneath is genuine
  tacit knowledge (the scene-texture failure mode hypothesis,
  why pairwise-confusion rankings beat per-class-recall rankings
  for surfacing non-intuitive results), but the framing could
  be tightened. Not a finding; an observation for the next
  capture-tacit-knowledge skill iteration.

**The skill is steering personas well.** Nothing in this run
reads like state replication or workflow directives. The
capture-tacit-knowledge skill's discipline is visibly internalised.

---

## Platform fitness

**Persona-filed friction.** Five findings filed in-arc, all
legitimate; my dispositions:

| Finding | Persona-set | Evaluator disposition |
|---|---|---|
| `findings/phase0/01-num-images-500-too-small-for-small-toronto-split.md` | "Phase 0 finding" (no severity) | **Medium / Doc gap.** Bootstrap doc prescribes an invocation that cannot succeed; loader correctly refuses; fix is one number change in the bootstrap doc plus a sentence about the trade-off. |
| `findings/curator/01-duplicate-image-classification-feature-rows.md` | Curator characterised, no severity | **Upstream cause of two other findings; root flagged as `findings/evaluator/01` at High/Bug.** Leave the Curator-perspective finding as-is — it's the right shape from inside the Curator arc. |
| `findings/curator/02-train-test-leakage-in-labeled-split-datasets.md` | Curator characterised, no severity | **Upgraded to `findings/evaluator/02` (High/Bug)** — the `split_dataset(row_per=feature_table)` default is the deeper template-shape bug, separable from the loader cause; either fix would address this catalog's leak, but the row_per fix is the durable answer for any future feature with non-1-to-1 row/image mapping. |
| `findings/analyst/01-roc-notebook-dry-run-cites-fake-rid.md` | Analyst characterised, no severity | **Medium / Bug.** Dry-run path is strictly less useful than real-run path because cell 3 crashes before any cells run. Workaround (skip dry-run gate) is trivial; the fix is a 3-line guard or an `ml.cite("0000")→"dry-run"` carve-out in deriva-ml. |
| `findings/analyst/02-roc-notebook-picks-wrong-gt-execution.md` | Analyst characterised, no severity | **High / Bug.** Notebook silently uses 250/550 test images and writes a catalog asset (`roc_analysis.md`, 990091 bytes) that under-reports sample size by 55%. Same root cause as evaluator/01; the notebook-side defence (pick max-rows, or accept explicit `gt_execution=`) is `findings/evaluator/04`. |

**Evaluator-filed findings** (under `findings/evaluator/`):

- **[`evaluator/01`](../../findings/evaluator/01-loader-retry-leaves-orphaned-gt-feature-rows.md)** — **High / Bug.** The loader leaves orphaned Image_Classification rows on retry. Root cause for curator/01, curator/02, and analyst/02.
- **[`evaluator/02`](../../findings/evaluator/02-split-dataset-row-per-feature-table-is-wrong-default.md)** — **High / Bug.** `split_dataset(row_per=feature_table)` is unsafe whenever a feature has >1 row per image. Separable from evaluator/01: even if the loader were fixed, this default is brittle for any multi-row feature.
- **[`evaluator/03`](../../findings/evaluator/03-template-dataset-descriptions-overstate-sizes.md)** — **Low / Doc gap.** TCC/VAP `with_description` strings still advertise 440/110 and 400/100; actuals are 361/105 and 339/95. Either fixes itself when evaluator/02 lands or needs a description rewrite.
- **[`evaluator/04`](../../findings/evaluator/04-roc-notebook-needs-explicit-gt-execution-knob.md)** — **Medium / Missing feature.** Even with the loader fixed and the heuristic improved, the right shape is an explicit `gt_execution=` knob on the notebook config, not a heuristic.

**Skill use observed.** Visible in the commits and tacit entries:

- `create-feature` selectors discipline is internalised — the
  Curator's tk-001 explicitly names "the HSR-filter is the
  durable answer" rather than reaching for `newest`. That's the
  selector teaching landing.
- `dataset-lifecycle` is internalised — the Modeler bundled
  `[M16, M1G]` as a single `cifar10_toronto_pair` group (one
  Training bag + one Testing bag, one execution) rather than
  two separate dataset configs, and tk-004 records *why*.
- `execution-lifecycle` is internalised — three training
  executions with status `Uploaded`, wired into `assets.py`
  immediately afterward, with proper provenance via
  `commit_output_assets`. The analyst arc's
  `scripts/build_joined_wide_table.py` creates its own
  Workflow + Execution for provenance rather than just writing
  CSVs into the worktree — that's the "track my work" trigger
  firing correctly.
- `capture-tacit-knowledge` is internalised — every persona
  wrote tk entries at decision points, none of them are state
  replication, and each entry has a "weighed alternatives"
  block.

**Skill / docs gaps the personas didn't flag:**

- **`run-notebook` skill silently lets dry_run hit the cell-3
  trap** (analyst/01). The skill should warn that dry-run on
  the shipped roc_analysis template will crash in the
  informational header cell. Better: deriva-ml's
  `ml.cite("0000")` should return a placeholder, not 404 — that
  fixes every notebook template that follows the same pattern.
- **No skill guidance on "when to write a standalone analysis
  script vs use the notebook template."** The Analyst's
  `scripts/build_joined_wide_table.py` is exactly the kind of
  artifact `dataset-lifecycle` + `execution-lifecycle` + a
  build-your-own-analysis-script skill could teach. The fact
  that the script works and is provenance-preserving says the
  pattern is supported; the lack of a skill to point at it
  means each Analyst has to figure it out from scratch.

**Missed friction (visible only on cold reading):**

- Three Modeler executions and the analysis execution all
  carry `Status_Detail` that's identical to `Description` — not
  a finding (it's how the harness emits them when the override
  composes a description) but the field becomes redundant from
  a Chaise-reader's perspective.
- `roc_analysis.md` (990091 bytes, asset ZZW) attached to ZW0
  is the catalog-stored markdown export of the executed
  notebook. On the n=250 truncated subset it documents the
  GT-heuristic miss prominently *in the notebook output*; a
  reader who finds it via Chaise would see the truncation. The
  silent-failure-mode framing in analyst/02 is the right
  framing for the platform-fitness story, but the catalog
  artifact itself is more honest than the notebook-run-only
  view would suggest.

---

## Comparison vs prior runs

Two prior evaluation reports exist on archive branches:
[`docs/reports/2026-05-27d-evaluation.md`](../reports/2026-05-27d-evaluation.md)
(commit `9b7037a`, first run under the four-document framework)
and `docs/reports/2026-05-27e-evaluation.md` (commit `617c488`,
second run, first phase-3 wide-table exercise). On the dimensions
that matter:

| Dimension | 27d (1st 4-doc run) | 27e (1st phase-3 run) | **27 (this run)** | Trend |
|---|---|---|---|---|
| Persona findings | 0 | 1 (analyst/01 = the PR-248 regression) | **5** (1 phase-0, 2 curator, 2 analyst) | +4 vs 27e — but this run's catalog has retried-load state by construction (a deliberate Phase 0 stressor), so several of the new findings exist *because* the bootstrap doc steered the operator into a known broken state. The pattern is "more friction surfaced", not "platform regressed". |
| Evaluator findings | 2 (Low/Polish) | 1 (High/Bug, upgrade of analyst/01) | **4** (2 High/Bug, 1 Medium/Missing feature, 1 Low/Doc gap) | Severity distribution shifted up. The two new High/Bug findings (evaluator/01, evaluator/02) are pre-existing platform defects that earlier runs didn't surface because earlier catalogs weren't retried-load. Not a regression — surfacing latent defects, which is what e2e is for. |
| Catalog ↔ claim agreement | clean | clean | **clean** | ✓ — three runs in a row with no catalog discrepancies |
| tacit-knowledge.md lines | 353 | 391 | **371** | flat, well below the 623-line 27c baseline |
| PR-number citations | 0 | 1 | **0** | back to clean |
| Wide-table phase-3 deliverable | n/a | exists, ran clean | **exists, ran clean** (550×38, byte-identical to catalog asset 1026) | ✓ |

**Headline trend:** the framework is producing reproducibly-clean
catalog ↔ claim agreement and tacit-knowledge quality across three
runs. The novelty this run is the *finding mix*: two High/Bug
evaluator findings exposed platform defects that prior runs'
catalogs didn't have the shape to surface. That's the e2e test
doing exactly what it's supposed to do — exercising the platform
through a real workflow that includes recovery paths (retry the
loader after a degenerate small-Toronto-split failure) and seeing
what falls out.

**Recurring vs new:**

- *Recurring across runs:* none. evaluator/01 from 27e (the
  PR #248 regression) does not reproduce this run because the
  Analyst arc didn't exercise the `config_name` auto-derive
  path — they used positional Hydra overrides (`assets=...`)
  per the run-notebook skill's guidance. Likely still latent,
  but no longer in the headline.
- *New this run:* evaluator/01 (loader-retry orphan rows),
  evaluator/02 (split_dataset row_per default), evaluator/03
  (TCC/VAP description drift), evaluator/04 (explicit
  gt_execution knob). All four are unsurfaced because prior
  runs hadn't been retried, hadn't used the leaky labeled-split
  family, and hadn't pushed the roc_analysis notebook against
  a multi-GT-execution catalog.

---

## Recommended actions

Ordered by my read on likely disposition. The user decides.

| Action | Suggested disposition | Why |
|---|---|---|
| **[`evaluator/01`](../../findings/evaluator/01-loader-retry-leaves-orphaned-gt-feature-rows.md)** — loader leaves orphaned Image_Classification rows | **Promote to GitHub issue (deriva-ml-model-template), High/Bug.** | Root cause of three other findings. Silent on fresh-load catalogs; arms after first retry. Fix is delete-then-insert in `--phase images`. |
| **[`evaluator/02`](../../findings/evaluator/02-split-dataset-row-per-feature-table-is-wrong-default.md)** — `split_dataset(row_per=feature_table)` default | **Promote to GitHub issue (deriva-ml-model-template), High/Bug.** | Separable from evaluator/01; defensive default for any future multi-row feature. One call-site change in `_cifar10_datasets.py`. |
| **[`analyst/02`](../../findings/analyst/02-roc-notebook-picks-wrong-gt-execution.md)** — notebook GT-execution heuristic | **Fix inline now.** | One-line fix: `.index[0]` → `idxmax()` on row count. Trivial; immediate user-visible improvement. After evaluator/01 + evaluator/04 land, the heuristic becomes belt-and-braces; before then it's the only thing standing between a user and a 55% silent sample loss. |
| **[`analyst/01`](../../findings/analyst/01-roc-notebook-dry-run-cites-fake-rid.md)** — `ml.cite("0000")` 404s during dry_run | **Fix inline now.** | Three-line guard in deriva-ml's `cite()` to return placeholder on the dry-run sentinel. Restores dry-run as a useful gate for every notebook template. |
| **[`phase0/01`](../../findings/phase0/01-num-images-500-too-small-for-small-toronto-split.md)** — bootstrap doc `--num-images 500` impossible | **Fix inline now.** | One-line change in `docs/test-plans/e2e-bootstrap.md` step 7. Without this fix, every future e2e run starts with a loader failure and a retry — which conveniently *was* useful for exposing evaluator/01 and evaluator/02, but should not be the framework's onboarding shape. |
| **[`evaluator/04`](../../findings/evaluator/04-roc-notebook-needs-explicit-gt-execution-knob.md)** — explicit `gt_execution=` knob | **Promote to GitHub issue (deriva-ml-model-template), Medium/Missing feature.** | Defence-in-depth for the GT-selection failure mode, even after evaluator/01 and analyst/02 are fixed. Modest lift. |
| **[`evaluator/03`](../../findings/evaluator/03-template-dataset-descriptions-overstate-sizes.md)** — TCC/VAP description drift | **Defer.** | Self-resolves if evaluator/02 lands. If evaluator/02 is deferred, file as a small doc-fix PR. Not urgent in either case. |

The phase-3 wide-table framing landed cleanly for the second run
in a row; the multipersona handoff chain works; the
tacit-knowledge skill is reliably steering personas off the
documented failure modes. The fix-pass agenda from this run is
short and high-leverage: two upstream `load-cifar10` bugs (the
real ones), a one-line notebook heuristic fix, and a dry-run
sentinel carve-out in `ml.cite()`. The framework itself needs no
changes I can see.
