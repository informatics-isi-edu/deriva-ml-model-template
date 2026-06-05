# Multipersona E2E Run — 2026-06-05 Evaluation

**Catalog:** id=69, hostname=localhost, name=`e2e-test-20260605`
**Branch:** `e2e-test/2026-06-05`
**Sibling versions:** deriva-ml `1.45.0.post1+g4d56677da` (one docs-only
commit past v1.45.0); deriva-mcp-core / deriva-ml-mcp-plugin at their
`main` HEADs (per tk-001); MCP test container rebuilt against these.
**Catalog access used:** `mcp__mcp-localhost__*` only (primer + auth
verified at start), plus direct `deriva-ml` / raw `deriva-py`
`ErmrestCatalog` Python. The `mcp-eye-ai-org` server was **not** used.

---

## Headline

This is the cleanest run the evaluator has data for. The three personas
produced a single coherent story — Curator proves the substrate is
leak-free and class-balanced, Modeler runs a controlled capacity sweep
that differentiates by design, Analyst scores the sweep against ground
truth and reconciles every number — and **catalog↔claim agreement held
on every spot-check, with zero discrepancies.** The load-bearing
integrity check (the headline accuracies 20/26/24%) was independently
recomputed by the evaluator from raw feature rows and matched
**byte-for-byte**; so did the coarse-vs-fine 79/81/82% and the macro/micro
AUCs to 3–4 decimals. There is no Blocker and no High catalog-state bug.

The single most important thing for the user to act on is **mundane and
template-level, not platform-level: the test suite ships red** on this
branch (`findings/evaluator/01`). A genuine template bug (the
`Image.filename` → `Image.Filename` denormalization-case fix) was
half-applied — source fixed, the three tests that guard it not — so
`pytest` is red and the regression guard is inert. It is mechanical to fix
and belongs as a `main` cherry-pick alongside the original case fix.

## Catalog ↔ claim agreement

The most load-bearing thread, and it is essentially flawless. Every claim
sampled from the artifacts was verified against the catalog directly
(MCP `mcp-localhost`, deriva-ml Python, and raw `ErmrestCatalog` set
arithmetic), and they agree:

| Claim (artifact) | Verified against catalog | Result |
|---|---|---|
| Two `Complete,Labeled` datasets H8M (live) / F2J (orphan), identical 1100-image membership (tk-003, curator/01) | `Dataset_Image` set arithmetic | ✅ `H8M set == F2J set`, both 1100 |
| KE0/KEA disjoint, partition H8M exactly (550+550) (tk-002) | direct set ops | ✅ disjoint, union == H8M |
| RQW(400)/RR6(100) ⊂ KE0, RR6 ∩ KEA = ∅, no test leakage (tk-002) | direct set ops | ✅ all hold |
| Feature table = 1400 rows = 1100 GT (CVP, Confidence NULL) + 3×100 predictions (tk-003/004, report) | MCP `count_table` + direct fetch | ✅ exactly 1400; CVP=1100, SR8/T1A/TAC=100 each |
| SR8/T1A/TAC all `Uploaded` with weights+log+CSV each; TKM `Uploaded` (tk-004, report) | `deriva_ml_list_executions`, `Execution_Asset` | ✅ 9 model assets + 7 analyst assets, all present |
| Headline test_acc 20/26/24% (report, tk-004/006) | **independent recompute from raw feature rows** | ✅ **byte-identical** 20.0/26.0/24.0% |
| All three runs scored the identical 100 images == RR6 (report) | image-set parity check | ✅ identical set, == RR6 members |
| Coarse animal-vs-vehicle 79/81/82% (report, tk-006) | recompute from joined CSV | ✅ 79/81/82% |
| macro/micro AUC 0.749/0.740/0.739 & 0.743/0.741/0.643 (report) | recompute via sklearn from softmax cols | ✅ matches to 3–4 dp |
| `get_lineage(TN0)` walks TN0←TKM←{ST6,T38,TCA}←{SR8,T1A,TAC}←RQP←KE0←H8M (tk-007, report) | `deriva_ml_get_lineage(TN0)` | ✅ `walked_complete: true` |
| 7 analyst output assets TN0/TN2/TN4/TN6/TN8/TNA/TNC under TKM (report) | `Execution_Asset` enumeration | ✅ all present, filenames match |

**No cross-channel disagreement surfaced**, so the tie-breaker (raw
`ErmrestCatalog`) was never needed to localize a bug — but I used the raw
channel anyway for the recomputations, and it agreed with both MCP and
deriva-ml. The platform's stored numbers are reproducible from its stored
raw data, with no fudge. The report's own self-described "load-bearing
integrity check" is corroborated independently.

One small reconciliation nuance, in the team's favour: the report's prose
chain "...← split exec ← KE0" compresses the actual two-hop
`...← QK4 (split exec) ← KE0 ← H7M ← H8M`. That is fair compression, not
an error — the full chain walks exactly as claimed.

Side-observation (not a discrepancy): execution **F1J** exists with status
`Failed` and is not named by RID in any tacit entry. It is fully explained
by `findings/phase0/01` — it is the aborted first `--phase datasets` run
(the `Image.filename` KeyError) and produced no orphan assets. Folded into
`findings/evaluator/03`; not a new finding.

## Coherence of the team's deliverables

A fresh reader given only the worktree + catalog could reconstruct what
the team did and why. The chain holds end to end:

- **Curator → Modeler.** The Curator's `scripts/curator_verify_splits.py`
  (19 read-only checks, all PASS — re-verified) established the precondition
  the Modeler needed: RQW/RR6 are leak-free, stratified, class-balanced
  (tk-002). The Modeler explicitly cites tk-002 as the reason the test
  numbers are trustworthy. The Curator also resolved the F2J ambiguity
  (use H8M, never F2J — tk-003) before it could trap a downstream persona.
- **Modeler → Analyst.** The Modeler ran a *controlled* capacity sweep
  (same dataset RQP, same seed=42, only capacity×duration varying) so the
  three runs are directly comparable — exactly what an Analyst needs to
  rank. Predictions committed as feature rows, three clean `Uploaded`
  executions with full asset sets, and the final-epoch caveat (tk-005)
  pre-flagged so the Analyst would not over-read the large run. The new
  experiment presets are RID-free reusable template config (verified:
  `experiments.py` references dataset/model *group names*), committed as
  `feat(experiments)`, not `[E2E-DROP]`.
- **Analyst report.** Answers the question it set out to ask — *which of
  the three runs to take forward, and is the platform's recorded number
  trustworthy* — with figures that support the conclusions and explicit
  caveats where the data is thin (100-image test set ⇒ ±~3 images noise;
  default-vs-large a near-tie on top-1; the clean separation is quick's
  calibration). It carries the two coexisting facts ("recorded ==
  recomputed" *and* "recorded ≠ best-epoch") without contradiction, and
  the domain reading (coarse learned, sensible confusions, mid-size wins)
  is the right altitude for a pipeline-validation run on 400 images.

The deliverable split the Analyst chose (pure RID-free join/metric logic in
`src/scripts/analyst_join.py` + unit tests; catalog RIDs only in the
`[E2E-DROP]` driver `scripts/analyst_analysis.py`) is a genuinely good
pattern worth copying, and the report documents it. Coherence: **no
findings.**

## tacit-knowledge.md quality

Seven entries (tk-001 … tk-007), read top to bottom as a future
contributor. This is strong tacit knowledge, largely free of the failure
modes the rubric warns about.

What's good (and why):

- **Convention entries that will save a future reader real time:** tk-003
  ("`Image_Classification` is dual-purpose — filter `Confidence IS NULL`
  for ground truth; the `newest` selector is *not* a safe substitute") and
  tk-005 ("the template records FINAL-epoch, not best-epoch predictions")
  are exactly the gotchas the file exists for. tk-005's "if best-epoch is
  ever wanted, that's a model-code change, not a config tweak" is the kind
  of forward-looking boundary that prevents a wasted afternoon.
- **A real dead-end / hazard, correctly dispositioned:** tk-003(2) on F2J
  ("pinning F2J would silently sever provenance; do-not-use; not deleted
  because deletion needs authorization") is high-leverage and names the
  consequence, not just the fact.
- **Decisions carry their rationale and alternatives:** tk-004/tk-005
  explain *why* dataset+seed are held fixed (so capacity is the only
  variable) and explicitly tell a future modeler to vary `datasets=` and
  hold the model fixed if they want the other axis — a convention that
  isn't otherwise documented.
- **The `Supported by:` back-references** turn the seven entries into a
  navigable dependency graph rather than a flat log.

Minor observations (not findings, calibration for future runs):

- The entries lean **long and somewhat narrative** — tk-004 and tk-006
  restate the leaderboard and the coarse-vs-fine result that also live in
  the report. This edges toward handoff-as-narrative, but stays on the
  right side: each restatement is attached to a *durable interpretive
  judgment* ("state the ranking as final-epoch states, never as best each
  model can do"), which is genuinely tacit, not just state. A future run
  could tighten by linking to the report for the numbers and keeping only
  the judgment.
- tk-001 reproduces some bootstrap state (image counts, split sizes) that
  the catalog also holds. It is borderline state-replication, but it is
  framed as *bootstrap provenance* (what the catalog was seeded from and
  why the floor is >1000) rather than a queryable snapshot, and it links
  to live RIDs rather than inlining their contents. Acceptable.

No PR-number-as-load-bearing-citation (phase0 finding 01 cites deriva-ml
#283 but only as the *cause-explanation footnote*, and it's in a finding
file, not tk). No TODO-list framing. No load-bearing `[inferred from
pattern]` claims. **No findings against the file.**

## Platform fitness

The platform supported the work well. Across three personas doing schema
characterization, controlled training, and provenance-linked analysis,
the only friction was three pre-existing/cross-tier seams, none of which
blocked a deliverable:

- **Template test suite red** (`findings/evaluator/01`, High) — the only
  finding that touches the shipped artifact. A half-applied case fix.
  Verifiable, deterministic, mechanical to fix.
- **Generic catalog tools reject the logical feature name**
  (`findings/evaluator/02`, Medium) — bit the Modeler *and* the Analyst at
  the same seam (logical `Image_Classification` vs physical
  `Execution_Image_Image_Classification`); the 409 "does not exist" message
  misleads. Promoted to a cross-persona finding precisely because it
  recurred. Both personas recovered via `deriva_ml_list_features` /
  `deriva_ml_list_feature_values`.
- **Orphan duplicate `Complete` dataset** (`findings/evaluator/03`,
  Medium) — loader idempotency didn't reuse the aborted run's `Complete`,
  leaving F2J indistinguishable from H8M in a bare listing. A genuine
  provenance-severing trap, handled correctly by the team via tk-003.
- **Bare `ErmrestCatalog` needs explicit credential**
  (`findings/evaluator/04`, Low) — friction/doc gap, not a defect; the
  recommended `DerivaML(...)` surface never hits it.

**Skill use:** the commit log and tacit entries show the right skills
firing at the right moments — capture-tacit-knowledge produced
seven well-shaped entries; the Analyst captured the analysis itself as a
provenance execution that *consumes the prediction CSVs as declared
inputs* (tk-007), which is exactly the execution-lifecycle discipline that
makes `get_lineage(TN0)` walk in one call. No persona reached for raw
deriva-py where a deriva-ml surface existed, **except** the deliberate
low-level `ErmrestCatalog` verification scripts (Curator's split-integrity
check, evaluator's recompute) — which is the appropriate tool for
independent, helper-free verification, not a skill miss.

**Missed friction (evaluator, in retrospect):** none beyond the four above.
The `count_table`/logical-name seam is the one I'd most want closed before
the next run, because it is the natural first move for a modeler verifying
their own output and it greets them with "does not exist."

## Comparison vs prior runs

No prior `*-evaluation.md` exists under `docs/reports/` — this is the
**first evaluation report in this worktree** under the e2e-evaluator
rubric. (Nine prior e2e branches are archived at
`origin/archive/e2e-test-2026-05-*` and `…-06-01`, but none carried a
committed evaluation report into this tree, so no quantitative
severity-trend comparison is possible.) The rubric notes a healthy trend
is "findings decrease and the remaining ones are less severe"; this run
sets the baseline at **0 Blocker, 1 High, 2 Medium, 1 Low**, with the High
being a template test gap rather than a platform-state bug and with
catalog↔claim agreement perfect. Future runs can be measured against this.

## Recommended actions

Organized by likely disposition; the user decides per item.

**Fix inline (cherry-pick to `main`, not `[E2E-DROP]`):**
- `findings/evaluator/01` (High) — update the three `cifar_canonical_partition`
  test fixtures + docstring `Image.filename` → `Image.Filename`, and the
  stale source comment at `_cifar10_datasets.py:426`. This belongs *with*
  the original case fix `65ae86b`; ship them together so the next clone is
  green. Mechanical, low-risk.

**Promote to GitHub issue:**
- `findings/evaluator/02` (Medium, cross-persona) — error-message hint on
  the generic catalog tools when a logical feature name is passed ("did
  you mean `Execution_<Target>_<Name>`?"), plus a skill/doc steer. Likely
  `deriva-mcp-core` + a deriva-ml-skills doc note. Recurred across two
  personas ⇒ worth a tracked issue.
- `findings/evaluator/03` (Medium) — loader idempotency: on
  `--phase datasets` re-run, reuse or tag the orphan `Complete` so a bare
  listing distinguishes live from orphan. Template repo issue.

**Defer:**
- `findings/evaluator/03`'s deriva-ml sub-suggestion — extend
  `split_dataset`'s `selection_fn` path to validate selector-read columns
  (the stratified path already does). Nice-to-have; would have made the
  Phase-0 `KeyError` self-explaining. deriva-ml change, lower priority.

**Dismiss (note + reason):**
- `findings/evaluator/04` (Low) — bare `ErmrestCatalog` credential. Correct
  platform behavior; the `DerivaML(...)` surface never hits it. Doc note at
  most; no code change warranted.

---

*Findings filed by the evaluator: `findings/evaluator/01`–`04`. Persona
findings folded in: `findings/phase0/01-02`, `findings/curator/01-03`,
`findings/modeler/01`. All catalog reads via `mcp-localhost` + direct
deriva-ml/deriva-py against localhost catalog 69.*
