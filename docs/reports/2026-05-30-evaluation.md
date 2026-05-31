# Evaluation — multi-persona e2e platform test (2026-05-30)

**Catalog:** `localhost` / catalog **168** (`e2e-test-20260530`)
**Branch:** `e2e-test/2026-05-30` (worktree
`deriva-ml-model-template-e2e`)
**Evaluator:** cold reader, ran after Curator → Modeler → Analyst; no
access to the persona prompts.

**Sibling versions:** deriva-ml `1.41.0` (commit 8c744045),
deriva-ml-mcp `0.5.10`, deriva-ml plugin `1.4.12`,
deriva-py `a8d42c8` (2.0.0.dev0). Dev-localhost MCP container runs
deriva-ml 1.41.0 / deriva-ml-mcp 0.5.10.

---

## Headline

This is the cleanest run I have reviewed. The Curator → Modeler →
Analyst chain tells one coherent story — characterize the substrate,
train a differentiated pair on the one leakage-free partition the
Curator identified, then score them held-out — and **every load-bearing
quantitative claim reproduces exactly from the catalog through both the
indirect (MCP/skills) and direct (deriva-ml Python) channels, with zero
discrepancies.** I re-derived the report's accuracy, per-class table,
mode-collapse counts, calibration table, and agreement breakdown from
the committed wide table *and* independently from the catalog's raw
`Image_Classification` feature values joined to ground truth — all three
agree to the last digit. The platform supported the work without
producing a single wrong result. The most important thing to act on is
not a correctness defect but a latent data-modeling gap the Curator
surfaced and correctly declined to paper over: the labeled-split
datasets are 100% derived from the training partition yet are tagged
`Dataset_Type=Testing` and are not registered as catalog children of
their source — a trap that *would* silently leak training data into a
held-out metric for anyone who trusts the type tag. The team avoided it;
a future user reading only the catalog might not.

---

## Catalog ↔ claim agreement (the load-bearing thread)

I sampled concrete claims from tacit-knowledge.md, the analysis report,
and the persona findings, and verified each against the catalog
directly. Cross-channel method: MCP resources for shape, deriva-ml
Python (`DerivaML.feature_values`, datapath on `Dataset_Image`) for the
authoritative re-derivation.

**Substrate (Curator, tk-001/002/003, findings/curator/01,02):**

| Claim | Catalog (direct) | Verdict |
|---|---|---|
| 13 datasets, types as tabulated | 13, types match | ✓ |
| F2T=550, F34=550, NEJ=110, PJ4=100, NE8=440, PHT=400, F3W=500, F46=500 | all match | ✓ |
| NEJ ⊆ F2T (110/110), PJ4 ⊆ F2T (100/100) — total leakage | `set(NEJ)≤set(F2T)`=True, `set(PJ4)≤set(F2T)`=True | ✓ |
| F2T∩F34=0, NE8∩F34=0, PHT∩F34=0 — all SAFE | all 0 | ✓ |
| NE8∪NEJ == F2T exactly | True | ✓ |
| F3W∩F2T=500, F46∩F34=500 (cross-family overlap warning) | 500, 500 | ✓ |
| F34 and F2T perfectly class-balanced (55/class) | confirmed via GT join | ✓ |
| Labeled-split roots NOT registered as children of F2T | `list_dataset_relations(F2T, both, recurse)` → parent F2J, **children []** | ✓ |

**Executions (Modeler, tk-004/005/006):** all 8 executions present and
`status=Uploaded`. RM8 (`cifar10_quick_toronto`) and SSE
(`cifar10_large_toronto`) both consumed dataset **F2J** (the Split
parent expanded to F2T+F34), seed=42, with `config_choices` matching the
claimed hyperparameters (RM8: 32→64 ch, 128 hidden, 3 epochs, batch 128;
SSE: 64→128 ch, 256 hidden, 20 epochs, batch 64). Each committed
weights + training_log + prediction CSV. The throwaway smoke run **QJ6**
consumed F3M (`cifar10_small_split`) exactly as tk-004 says, and — a
real trap the team navigated correctly — QJ6 *also* wrote 500
`Image_Classification` prediction rows on F46. The Analyst correctly
excluded QJ6 from every reported number.

**Feature values (tk-006 dual-purpose claim — the most dangerous join):**

- Catalog holds **2700** `Image_Classification` rows, partitioned by
  producing execution: **CVC=1100** (loader ground truth, all
  `Confidence` NULL), **QJ6=500**, **RM8=550**, **SSE=550**. Matches the
  report's "2700 rows ... GT + every model's predictions interleaved"
  exactly.
- Accuracy re-derived **directly from catalog feature values** (GT =
  CVC rows, predictions scoped by execution): **RM8 27.6364%, SSE
  37.6364%** — identical to the report and to `roc_metrics.csv`.

**Wide table ↔ catalog parity (Analyst's primary deliverable):**

- `prediction_wide_table.csv`: 550 rows, 550 distinct `Image_RID`, exactly
  the RM8/SSE prediction image set (= held-out F34).
- Row-by-row diff of the CSV against the catalog feature values:
  **0 ground-truth mismatches, 0 RM8 class mismatches, 0 SSE class
  mismatches, 0 RM8 confidence mismatches, 0 SSE confidence mismatches**
  across all 550 images.
- Every number in the report recomputes from the CSV: per-class table
  (airplane 47/44, automobile 13/55, bird 9/9, cat 4/27, deer 5/38, dog
  11/24, frog 73/40, horse 20/33, ship 29/40, truck 65/67 — all exact);
  RM8 mode collapse (frog 171, truck 115 = 52%, cat/deer ~10); calibration
  (SSE >0.70 wrong=223 vs right=170; RM8 conf_right 0.30); agreement (same
  197/36%, both-right 97, both-wrong 288, only-SSE 110, only-RM8 55) — all
  reproduce.
- Every output asset RID cited in the report exists in the catalog under
  execution **TYR** with the matching filename (V0R wide table, V14
  roc_metrics, V0T/V0W/V0Y ROC figures, V10/V12 confusion matrices, V28
  .ipynb, V2A .md). AUC values in the report (SSE 0.81 / RM8 0.73) are the
  correctly-rounded `AUC_Micro` from `roc_metrics.csv` (0.8118 / 0.7261).

**Nothing failed to hold.** The direct and indirect channels never
disagreed, so no cross-channel tie-break to raw ERMrest was needed.

---

## Coherence of the team's deliverables

The chain reads as one continuous, well-handed-off story.

- **Curator → Modeler.** The Curator did exactly the job the role exists
  for: not just "the data is balanced" but the *actionable* finding that
  the `Testing`-tagged labeled splits leak against an F2T-trained model,
  with the safe/leaky pairing table spelled out. The Modeler's tk-004
  cites that finding as the *reason* it picked the F2J pair — the handoff
  is explicit and load-bearing, not decorative. A Modeler who skipped the
  Curator's note would plausibly have trained on F2T and "evaluated" on
  NEJ, producing a leaked 90%+ number with no warning. The Curator
  prevented exactly that.
- **Modeler → Analyst.** The Modeler produced two runs that differentiate
  cleanly (a ~10-point held-out gap) on a shared, leakage-free eval set,
  committed comparable artifacts (per-run prediction CSV with the full
  probability vector, not just the argmax), and wired the join targets
  into `assets.py` as a named group. tk-006 pre-warns the Analyst about
  the dual-purpose feature — the single most error-prone step in the whole
  arc — and the Analyst's report shows it heeded the warning (the "How
  ground truth was joined" section scopes by producing execution, which is
  why my parity check found 0 GT mismatches).
- **The Analyst's report answers its question.** "Which model is better,
  and in what way?" is answered with a headline, a where-the-win-lives
  per-class breakdown, the frog artifact correctly flagged as *not* a
  strength, and the overconfidence caveat tied back to the Modeler's
  overfit observation. It is written for a non-ML reader without dumbing
  down the statistics, and it is fully re-derivable from one CSV — which I
  confirmed by re-deriving it.

A fresh reader given only this worktree + catalog could reconstruct what
the team did and why. No step requires having been in the room. The one
seam worth naming: the worktree's `notebooks/roc_analysis.ipynb` is the
*unexecuted* source (all `execution_count` null); the executed copy lives
as catalog assets V28/V2A. This is the correct DerivaML pattern and the
report links the executed assets — but a reader who opens the repo
notebook expecting outputs should know to follow the V28/V2A links. Not a
defect.

---

## tacit-knowledge.md quality

Strong. Eight entries, each a genuine *decision record* (the why), not a
state dump. It opens with an explicit "don't replicate catalog facts
here; link instead" preamble and **honors it** — RIDs appear as
`deriva://`-style links, not inlined tables of counts. Specific reads:

- **tk-002 (the leakage trap)** is the best entry in the file: a dead-end
  hazard ("training on F2T then evaluating on NEJ/PJ4 is total leakage"),
  a verified-facts block, the *operational* mechanism (the harness
  dispatches by `Dataset_Type`, so the trap fires with no warning), and a
  weighed-alternative (retag to `Validation`?) with a defensible reason it
  was *not* done. This is exactly what tacit knowledge is for.
- **tk-006 (dual-purpose feature convention)** is a true convention entry
  — it names a non-obvious gotcha (one feature, two producers, not
  distinguishable by table membership) and tells the next reader how to
  scope. My parity check is downstream proof the convention is correct.
- **tk-004** carries a useful term-of-art ("lane" dispatch by
  `Dataset_Type`) a domain reader needs and the catalog does not state.
- **tk-008** records a real platform gotcha (notebook `Execution.Description`
  shows the static config text, not the override; the real choice is in
  `config_choices`) — I verified the gotcha directly: TYR's prose says
  "default: quick vs extended" while `config_choices.assets` =
  `roc_quick_vs_large_toronto`. Recording the *behaviour* (not a PR
  number) is the right call; this entry will age well.

Discipline notes (minor, not failure modes): the two `[inferred from
pattern]` / `[observed]` markers (tk-007 RM8 mode-collapse causal story;
tk-008 cosmetic-description claim) are correctly *labeled* as inference
rather than asserted as fact — this is the file using the marker
convention the way it is meant to be used, not a load-bearing unverified
claim. tk-001 leans slightly toward enumerating per-split counts, but it
frames them as audit findings (min==max balance) rather than a lookup
table, and links F28 rather than dumping membership — it stays on the
right side of the state-replication line. No PR-number citations, no
TODO-list framing, no handoff-as-narrative. I did not modify the file.

---

## Platform fitness

The platform got out of the way almost entirely. Skills that should have
fired appear to have: the Curator used `list_dataset_relations` /
set-intersection for the lineage walk, the Modeler used the
experiment/execution config surface and `deriva-ml-run`, the Analyst used
the run-notebook positional-override pattern (tk-008 cites the
run-notebook skill's guidance by name). The dual-purpose-feature scoping
(create-feature skill territory) was handled correctly end-to-end.

Friction captured by the personas, with my classification:

- **`findings/modeler/01`** — `deriva-ml-run --info <group>` rejects a
  group-name argument (the positional-arg guard fires before Hydra's
  native `--info <thing>` parsing), and `--cfg job` is not passed through
  to Hydra. Both are legitimate; the guard shadows two standard Hydra
  inspection moves. I **uphold** this as **Medium / Polish** (workaround
  exists: `--info` with no arg, or `dry_run=true` as a heavier preflight).
  See `findings/evaluator/01-info-flag-passthrough.md`.
- **`findings/curator/02`** — labeled-split roots not registered as
  children of F2T. I **upgrade the platform angle** of this to a
  first-class finding: it is a bootstrap-loader gap (the canonical splits
  *do* register parent→child; the labeled-split path does not), and it
  combines with the `Dataset_Type=Testing` tag (curator/01) into a real
  leakage hazard that the catalog surfaces no guard against. **Medium /
  Bug** (in the loader's split-registration; the catalog itself is
  internally consistent). See `findings/evaluator/02-labeled-split-leakage-trap.md`.

Missed friction / shape observations:

- **No `findings/analyst/` directory.** The Analyst filed zero friction
  findings. Reading the report and the catalog, this looks like a
  genuinely smooth arc rather than silently-swallowed friction — the most
  error-prone step (the GT join) was done correctly, and the run-notebook
  override path worked. I note it for completeness, not as a concern.
- The notebook-description cosmetic mismatch (tk-008) is the one
  rough edge the platform itself introduced; it is correctly characterized
  as cosmetic (provenance intact in `config_choices`). **Low / Polish** if
  anyone wants to act on it; I did not file a separate finding since
  tk-008 already documents it well.

---

## Comparison vs prior runs

**No prior evaluation report is present in this worktree** (only
`docs/reports/2026-05-30-analysis.md`; the archived prior worktrees are
not available to me). Qualitatively: this run's defining characteristic
is the *absence* of catalog↔claim discrepancies, which is the failure
mode this evaluation thread exists to catch. The team also demonstrated
the platform's leakage-detection value proposition end-to-end — a Curator
finding directly changed a Modeler decision, which is the multi-persona
hand-off working as designed.

---

## Recommended actions (disposition is the user's call)

**Fix inline / quick:**
- None required for correctness — the run is clean.

**GitHub issue (platform):**
- Bootstrap loader: register labeled-split roots (NE0, PHJ) as catalog
  children of their source partition F2T, so `get_lineage` /
  `list_dataset_relations` surface the derivation as a walkable edge
  rather than description prose only (evaluator/02). Optionally pair with
  a `Dataset_Type` reconsideration (`Testing` vs `Validation`) for
  training-pool-derived splits — but that is a judgment call the Curator
  deliberately left to the user.
- `deriva-ml-run`: let the positional-arg guard pass through Hydra's
  `--info <group>` and `--cfg`, or have the error suggestion mention the
  `--info` (no-arg) and `dry_run=true` inspection paths (evaluator/01).

**Defer:**
- tk-008 notebook `Execution.Description` cosmetic mismatch — document-only;
  already captured in tacit knowledge. Act only if execution-description
  skimming becomes a common workflow.

**Dismiss:**
- The `roc_comparison.jpg` (worktree) vs `roc_comparison_TYR.jpg` (catalog
  asset V0Y) filename difference — benign provenance-suffix rename; the
  report's relative link resolves correctly on disk.
