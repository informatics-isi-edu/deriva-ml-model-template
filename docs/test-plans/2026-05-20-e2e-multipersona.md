# End-to-End Multi-Persona Platform Test

**Author:** Carl Kesselman (with Claude)
**Date:** 2026-05-20
**Supersedes:** `docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md`
**Status:** Approved for execution.

---

## 1. What this test is for

The DerivaML platform (deriva-ml core library + the two MCP servers +
the two Claude Code skill plugins + this model template) is meant to
support several distinct kinds of work, each done by a different kind
of user. The May 2026 single-agent platform-fitness run (see the prior
spec) shook out 19 bugs and got the stack into shape; the platform is
now stable enough that the question shifts from "does it work?" to
"what's it like to *use* it?"

This test answers that question by putting **multiple persona agents**
through end-to-end workflows on a shared catalog. Each persona acts
like a real user with real goals; each surfaces friction — anything
that gets in their way, whether it's a bug, a missing skill, a
confusing description, a misleading error, or a documentation gap.

The output is a friction map per persona, captured as structured
findings during the run.

### Primary goals

1. **Characterize the user experience per persona.** Where is the
   platform smooth? Where is it rough? Friction is the unit of
   measurement.
2. **Test `tacit-knowledge.md` as a real knowledge-transfer
   artifact.** Each persona writes to it during their work; the
   next persona reads from it before starting. Gaps in the handoff
   are findings — about the file, the prior persona's writing, or
   the `capture-tacit-knowledge` skill itself.
3. **Confirm what the indirect channel (skills + MCP tools) reports
   matches the catalog's actual state.** Each persona's normal work
   uses skills and MCP tools, but before declaring their arc done
   they must verify directly (deriva-ml Python API, raw ermrest)
   that the catalog actually contains what the tools said happened.
   Disagreement is a finding — and historically the most valuable
   kind. See §3.4.
4. **Surface bugs and rough spots** *as a byproduct* of the personas
   doing their work. The personas are not bug hunters; they're users.
   Bugs they hit are findings; bugs they don't hit aren't relevant
   to this test.

### Non-goals

- Coverage of every model config / experiment combination.
- Performance benchmarking.
- Multi-host / cluster scenarios — `localhost` only.
- Inline bug-fixing during persona arcs. Findings are *captured*, not
  *resolved*, mid-arc. A separate fix-pass agent (post-run, or
  between phases in interactive mode) handles resolution.

---

## 2. Personas

Three personas exercise distinct slices of the platform. Each has a
**goal** (what they're trying to accomplish), **skills they should
reach for**, and **success criteria** (how we know they got there).

### 2.1 The Curator

> *"Someone handed me a freshly-bootstrapped catalog of image data.
> My job is to understand what's in it, make sure the canonical
> datasets and ground-truth labels are sane, create the dataset
> variants downstream users will actually train on, and document
> the catalog's shape for them. I don't train models; I curate."*

**Inputs (set up by Phase 0 bootstrap, before this persona starts):**
- A fresh catalog at `localhost` named `e2e-test-<YYYYMMDD>`.
- Domain schema populated by `load-cifar10` (Image table, vocabularies,
  built-in datasets, `Image_Classification` ground-truth feature
  values for labeled partitions).
- `src/configs/deriva.py` `default_deriva` already points at the new
  catalog id; `src/configs/datasets.py` already carries the
  loader-produced RIDs. Both edits are `[E2E-DROP]` commits on the
  shared `e2e-test/<YYYY-MM-DD>` branch (see §3.5).
- `tacit-knowledge.md` has a single "Bootstrap" entry from Phase
  0 noting what was created and how.

**Goal:** Audit the bootstrapped catalog, verify it's in shape for
downstream personas, then *add value* on top of it: create at least
one curated dataset variant (a subset or a new split) that exercises
the dataset-lifecycle skill, and document the catalog's shape and
the curation rationale for downstream personas.

**Primary skills/tools:** `dataset-lifecycle`, `create-feature` (in
query mode), `manage-vocabulary`, `capture-tacit-knowledge`.

**Success criteria:**
- Curator has inspected the built-in datasets and confirmed their
  shape matches what the spec said Phase 0 would produce. Any
  mismatch is a Phase 0 finding, not a Curator finding.
- `Image_Classification` ground-truth values are present for the
  labeled partitions; curator has spot-checked a sample.
- At least one new dataset (a curated subset or new split) created
  via `dataset-lifecycle`, with a real motivation that a downstream
  persona would care about (not "to exercise the API").
- `tacit-knowledge.md` contains entries explaining: what the
  curator inherited and what their assessment of it is, what new
  dataset was created and why, what downstream consumers should know.
- A "handoff summary" to the next persona at the bottom of the
  curator's notes: what's ready, what's pinned, gotchas.

---

### 2.2 The Model Developer

> *"I want to train a model on the curator's data and find out which
> architecture or hyperparameter setting works best. I care about
> reproducibility (so I can rerun the winner) and about not breaking
> anything the analyst depends on downstream."*

**Goal:** Train two model variants against the curator's datasets,
run a multirun parameter sweep, and leave the resulting executions
(with predictions and weights) for the analyst to compare. Document
which configs were tried and why.

**Primary skills/tools:** `execution-lifecycle`, `run-notebook`
(if a notebook entry-point feels natural), `configure-experiment`,
`write-hydra-config`, `compare-model-runs` (optional, in a "did my
new variant beat the baseline?" sense), `capture-tacit-knowledge`.

**Success criteria:**
- At least two distinct training runs completed, with weights and
  predictions uploaded as `Execution_Asset` rows.
- At least one multirun (e.g., `quick_vs_extended` or `lr_sweep`)
  completed; parent and child executions correctly linked.
- New experiment config registered in `src/configs/experiments.py`
  (on the shared e2e branch) if the developer needed one beyond the
  existing ones.
- `tacit-knowledge.md` contains entries explaining: which
  variants and why, which seed strategy, what success looked like.
- Handoff summary: which executions the analyst should look at,
  which prediction assets feed the analysis, any caveats.

---

### 2.3 The Analyst

> *"I want to look at the model developer's runs and figure out
> which one's best, build a few plots, and write up the result. I
> don't train models; I consume them."*

**Goal:** Compare the developer's training runs (ranking by
accuracy / AUC / etc.), produce an analysis notebook (ROC, confusion
matrix, or similar), and write a short markdown report a reviewer
could read in 5 minutes. As part of the analysis, exercise the
**dataset denormalize** path (`deriva_ml_denormalize_dataset` via
MCP and/or the corresponding deriva-ml Python API) to materialise a
wide/flat view of the dataset the developer trained on and use it
to drive the comparison — this is the test's deliberate exercise of
the denormalize surface.

**Primary skills/tools:** `compare-model-runs`, `run-notebook`,
`execution-lifecycle` (for executing the notebook with provenance),
`dataset-lifecycle` (specifically the denormalize / wide-table
section), `capture-tacit-knowledge`.

**Success criteria:**
- A ranking of the developer's executions by at least one metric.
- One executed analysis notebook (e.g., `notebooks/roc_analysis.ipynb`
  or a new one) producing plot asset(s) + a summary CSV asset.
- **Denormalize exercised end-to-end:** the persona calls
  `deriva_ml_denormalize_dataset` (or the deriva-ml Python equivalent)
  on at least one of the developer's training/evaluation datasets,
  uses the resulting wide table in the analysis (e.g., to join
  predictions to ground-truth labels for ROC / confusion matrix),
  and verifies the wide table's shape and contents against the
  direct-channel dataset members (§3.4) — row count, label
  distribution, and join keys must match. Disagreement is a finding
  filed against the denormalize surface specifically.
- A short markdown report under `docs/reports/` (created by this
  persona) summarizing the comparison, what's in the catalog now,
  any caveats, AND a brief subsection on the denormalize experience
  — was it discoverable, did the output match expectations, did the
  column naming / element-type ordering match what the persona
  needed for the analysis.
- `tacit-knowledge.md` contains entries explaining: which runs
  were compared and why, what metric was chosen, how surprises
  (if any) were interpreted, and the rationale for the denormalize
  call (which element type was treated as the "root", why).

---

### Persona ordering and dependencies

Curator → Developer → Analyst. Strictly sequential. The developer
cannot start until the curator has produced datasets the developer
can train on; the analyst cannot start until the developer has
produced runs the analyst can compare.

---

## 3. Execution model

### 3.1 Modes — pick one at session start

**Interactive mode.** After each persona's arc finishes, the run
pauses. The user reviews the persona's summary, the findings file,
and the tacit-knowledge handoff. The user can redirect, ask
for elaboration, request a re-do of a specific step, or proceed to
the next persona. This mode is for first-time runs and runs where
the user wants to verify the personas are behaving sensibly.

**Autonomous mode.** All three personas run their arcs back-to-back
without checkpoints. At the end, the orchestrator produces a
consolidated friction map and findings report for the user to read.
This mode is for repeat runs once the user trusts the personas, for
overnight execution, or for batch comparison of multiple platform
versions.

The mode is selected once, at session start, by the user. It does
not change mid-run. (If the user is interactively monitoring and
wants to step away, the choice is to abort and re-launch in
autonomous mode, not to switch modes inside one run.)

**Agent-initiated inquiry is allowed in either mode.** The mode flag
governs *checkpoint pauses* (does the orchestrator wait between
persona arcs?) — it does **not** restrict persona agents from raising
a short clarifying question to the user *during* an arc when the
answer would materially improve what gets recorded in
`tacit-knowledge.md` or `findings/`. Inquiry is distinct from a
checkpoint: it's an inline question that doesn't pause the arc, and
the user's answer feeds the next sentence the agent writes. In
autonomous mode the bar is higher (asking interrupts the autonomy
contract), so default to provenance markers and inquire only when a
load-bearing claim would otherwise be `[inferred from pattern]` —
see the `capture-tacit-knowledge` skill's "When to inquire"
section for the budget, threshold, and confirmatory-shape rules.

### 3.2 Decision rights — what an agent can decide alone

The personas need clear ground rules about when to act and when to
escalate. The rules differ by mode.

| Decision | Interactive | Autonomous |
|---|---|---|
| Which existing dataset/feature/config to use for an obvious task | Decide | Decide |
| Reasonable parameter choice (split ratio, learning rate, epoch count) within typical range | Checkpoint summary | Decide; note the choice in `tacit-knowledge.md` |
| Pick between two equally-valid skills | Checkpoint summary | Decide |
| Create a new dataset / feature / config not strictly required by the success criteria | Checkpoint, ask first | Decide if it serves the persona's goal; note rationale |
| Destructive operations (delete catalog, drop schema, force-push, rm -rf working dir) | Always ask | Always ask — abort the persona if blocked |
| Schema migrations (new column, FK change, drop table) | Always ask | Always ask — abort the persona if blocked |
| Fix a bug encountered mid-arc | Always ask | Never. File a finding and route around if possible. |
| Skip a success criterion because the platform won't support it | Checkpoint, explain | File a finding with "blocked at" detail; proceed if possible |

The bright lines: destructive operations and schema migrations always
require explicit user authorization, regardless of mode. Persona
agents never fix bugs mid-arc — that's a separate fix-pass.

### 3.3 Per-persona workflow

Each persona, regardless of mode, follows the same arc:

1. **Read context.** Project's CLAUDE.md, the persona's own brief in
   this spec, and (critically) `tacit-knowledge.md` if it
   exists. The previous persona's handoff is in that file. Surface
   any handoff gaps as findings immediately.
2. **State the plan.** Persona writes a 5-bullet plan of what they're
   about to do. In interactive mode, this is shown to the user as
   the entry checkpoint. In autonomous mode, it's the persona's
   own first decision-log entry.
3. **Do the work.** Persona executes their plan, reaching for the
   skills and tools listed in §2 first. Friction at every step
   gets captured (§4).
4. **Capture rationale.** As decisions are made, persona writes
   them to `tacit-knowledge.md` via `capture-tacit-knowledge`.
   At minimum: one entry per major decision (dataset choice, split
   strategy, model config selection, metric choice).
5. **Cross-channel verification.** Before declaring the arc done,
   the persona verifies that the catalog *actually* contains what
   their skills and tools *said* they created. See §3.4. Disagreement
   is a finding.
6. **Write handoff.** At end of arc, persona appends a "handoff
   summary" section to `tacit-knowledge.md` named for the
   next persona, describing what's ready and what's pinned. This
   is the explicit knowledge-transfer step.
7. **Produce arc summary.** A markdown summary of what was done,
   findings raised, decisions captured, and success-criteria
   status (which met, which not, why). In interactive mode this
   is the exit checkpoint; in autonomous mode it feeds the final
   consolidated report.

### 3.4 Cross-channel verification

The single most important methodology principle from the May 2026
run: **the catalog's actual state and what the skills/MCP tools
report about its state must agree.** When they don't agree, it's
usually the skill/MCP side that's wrong, and the discrepancy is
exactly the kind of friction this test exists to surface.

Each persona's normal work uses the **indirect channel** — skills
and MCP tools, the surface a real user would see. Before declaring
their arc done, the persona must check the **direct channel** —
deriva-ml Python API or raw ermrest, with no skill or MCP indirection
— and confirm the catalog state matches the indirect channel's
reports.

**What to verify** depends on the persona; minimums:

- **Curator:** every dataset they reported creating, every dataset
  type assigned, every member added — visible via `ml.find_datasets`,
  `ml.lookup_dataset(rid).list_dataset_members()`, with counts
  matching what the skill said.
- **Developer:** every Execution row reported as committed, every
  Execution_Asset uploaded — visible via `ml.find_executions`,
  `ml.lookup_execution(rid).list_assets()`, with counts and statuses
  matching.
- **Analyst:** every plot, summary CSV, or notebook asset reported
  uploaded — visible via direct asset queries. Predictions used in
  the analysis match what the developer's executions actually
  produced (cross-persona check). **Denormalize output verified
  against direct channel:** the wide table returned by
  `deriva_ml_denormalize_dataset` is reconciled against the dataset's
  members as seen via `ml.lookup_dataset(rid).list_dataset_members()`
  and the underlying feature-value query — row count, the set of
  member RIDs, and label distribution must agree. If the wide table
  is missing rows, has duplicated rows, or carries labels that don't
  match the ground-truth feature values, that's a high-severity
  finding against the denormalize surface, filed even if the
  analyst's downstream deliverables happen to still be producible.

**What to do on disagreement:**

1. Write a finding (§4) at the exact point of disagreement. Capture
   both the skill/MCP report AND the direct-channel query result
   verbatim.
2. If the persona's deliverable depends on the catalog actually being
   in the state the skill reported, the persona is blocked. Note in
   the arc summary which success criterion failed and why.
3. If the deliverable is unaffected (the discrepancy is in metadata
   the persona didn't need), proceed; the finding documents the
   discrepancy for the fix-pass.

**Tie-breaker channel:** if direct (deriva-ml Python) and indirect
(MCP / skill) disagree and `deriva-ml` is in both code paths (which
it is for most catalog operations), the persona should drop one
level lower and use raw `ermrest_catalog.get(...)` or
`DatapathBuilder` with no deriva-ml helpers to break the tie. This
identifies whether the bug is in deriva-ml itself or in the layer
above it.

This step is mandatory regardless of mode. Personas don't get to
skip it because they "feel good about the work" — the May 2026 run
caught multiple high-severity bugs precisely because the indirect
channel reported success while the direct channel revealed silent
failures.

### 3.5 Multi-agent setup

Each persona runs as its own Agent-tool invocation with a dedicated
system prompt drawn from §2. **All three personas share a single git
worktree** on a single dedicated e2e branch — they run sequentially in
the same working tree, not in per-persona worktrees. The catalog is
also shared.

Branch / worktree convention:

```
git worktree add ../deriva-ml-model-template-e2e \
    -b e2e-test/<YYYY-MM-DD>
```

This worktree is created in Phase 0 step 0 (§6.2) before any persona
runs. All persona work — config edits, `tacit-knowledge.md`
appends, findings under `findings/<persona>/`, helper scripts,
commits with `[E2E-DROP]` markers — happens here, on this branch.

**Why single-worktree, not worktree-per-persona.** The May 2026 spec
chose worktree-per-persona to prevent file-stomping between
concurrent agents. Personas in this run are sequential, not
concurrent, so the file-stomping risk doesn't apply. The cost of
per-persona worktrees was much higher: each persona's
`tacit-knowledge.md`, config edits, and findings lived in a
separate working tree, and the orchestrator had to merge between
branches to carry the handoff forward. That made the knowledge-
transfer artifact — the whole point of §5 — implicit in the
orchestrator's merging discipline rather than naturally available to
the next persona. Single-worktree restores the handoff as the
straightforward chain it should be: persona N writes,
persona N+1 reads from the same files.

**Concurrent variant (future).** If a future run ever wants to
exercise concurrent personas (e.g., Curator on labeling while
Developer trains on an earlier dataset version), reintroduce
per-persona worktrees and treat each merge as an explicit
synchronization point. Out of scope here.

---

## 4. Capturing findings

A finding is *anything that got in the persona's way*: a bug, a
broken skill route, a missing tool, a confusing error message, a doc
gap, a workflow that felt longer than it needed to be. Findings are
captured immediately at point of friction, not retrospectively.

### 4.1 File layout

Findings live in `findings/<persona>/<NN>-<slug>.md` in the persona's
worktree. Numbered for ordering; slugged for readability.

```
findings/
  curator/
    01-dataset-types-not-discoverable.md
    02-add-term-error-message-cryptic.md
  developer/
    01-multirun-parent-execution-dry-run-warning.md
  analyst/
    01-compare-model-runs-no-prediction-csv-pattern.md
```

### 4.2 Finding-file template

```markdown
# <Short title>

**Persona:** Curator | Developer | Analyst
**Phase:** <what the persona was trying to do>
**Severity:** Blocker | High | Medium | Low | Polish
**Component:** <repo or skill name, if known>

## What happened

<Free-form: what the persona was doing, what they expected,
what actually occurred. Include exact commands, error messages,
file paths, RIDs.>

## Reproduction

<Exact steps. RIDs are catalog-specific; describe how a future
reader would re-find the relevant entity (e.g., "the latest
training execution against dataset cifar10_labeled_training_localhost").>

## Impact on the persona's work

<Did it block them? Did they route around it? How much time did
it cost? Did it affect a deliverable in §2 success criteria?>

## Suggested classification

<Bug | Missing feature | Skill issue (triggering / routing /
behavior / missing) | Doc gap | Tool gap | Polish.>

## Notes for the fix-pass

<Anything you noticed about scope, related code, things to verify
when fixing. Keep brief.>
```

### 4.3 Promotion to GitHub issues

Persona agents do not file GitHub issues during the run. The local
files are the durable artifact. After the run, the user reviews the
findings collection and decides which to promote to issues, which
to fix inline, and which to discard.

The fix-pass agent (or the user) handles promotion. A small helper
script under `scripts/` could automate the promotion step but is
not part of this spec.

### 4.4 The friction map (final report)

After all three personas finish, the orchestrator produces a
consolidated report at `findings/REPORT-<YYYY-MM-DD>.md`:

```markdown
# E2E Multi-Persona Friction Map — <date>

## Per-persona summary

### Curator (N findings)
- 01-<slug>: <one-line summary> — <severity>
- ...

### Developer (N findings)
- ...

### Analyst (N findings)
- ...

## Patterns

<Cross-cutting observations: friction the same persona hit twice,
friction multiple personas hit in different forms, places the
platform asked the user to know something they shouldn't have to.>

## Handoff quality

<Did each persona understand the prior persona's intent from
`tacit-knowledge.md`? Specific examples of what carried over
well vs. what was unclear.>

## Success-criteria scorecard

| Persona | Criteria met | Criteria missed | Notes |
|---|---|---|---|

## Recommended action

<Suggestion to the user: which findings look like bug-fixes,
which look like design discussions, which look like one-line doc
fixes. Not prescriptive — the user decides.>
```

---

## 5. `tacit-knowledge.md` as test artifact

The file lives in the project root and is tracked in git. Each
persona is expected to:

- **Read** the file at startup, before doing any work, to inherit
  prior personas' context.
- **Write** to it via `capture-tacit-knowledge` at decision
  points throughout their arc.
- **Append a handoff section** at end-of-arc with explicit
  instructions for the next persona.

The "did the handoff work?" assessment is part of each persona's
arc summary. Specific questions to answer in the summary:

- What entries did the prior persona write that I actually used?
- What was unclear or missing?
- Did I have to go to the catalog to recover context that should
  have been in the file?
- Was there ambiguity I had to resolve by guessing?

Gaps go in `findings/` like any other friction.

---

## 6. Bootstrap (Phase 0)

Run once, by the orchestrator (or the user) before launching the
curator. None of this is persona work — this is infrastructure setup
that must complete *before* any persona starts. A failure here is a
Phase 0 finding and may block the test entirely.

### Why Phase 0 is not the Curator

The Curator persona inherits a bootstrapped catalog rather than
creating it. This is a deliberate choice, not an oversight, and the
spec calls it out so future readers don't relitigate the question:

- **The test measures user experience, not infrastructure setup.**
  `load-cifar10` is mechanical (one CLI invocation) and reveals no
  judgment-laden friction. A Curator arc that includes bootstrap
  dilutes the persona's role away from their actual value-add:
  *deciding what dataset variants serve downstream personas* —
  audit, curation, naming, versioning, handoff documentation.
- **`load-cifar10` is the test harness, not the test subject.**
  Its bugs were shaken out in earlier runs (B17 stratified sampling,
  the Toronto migration). Re-running it through a persona adds no
  new signal.
- **In real organizations, role overlap varies.** Some shops have
  separate data-engineering and data-curation roles. Others combine
  them in one person. The persona is an abstraction, not a roleplay
  — treat Phase 0 as "the data-engineering hat" the same human (or
  a different one) wears before the curation hat goes on. The
  abstraction holds either way.
- **Bootstrap failure modes are still surfaced.** Phase 0 step 4
  runs the same cross-channel verification (§3.4) that personas do.
  If `load-cifar10` breaks the catalog or the MCP surface lies about
  what it produced, the discrepancy is a Phase 0 finding before any
  persona starts.

### 6.1 What Phase 0 produces (the persona inputs)

By the time Phase 0 is done, the following is true:

- A single shared git worktree exists at
  `../deriva-ml-model-template-e2e` on branch `e2e-test/<YYYY-MM-DD>`,
  cut from `main` of this repo. All persona work happens here (§3.5).
- A fresh catalog exists at `localhost` named `e2e-test-<YYYYMMDD>`.
- The catalog has the cifar10 domain schema populated by `load-cifar10`
  (Image table, vocabularies including `Image_Class`, the built-in
  datasets, ground-truth `Image_Classification` feature values).
- `src/configs/deriva.py` in the e2e worktree has been edited so
  `default_deriva` points at the new catalog id (a `[E2E-DROP]`
  commit). `src/configs/datasets.py` has been edited with the
  loader-produced RIDs (also `[E2E-DROP]`). The base config files are
  edited *directly* — `configs/dev/` no longer exists in this
  template; the dev-overlay pattern was retired with the 2026-05-21
  rewrite.
- `tacit-knowledge.md` contains a single "Bootstrap" entry
  recording catalog name, dataset RIDs, the `load-cifar10` invocation
  that created them, and the sibling versions of the platform stack
  at run-start.
- The dev-localhost MCP container is rebuilt against the current
  sibling versions and Claude Code's MCP server connection is
  restarted. The **OAuth flow is completed as step 1 of §6.2** —
  it's the first action Phase 0 performs so the orchestrator fails
  fast if auth can't be established. `claude mcp list` should
  report `dev-localhost: ... - ✓ Connected` after step 1, and the
  `deriva_ml_*` tools should be callable. Without this, Phase 0
  part E (cross-channel verification) and every persona's indirect-
  channel work is blocked.

### 6.2 Phase 0 steps (in order)

**Preflight first, then authentication.** P0 begins with a sync
audit (step 0) and an MCP-auth handshake (step 1). Both are fail-
fast gates: if the workspace is drifted or auth can't be
established, no further P0 work is reachable.

0. **Sync audit (preflight).** Verify the workspace is internally
   consistent before doing any setup work. The 2026-05-21 e2e run
   surfaced two distinct kinds of drift the orchestrator can't
   recover from later: stale Claude Code plugins (skill docs were
   one minor version behind the API they document) and a stale MCP
   container image (a deriva-mcp-test image built against an older
   deriva-ml). Both look healthy on inspection (plugin lists,
   `docker ps`) yet ship the wrong code.

   Run these checks in order; bail at the first failure rather
   than papering over it:

   a. **Repo state.** For each of `deriva-ml`, `deriva-mcp-core`,
      `deriva-ml-mcp`, `deriva-skills`, `deriva-ml-skills`,
      `deriva-ml-model-template`:
      ```
      git -C <repo> fetch --prune origin
      git -C <repo> status -b --short      # expect: clean, == origin/main
      git -C <repo> log --oneline -1 main  # note the SHA
      ```
      No repo should have uncommitted changes or be ahead/behind
      its origin/main.

   b. **Stale local branches.** For each repo above, list local
      branches whose upstream is `gone` (PR was merged + branch
      deleted on GitHub). They are harmless but accumulate, and
      `git fetch --prune` will mark them:
      ```
      git -C <repo> for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads \
        | awk '$2 ~ /gone/ {print $1}'
      ```
      Delete any whose tip is also in main (`git branch -d`).

   c. **Lockfile freshness.** In `deriva-ml-mcp`,
      `deriva-ml-model-template`, and `deriva-ml-skills`:
      ```
      uv sync --upgrade-package deriva-ml
      uv sync --upgrade-package deriva   # deriva-py
      ```
      If either of these produces a diff to `uv.lock`, commit it as
      `chore(deps): sync ...` and push before proceeding. The run
      becomes unreconstructable if the lockfile drifts mid-test.

   d. **Local venv sanity.** From the model-template:
      ```
      uv run python -c "
      import deriva_ml, inspect
      from deriva_ml.dataset.split import split_dataset
      print(deriva_ml.__version__)
      print('execution param:', 'execution' in inspect.signature(split_dataset).parameters)
      "
      ```
      Version should match the lockfile pin; the `execution` param
      check is a fast sentinel that catches "split_dataset signature
      drift" — a stand-in for "is the venv on the new contract".

   e. **Claude Code plugin freshness.** The skill docs that
      Curator / Developer / Analyst will lean on must match the
      API they describe.
      ```
      claude plugin list | grep deriva
      ```
      For each `deriva*@deriva-plugins` entry, compare its version
      against the latest tag on origin:
      ```
      git -C deriva-skills    tag --list | sort -V | tail -1
      git -C deriva-ml-skills tag --list | sort -V | tail -1
      ```
      If installed < latest tag, run
      `claude plugin update <name>@deriva-plugins` and restart
      Claude Code before continuing.

   f. **MCP container freshness.** This is the trap. The compose
      file declares two services that build distinct images
      (`deriva-mcp` and `deriva-mcp-test` — the latter extends the
      former but yields a *separate* tag), and rebuilding one
      does NOT rebuild the other. Verify the actual running test
      image:
      ```
      docker exec deriva-mcp-test python -c '
      import deriva_ml, importlib.metadata as md
      print("deriva-ml:    ", deriva_ml.__version__)
      print("deriva-ml-mcp:", md.version("deriva-ml-mcp"))
      '
      ```
      Both versions must match the SHAs from step (a). If either
      lags:
      ```
      cd deriva-docker/deriva
      docker compose --env-file ~/.deriva-docker/env/localhost.env \
                     build --no-cache deriva-mcp-test
      docker compose --env-file ~/.deriva-docker/env/localhost.env \
                     up -d --force-recreate deriva-mcp-test
      ```
      Re-run the version check before proceeding to step 1. **Do
      not rely on `--no-cache deriva-mcp` to rebuild the test
      image** — they are separate tags. Always name the
      `-test` service explicitly.

   g. **`main` is at template state.** The persona arcs start from
      a worktree cut from `main`, so `main` itself must be in its
      pristine, no-prior-run state. Every previous multipersona
      run produced `[E2E-DROP]` commits that mutate
      `src/configs/deriva.py`, `src/configs/datasets.py`, and
      `tacit-knowledge.md`. Wrap-up step 4 of the test plan
      drops those commits when cherry-picking back to `main`, but
      the bookkeeping is easy to get wrong, and a poisoned `main`
      means the *next* multipersona run inherits last run's
      catalog id, dataset RIDs, and Bootstrap note. The persona
      cannot detect the drift — they just see a stale catalog ref
      in the config they're "starting fresh" with.

      Check each file is at its template state:

      - `src/configs/deriva.py` should have `catalog_id=0` in
        `default_deriva` (the placeholder). Anything else means a
        prior E2E-DROP leaked through.
      - `src/configs/datasets.py` should have empty placeholder
        list literals for every dataset group, not RID strings.
        The docstring at the top of the file calls itself out as
        "intentionally empty by default."
      - `tacit-knowledge.md` should be the template header
        only — three short lines of intro + a horizontal-rule
        separator + nothing else. No "Bootstrap" entry, no
        per-persona decision logs, no model-tuning notes.

      Fast cross-check (ignores commented-out example lines in
      `datasets.py`, which legitimately contain RID strings inside
      `# DatasetSpecConfig(rid="..."` examples):
      ```
      grep -E "^[^#]*catalog_id=[1-9]" src/configs/deriva.py \
        && echo "FAIL: deriva.py has a real catalog_id"
      grep -E "^[^#]*rid=\"[0-9]" src/configs/datasets.py \
        && echo "FAIL: datasets.py has RIDs filled in"
      [ "$(wc -l < tacit-knowledge.md)" -gt 17 ] \
        && echo "FAIL: tacit-knowledge.md is non-template"
      ```

      If any check fails: `git log --oneline -- <path>` to find
      the offending E2E-DROP commit, then `git revert <sha>` (or
      `git restore --source=<known-good-sha> <path>` if reverting
      is messy because of subsequent template-evolution commits)
      and push to origin/main *before* proceeding.

   If any sub-check (a-g) fails, fix it and re-run from (a). The
   cost of bailing here is minutes; the cost of running a
   multipersona arc against drifted siblings or a poisoned `main`
   is the entire run.

1. **Authenticate the dev-localhost MCP server (OAuth).** The
   `dev-localhost` MCP server uses a browser-based OAuth flow that
   must be completed once per Claude Code session before its tools
   become available. P0 starts here so the orchestrator (or the
   user) fails fast: if auth can't be completed, the rest of P0
   produces nothing usable.

   Prerequisites (none of these are P0 steps themselves; they're
   workspace setup the orchestrator inherits or completes outside
   the test):

   - The dev-localhost MCP container is built and running. If it
     isn't, run `cd deriva-docker/deriva && docker compose up -d
     deriva-mcp` (or the equivalent for your local rig). For an
     e2e run that needs fresh sibling versions, rebuild via
     `docker compose build --no-cache deriva-mcp` first.
   - The MCP server is registered with Claude Code (it appears in
     `claude mcp list`). If it isn't, follow the deriva-docker
     setup notes to register it.

   Procedure:

   a. Confirm the server is registered and its current state:
      ```
      claude mcp list
      ```
      Expected line:
      `dev-localhost: https://localhost/mcp (HTTP) - ! Needs authentication`
      If it says `Connected` already, skip to (d). If `Failed to
      connect`, the container isn't healthy — return to the
      prerequisites above and resolve before continuing.

   b. Trigger the authorization URL:
      ```
      mcp__dev-localhost__authenticate
      ```
      The tool prints an `https://localhost/authn/authorize?...` URL
      and a fallback path (`mcp__dev-localhost__complete_authentication`)
      for the case where the redirect lands on a port nothing is
      listening on.

   c. Open the URL in a browser, sign in, and complete the consent
      flow. The page redirects to
      `http://localhost:8080/callback?code=...&state=...`. Two
      outcomes:
      - **Page loads cleanly.** The MCP server received the code,
        exchanged it for a token, and the `deriva_ml_*` and other
        tools become available automatically. The session emits a
        notification listing the newly-available deferred tools.
      - **Browser shows "connection error".** Nothing listened on
        port 8080. Copy the full URL from the browser's address bar
        and call `mcp__dev-localhost__complete_authentication` with
        it to finish the handshake.

   d. Sanity-check: a follow-up `claude mcp list` should now show
      `dev-localhost: ... - ✓ Connected`. Confirm a representative
      tool works:
      ```
      mcp__dev-localhost__get_catalog_info(hostname=localhost, catalog_id=1)
      ```
      (Any catalog id is fine — even a missing one returns a
      meaningful error rather than an auth failure.)

   Notes:

   - The orchestrator session's OAuth token is **not inherited by
     sub-agents spawned via the `Agent` tool**. The 2026-05-21 run
     observed that sub-agents DID inherit auth (the dev-localhost
     tools were immediately available to personas without re-auth);
     verify this holds for your run by including a check in the
     persona's startup instructions.
   - If auth expires mid-run (long sessions), tool calls start
     returning auth errors. Re-run (b) and (c).
   - This step is per-Claude-Code-session, not per-catalog. If you
     run a second e2e on the same day in the same session, you don't
     need to re-auth.

2. **Create the shared e2e worktree.** Pick the run date as
   `<YYYY-MM-DD>` (all later artifacts key off this) and:
   ```
   git -C deriva-ml-model-template worktree add \
       ../deriva-ml-model-template-e2e -b e2e-test/<YYYY-MM-DD>
   ```
   Refuse to proceed if a prior catalog at the target name exists
   unless the user explicitly says delete-and-reuse. If an
   `e2e-test/<YYYY-MM-DD>` branch already exists, abort or use a
   suffixed date — never overwrite.

   Immediately re-verify the *worktree's* template-state files
   match `main` (step 0(g) checked `main` itself; this checks the
   worktree the personas will actually inhabit):
   ```
   cd ../deriva-ml-model-template-e2e
   grep -E "^[^#]*catalog_id=[1-9]" src/configs/deriva.py \
     && echo "FAIL: deriva.py is non-template in the worktree"
   grep -E "^[^#]*rid=\"[0-9]" src/configs/datasets.py \
     && echo "FAIL: datasets.py has RIDs in the worktree"
   [ "$(wc -l < tacit-knowledge.md)" -gt 17 ] \
     && echo "FAIL: tacit-knowledge.md is non-template in the worktree"
   ```
   None of the FAIL lines should print. If any does, something
   between `main`'s tip and the new branch's tip is wrong —
   abort and inspect (`git diff main..e2e-test/<YYYY-MM-DD> -- \
   src/configs/ tacit-knowledge.md` will be empty for a
   clean cut).

3. **Verify clean state.** Model template `main` is at the latest
   commit; no stale `e2e-test/*` worktrees or branches conflict;
   prior test catalogs (if any) are either kept intentionally or
   deleted with user confirmation.

4. **Refresh sibling versions.** `uv sync --upgrade` inside the e2e
   worktree to pick up the latest `deriva-ml`, `deriva-ml-mcp`,
   `deriva-mcp-core`, `deriva-skills`, `deriva-ml-skills` versions.
   Confirm versions match their `main` HEADs (or the run will pin to
   stale versions and the run is not reconstructable from sibling
   tags alone). If sibling versions have advanced enough to need a
   container rebuild, rebuild the dev-localhost MCP container
   against those versions and restart Claude Code's MCP servers,
   then **re-do step 1** to re-authenticate the freshly restarted
   server.

5. **Phase 0 part A — create the catalog.** From the e2e worktree:
   ```
   uv run load-cifar10 --hostname localhost \
       --create-catalog e2e-test-<YYYYMMDD> --phase schema
   ```
   This creates the catalog and the domain schema only. Capture the
   numeric catalog id printed by the loader — every later step
   needs it.

6. **Phase 0 part B — update `deriva.py`.** Edit
   `src/configs/deriva.py` in the e2e worktree so the `default_deriva`
   entry has `hostname="localhost"` and `catalog_id=<new_id>`.
   Commit on `e2e-test/<YYYY-MM-DD>` with an `[E2E-DROP]` marker so
   the commit can be dropped from `main` at wrap-up. After this step,
   `uv run deriva-ml-run` (and `deriva-ml-run-notebook`) in the e2e
   worktree default to the new catalog with no CLI overrides.

7. **Phase 0 part C — load assets and datasets.** Re-invoke the
   loader against the now-existing catalog:
   ```
   uv run load-cifar10 --hostname localhost \
       --catalog-id <new_id> --num-images 500 --phase images
   uv run load-cifar10 --hostname localhost \
       --catalog-id <new_id> --num-images 500 --phase datasets
   ```
   Run the phases separately (not `--phase all`) so a failure in
   `datasets` doesn't require re-uploading the images. Each phase is
   intended to be idempotent against partial state, though the
   2026-05-21 run found this guarantee imperfect — see Phase 0
   findings 04, 05.

8. **Phase 0 part D — update `datasets.py`.** Edit
   `src/configs/datasets.py` in the e2e worktree, replacing the empty
   placeholder lists with the dataset RIDs the loader produced.
   Discover them with `ml.find_datasets()` from a quick Python
   session against the new catalog. Commit on
   `e2e-test/<YYYY-MM-DD>` with an `[E2E-DROP]` marker.

9. **Phase 0 part E — validate (cross-channel).** Run the same
   cross-channel verification (§3.4) that personas run — both via
   direct deriva-ml inspection AND via the MCP tools
   (`deriva_ml_list_datasets`, `deriva_ml_list_features`,
   `deriva_ml_list_vocabulary_terms`). The two channels must agree
   on:
   - Catalog exists at the expected name + the numeric catalog id
     recorded in `deriva.py`.
   - The expected dataset hierarchy is present, and the RIDs recorded
     in `datasets.py` resolve via both channels.
   - `Image_Classification` feature values are populated for the
     labeled partitions (count > 0).
   - Class distribution is balanced across all 10 CIFAR-10 classes
     (post-#15 fix; not the pre-fix bird+ship-dominant skew).

   If the two channels disagree, that's a Phase 0 finding (likely an
   MCP-side bug, given the May 2026 pattern). If either channel
   fails any of the listed checks, that's also a Phase 0 finding. The
   test either aborts or proceeds with the finding documented and
   the Curator's success criteria adjusted accordingly. User decides.

10. **Seed `tacit-knowledge.md`** with the "Bootstrap" entry — a
   short note recording what was created in parts A-C, what the
   ground state looks like, the new catalog id, the
   `load-cifar10` invocations, and the sibling versions
   (commit SHAs or release tags) so the run is reconstructable.

11. **Audit Claude Code skill registry.** Verify which skills are
   auto-fire vs slash-only by reading frontmatter; this is the
   ground state the personas will see. Mismatches against the
   personas' expected skill list go in `findings/setup/` as a
   pre-curator finding bucket.

12. **Mode selection.** Ask the user — interactive or autonomous?
    (See §3.1.)

13. **Launch curator** in the shared e2e worktree with their persona
    prompt. (Developer and Analyst launch later, sequentially, in the
    *same* worktree — there are no per-persona worktrees in this
    revision of the spec; see §3.5.)

### 6.3 What's *not* Phase 0

- `load-cifar10` itself. The script lives in `src/scripts/load_cifar10.py`
  and is treated as platform code, not test code. If it breaks during
  step 5 (Phase 0 part A) or step 7 (Phase 0 part C), that's a
  finding against the script (or against `deriva-ml` if the failure
  is in a library call), not test-design feedback.
- Schema or vocabulary creation beyond what `load-cifar10` does. Any
  curation work belongs to the Curator persona, not bootstrap.
- Feature populations beyond ground-truth. The Curator is the persona
  who decides whether additional features are needed downstream.

---

## 7. Wrap-up

When all three personas finish (or the user aborts):

1. **Verify final state of catalog** via direct deriva-ml inspection.
   Persona findings + the catalog state should agree on what's in
   the catalog.
2. **Generate the friction map** at `findings/REPORT-<YYYY-MM-DD>.md`
   per §4.4.
3. **User reviews and decides** per-finding disposition:
   - Promote to GitHub issue (and which repo).
   - Fix inline now via a fix-pass agent.
   - Defer (note in the report).
   - Discard (note in the report with reason).
4. **Cherry-pick genuine template fixes** from the shared
   `e2e-test/<YYYY-MM-DD>` branch back to `main` of the model
   template. Test-mutation commits (anything tagged `[E2E-DROP]`,
   e.g., the `deriva.py` and `datasets.py` repointing commits) are
   dropped, not cherry-picked.
5. **Worktree teardown** with explicit user confirmation: `git
   worktree remove ../deriva-ml-model-template-e2e`, then
   `git branch -D e2e-test/<YYYY-MM-DD>`.
6. **Catalog disposition** with explicit user confirmation: delete
   or preserve.

---

## 8. What the report should let the user do

The friction map at the end of the run is the test's actual output.
A successful run is one where the user can answer, in 15 minutes of
reading the report:

- For each persona, what was the worst thing about being them?
- Which findings are technical bugs and which are platform design
  questions?
- What's the smallest set of changes that would meaningfully
  improve the next user's experience?
- Is the platform ready for an external user, or do we have more
  rough-edge polishing first?

If the report doesn't support those questions, the test format
itself is broken and that's its own finding worth investigating.

---

## 9. Things that are NOT in this spec

- **What specific findings will look like.** That's the test's
  output. Pre-specifying would defeat the purpose.
- **How to fix any specific bug.** The fix-pass is a separate
  workflow.
- **Multi-host scenarios, performance benchmarks, schema migration
  exercises.** Out of scope; tracked elsewhere.
- **Concurrent persona execution.** Sequential only for this run.
  Concurrent execution is a future variant once the sequential
  baseline reveals the cross-persona friction patterns.
- **Persona other than the three named.** Platform integrator,
  reviewer, ops, etc. — each is worth a run, but not this run.

---

## Quick reference

| Question | Answer |
|---|---|
| Where does this spec live? | `docs/test-plans/2026-05-20-e2e-multipersona.md` |
| Where do findings go? | `findings/<persona>/<NN>-<slug>.md` in the shared e2e worktree |
| Where does the persona handoff happen? | `tacit-knowledge.md` (project root, in the shared e2e worktree) |
| Who creates the catalog? | Phase 0 bootstrap (§6), via `load-cifar10` — *before* any persona runs |
| What's the catalog name? | `e2e-test-<YYYYMMDD>` (chosen at run start) |
| Cross-channel verification? | Each persona must verify, before declaring arc complete, that direct deriva-ml inspection of the catalog matches what the skills/MCP tools said happened. Disagreement is a finding (§3.4). |
| Mode flag? | Interactive (checkpoint per persona) or Autonomous (final report only); chosen at start |
| Branch / worktree? | Single shared branch `e2e-test/<YYYY-MM-DD>` cut from `main`, checked out at `../deriva-ml-model-template-e2e`. All three personas run sequentially in this one worktree (see §3.5). |
| Final artifact? | `findings/REPORT-<YYYY-MM-DD>.md` |
| Who fixes bugs surfaced? | A fix-pass agent (post-run or between phases in interactive). Personas never fix mid-arc. |
