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
2. **Test `experiment-decisions.md` as a real knowledge-transfer
   artifact.** Each persona writes to it during their work; the
   next persona reads from it before starting. Gaps in the handoff
   are findings — about the file, the prior persona's writing, or
   the `maintain-experiment-notes` skill itself.
3. **Surface bugs and rough spots** *as a byproduct* of the personas
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

> *"I have raw data I want to make available as well-versioned,
> well-labeled datasets that downstream users can train on. I care
> about provenance and reproducibility. I don't train models."*

**Goal:** Bootstrap a fresh test catalog from the CIFAR-10 source,
register the canonical datasets (split, labeled, small, etc.), make
sure ground-truth labels are populated, and document the catalog's
shape for downstream personas. Create a non-trivial dataset variant
(curated subset or new split) to demonstrate the lifecycle.

**Primary skills/tools:** `setup-ml-catalog`, `dataset-lifecycle`,
`create-feature` (in query mode), `manage-vocabulary`,
`maintain-experiment-notes`.

**Success criteria:**
- Catalog exists at `localhost`, named `e2e-test-<YYYYMMDD>`.
- All built-in datasets present and versioned.
- `Image_Classification` ground-truth feature is populated for at
  least the labeled partitions.
- At least one new dataset (a curated subset or new split) created
  via `dataset-lifecycle`.
- `experiment-decisions.md` contains entries explaining: why this
  catalog, what the canonical splits are for, which dataset variant
  was created and why.
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
new variant beat the baseline?" sense), `maintain-experiment-notes`.

**Success criteria:**
- At least two distinct training runs completed, with weights and
  predictions uploaded as `Execution_Asset` rows.
- At least one multirun (e.g., `quick_vs_extended` or `lr_sweep`)
  completed; parent and child executions correctly linked.
- New experiment config registered in `src/configs/dev/experiments.py`
  if the developer needed one beyond the existing ones.
- `experiment-decisions.md` contains entries explaining: which
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
could read in 5 minutes.

**Primary skills/tools:** `compare-model-runs`, `run-notebook`,
`execution-lifecycle` (for executing the notebook with provenance),
`maintain-experiment-notes`.

**Success criteria:**
- A ranking of the developer's executions by at least one metric.
- One executed analysis notebook (e.g., `notebooks/roc_analysis.ipynb`
  or a new one) producing plot asset(s) + a summary CSV asset.
- A short markdown report under `docs/reports/` (created by this
  persona) summarizing the comparison, what's in the catalog now,
  and any caveats.
- `experiment-decisions.md` contains entries explaining: which runs
  were compared and why, what metric was chosen, how surprises
  (if any) were interpreted.

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
and the experiment-decisions handoff. The user can redirect, ask
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

### 3.2 Decision rights — what an agent can decide alone

The personas need clear ground rules about when to act and when to
escalate. The rules differ by mode.

| Decision | Interactive | Autonomous |
|---|---|---|
| Which existing dataset/feature/config to use for an obvious task | Decide | Decide |
| Reasonable parameter choice (split ratio, learning rate, epoch count) within typical range | Checkpoint summary | Decide; note the choice in `experiment-decisions.md` |
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
   this spec, and (critically) `experiment-decisions.md` if it
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
   them to `experiment-decisions.md` via `maintain-experiment-notes`.
   At minimum: one entry per major decision (dataset choice, split
   strategy, model config selection, metric choice).
5. **Write handoff.** At end of arc, persona appends a "handoff
   summary" section to `experiment-decisions.md` named for the
   next persona, describing what's ready and what's pinned. This
   is the explicit knowledge-transfer step.
6. **Produce arc summary.** A markdown summary of what was done,
   findings raised, decisions captured, and success-criteria
   status (which met, which not, why). In interactive mode this
   is the exit checkpoint; in autonomous mode it feeds the final
   consolidated report.

### 3.4 Multi-agent setup

Each persona runs as its own Agent-tool invocation with a dedicated
system prompt drawn from §2. Each gets its own git worktree branched
from `main` of the model template (worktrees prevent the inter-agent
file-stomping the May run hit). The catalog is shared.

Branch / worktree convention:

```
git worktree add ../deriva-ml-model-template-curator e2e-test/<YYYY-MM-DD>-curator
git worktree add ../deriva-ml-model-template-developer e2e-test/<YYYY-MM-DD>-developer
git worktree add ../deriva-ml-model-template-analyst e2e-test/<YYYY-MM-DD>-analyst
```

Each branch is cut from `main` at run-start. The curator's worktree
is the one that touches `dev/*_localhost.py` for catalog repointing;
the developer's worktree picks up those changes via merge from the
curator's branch before starting. Same chain analyst-from-developer.
This makes the handoff observable in git, not just in
`experiment-decisions.md`.

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
`experiment-decisions.md`? Specific examples of what carried over
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

## 5. `experiment-decisions.md` as test artifact

The file lives in the project root and is tracked in git. Each
persona is expected to:

- **Read** the file at startup, before doing any work, to inherit
  prior personas' context.
- **Write** to it via `maintain-experiment-notes` at decision
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
curator. None of this is persona work.

1. **Choose date**: pick the run date as `<YYYY-MM-DD>`. All branches,
   catalog name, journal, and report use this.
2. **Verify clean state**: model template `main` is clean, no
   stale `e2e-test/*` branches conflict, prior test catalogs (if
   any) are either kept intentionally or deleted with user
   confirmation.
3. **Refresh sibling versions**: `uv sync` in the workspace; verify
   `deriva-ml`, `deriva-ml-mcp`, `deriva-mcp-core` versions; rebuild
   the dev-localhost MCP container against the current sibling
   versions; restart Claude Code's MCP servers.
4. **Audit Claude Code skill registry**: verify which skills are
   auto-fire vs slash-only by reading frontmatter; this is the
   ground state the personas will see. Mismatches against the
   personas' expected skill list go in `findings/setup/` (a
   pre-curator finding bucket).
5. **Create worktrees**: one per persona, per §3.4.
6. **Mode selection**: ask the user — interactive or autonomous?
7. **Launch curator** in their worktree with their persona prompt.

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
4. **Cherry-pick genuine template fixes** from persona branches
   back to `main` of the model template. Test-mutation commits
   on the persona branches (e.g., `dev/*_localhost.py` repointing)
   are dropped.
5. **Worktree teardown** with explicit user confirmation: `git
   worktree remove` each, `git branch -D` each.
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
| Where do findings go? | `findings/<persona>/<NN>-<slug>.md` per worktree |
| Where does the persona handoff happen? | `experiment-decisions.md` (project root) |
| What's the catalog name? | `e2e-test-<YYYYMMDD>` (chosen at run start) |
| Mode flag? | Interactive (checkpoint per persona) or Autonomous (final report only); chosen at start |
| Branch naming? | `e2e-test/<YYYY-MM-DD>-<persona>`, branched from `main` |
| Final artifact? | `findings/REPORT-<YYYY-MM-DD>.md` |
| Who fixes bugs surfaced? | A fix-pass agent (post-run or between phases in interactive). Personas never fix mid-arc. |
