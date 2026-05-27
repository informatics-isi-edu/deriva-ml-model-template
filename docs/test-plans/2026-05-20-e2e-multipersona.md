# End-to-End Multi-Persona Platform Scenario

**Author:** Carl Kesselman (with Claude)
**Date:** 2026-05-20
**Supersedes:** `docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md`
**Status:** Approved for execution.

---

## 1. What the team is doing

A three-person team is handed a catalog with some initial data and
asked to take it from raw arrival to a usable analytical result.
That work has three natural phases, each owned by one persona:

1. **Understand the data.** Check that it's clean. Notice
   limitations, peculiarities, gaps, things that don't pass a smell
   test. Prepare what's there for the people downstream.
2. **Build models against the data.** Try alternative model
   variants and alternative parameterisations. Confirm that the
   modelling pipeline produces a result the next person can use.
3. **Evaluate what the models actually do.** Look at predictions
   against ground truth, build pictures (ROC curves, confusion
   matrices, per-class breakdowns), and summarise the findings in a
   form a non-ML collaborator can read.

The team's collective deliverable is a coherent story: this is
what was in the catalog, this is what we built on top of it, this
is what it tells us. Each persona owns one phase; together they
produce the story end-to-end.

### What this document is

This document is the **scenario** half of a two-part exercise. It
describes the work the team does — who the personas are, what
they have to work with, how they hand off — without prescribing
how they do it. Each persona's *skills* drive the work; this
document just gives them the setting and the motivation.

The **evaluation** half — what counts as a finding, whether the
catalog matches what the skills claimed happened, whether the
team produced a useful story, whether the personas reached for
the right skills — lives in a separate evaluator rubric that
runs *after* the team finishes. Keeping the halves separate is
deliberate: the personas should do their work the way a real
team would, without writing-to-the-test. The evaluator looks
cold at the artifacts the team produced and forms its findings
independently.

### Non-goals

- Coverage of every model config / experiment combination.
- Performance benchmarking.
- Multi-host / cluster scenarios — `localhost` only.
- Inline bug-fixing during persona arcs. The evaluator's findings
  feed a separate fix-pass workflow.

---

## 2. Personas

Three personas — a Curator, a Modeler, and an Analyst — run
sequentially against a shared catalog. Each section below describes
the kind of professional the persona is and what they typically
care about. It does **not** prescribe what work they do, in what
order, or what artifacts they produce; those follow from the
persona's interests and from the catalog state they encounter.

### 2.1 The Curator

The Curator is a domain or data person who has been handed a
freshly-bootstrapped catalog. Their orientation is **exploratory
and skeptical**: they want to *understand what's in the catalog*,
check that the data makes sense, notice what's missing or
surprising, and prepare what they find for the people who'll work
with it next. They aren't doing modelling — they're characterising
the substrate everyone else will build on.

What a Curator typically wants to know:

- What datasets exist, what's in each, how they relate to each
  other.
- Whether the ground-truth labels are sensible (class balance,
  obvious errors, gaps).
- Whether the canonical splits (training / testing / labeled /
  small / etc.) actually represent what their names imply.
- Whether anything in the catalog needs attention before downstream
  work — missing data, oddities, things that don't pass a smell
  test.
- What downstream personas (the modeler, the analyst) might need
  that doesn't yet exist.

The Curator might end up creating a new dataset variant, fixing
class-balance issues by curating a stratified subset, adding a
vocabulary term that turns out to be missing, or writing down a
gotcha they noticed about how a particular partition was
constructed. They might also conclude the catalog is in good shape
as-is and leave it alone. Both are reasonable outcomes — the
exploration itself is the work.

**What they typically reach for:** `dataset-lifecycle` for inspecting
and creating datasets; `manage-vocabulary` for vocabulary work;
`browse-erd` and `using-deriva-mcp` for catalog exploration;
direct deriva-ml Python for checks that don't have a skill route;
`capture-tacit-knowledge` for recording what they decided and why.

### 2.2 The Modeler

The Modeler is an ML practitioner who wants to **try things and
see if they work**. They aren't aiming for a publishable result;
they're stress-testing the modelling pipeline against the data the
Curator handed them. They want to confirm that training runs
launch cleanly, that the pipeline produces something that looks
like learning, and that the outputs land in the catalog the way
they should. The platform itself is half the subject of their
inquiry — does it support a normal modelling workflow without
getting in the way?

What a Modeler typically wants to know:

- Which datasets in the catalog are appropriate for training (vs
  for held-out evaluation, vs unlabeled).
- Whether the training pipeline runs end-to-end against a real
  dataset, not just a fixture.
- What happens when they vary hyperparameters — does the output
  reflect the variation, or do all runs look the same?
- Whether the predictions, weights, and training logs that come
  out the other end land in the catalog with provenance the next
  persona can use.
- Whether reproducibility-affecting features (seeds, deterministic
  ops) actually work the way they're advertised.

The Modeler will typically run a few training executions with
different parameters — a smoke run on a small dataset, a couple
more substantive runs on bigger ones — and confirm that the
results differentiate. They might add an experiment config if the
existing ones don't cover what they want to try. They aren't
trying to win a benchmark; they're trying to convince themselves
the pipeline works.

**What they typically reach for:** `execution-lifecycle` and
`configure-experiment` for running training; `write-hydra-config`
for new variants; `model-development-workflow` and `new-model` for
broader orientation; `compare-model-runs` to look at what they
produced; `capture-tacit-knowledge` for decisions.

### 2.3 The Analyst

The Analyst is a domain expert. They want to **understand what
the models are doing** — what they got right, what they got
wrong, where the confusion lies. They aren't an ML person; they
aren't going to retrain anything. They consume what the Modeler
produced and form a judgment about it. Their natural mode is
inquiry: form a hypothesis, look at the data, see if the picture
matches the hypothesis, repeat.

What an Analyst typically wants to know:

- Which of the Modeler's runs performed best, and by what measure
  (top-1 accuracy isn't always the right question).
- Where the models are confident and right, where they're
  confident and wrong, where they're uncertain.
- How performance varies across classes (or other domain-relevant
  partitions).
- Whether the model's behavior matches their domain intuitions —
  classes that should be confusable being confused, classes that
  shouldn't be confused being separable.
- Whether the catalog supports the kind of analysis they want to
  do, or whether the data shape gets in the way.

The Analyst will typically rank the Modeler's runs by one or more
metrics, build some pictures (ROC curves, confusion matrices,
per-class breakdowns), and write up what they found in a form a
non-ML collaborator could read. They'll touch the data the Modeler
trained on directly — joining predictions to ground truth,
denormalizing dataset members for plotting, reconciling what the
catalog says against what the prediction files say.

**What they typically reach for:** `compare-model-runs` for
ranking; `run-notebook` for analysis pipelines; `execution-lifecycle`
for capturing the analysis with provenance; `dataset-lifecycle`
(especially the denormalize section) for materialising joined
views; `capture-tacit-knowledge` for interpretive judgments.

### 2.4 Inputs the Curator inherits

Phase 0 (§5) bootstraps a fresh catalog before the Curator starts.
At that point:

- A catalog exists at `localhost` named `e2e-test-<YYYYMMDD>`.
- The domain schema is populated by `load-cifar10` (Image table,
  vocabularies including `Image_Class`, the built-in datasets,
  `Image_Classification` ground-truth feature values for labeled
  partitions).
- `src/configs/deriva.py` in the e2e worktree has `default_deriva`
  pointing at the new catalog id (an `[E2E-DROP]` commit).
- `src/configs/datasets.py` carries the loader-produced RIDs
  (also `[E2E-DROP]`).
- `tacit-knowledge.md` has a single "Bootstrap" entry recording
  what was created and the sibling versions of the platform stack.

### Persona ordering and dependencies

Curator → Modeler → Analyst. Strictly sequential. The Modeler
cannot start until the Curator has finished (the Modeler may want
to use datasets the Curator created); the Analyst cannot start
until the Modeler has finished (the Analyst needs runs to compare).
Each persona inherits whatever state the previous one left in the
catalog, in the worktree, and in `tacit-knowledge.md`.

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
without checkpoints. The evaluator runs after the personas finish
and produces its evaluation report. This mode is for repeat runs
once the user trusts the personas, for overnight execution, or for
batch comparison of multiple platform versions.

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
| Create a new dataset / feature / config that serves the persona's professional motivation | Checkpoint, ask first | Decide if it serves the persona's interests; note rationale |
| Destructive operations (delete catalog, drop schema, force-push, rm -rf working dir) | Always ask | Always ask — abort the persona if blocked |
| Schema migrations (new column, FK change, drop table) | Always ask | Always ask — abort the persona if blocked |
| Fix a bug encountered mid-arc | Always ask | Never. File a finding and route around if possible. |
| Stop the arc because the platform won't support what the persona wants | Checkpoint, explain | File a finding; produce whatever summary the persona can with what's been done |

The bright lines: destructive operations and schema migrations always
require explicit user authorization, regardless of mode. Persona
agents never fix bugs mid-arc — that's a separate fix-pass.

### 3.3 What a persona's arc looks like

Each persona enters their arc with the project's `CLAUDE.md`,
their own section of this document, and whatever the previous
persona left in the catalog and the worktree. The persona's
*skills* drive how they work from there — what to consult, when
to record something, how to verify their assumptions. This
document deliberately does not prescribe a checklist; an arc
that goes well looks like a competent professional doing their
job, not like a script being executed.

At the end of their arc, the persona produces a short summary of
what they did. Whatever else they leave behind is determined by
the work, not by this document.

### 3.4 Multi-agent setup

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

This worktree is created in Phase 0 step 0 (§5.2) before any persona
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
transfer artifact (`tacit-knowledge.md`) implicit in the
orchestrator's merging discipline rather than naturally available
to the next persona. Single-worktree restores the handoff as the
straightforward chain it should be: persona N writes,
persona N+1 reads from the same files.

**Concurrent variant (future).** If a future run ever wants to
exercise concurrent personas (e.g., Curator on labeling while the
Modeler trains on an earlier dataset version), reintroduce
per-persona worktrees and treat each merge as an explicit
synchronization point. Out of scope here.

---

## 4. Findings written during an arc

When something gets in a persona's way mid-work — a bug, a confusing
error message, a tool that wasn't there, a skill that didn't route
to what they needed, an output that doesn't match what they
expected — the persona may record it as a finding so the
evaluator (and the eventual fix-pass) can find it later. Personas
don't have to file findings; they're an in-arc convenience for
"this is friction I want to remember without losing my place."
The evaluator's pass will discover findings of its own based on
the artifacts the personas produce, and is the authoritative
source for what counts as a finding for the overall run.

Findings live at `findings/<persona>/<NN>-<slug>.md` in the
shared worktree. The format is short and free-form — enough for
the evaluator (or a fix-pass agent) to know what happened and
where to look:

```markdown
# <Short title>

**Persona:** Curator | Modeler | Analyst
**Phase:** <what the persona was trying to do>

## What happened

<What was being attempted; what was expected; what actually
occurred. Include exact commands, error messages, file paths,
RIDs as available.>

## Reproduction

<Steps a future reader could use to re-create the situation.>

## Notes

<Anything else relevant — workarounds tried, related code, hunches
about scope. Keep brief.>
```

Severity and component classifications are *not* set by the
persona — those are evaluation judgments, made later by the
evaluator (with full context across personas) or the fix-pass.

## 5. Bootstrap (Phase 0)

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
- **Bootstrap failure modes are still surfaced.** Phase 0 part E
  (step 9 below) is a fail-fast sanity gate on the catalog the
  bootstrap produced. If `load-cifar10` breaks the catalog or
  produces obviously wrong state, that's a Phase 0 finding before
  any persona starts.

### 5.1 What Phase 0 produces (the persona inputs)

By the time Phase 0 is done, the following is true:

- A single shared git worktree exists at
  `../deriva-ml-model-template-e2e` on branch `e2e-test/<YYYY-MM-DD>`,
  cut from `main` of this repo. All persona work happens here (§3.4).
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
  restarted. The **OAuth flow is completed as step 1 of §5.2** —
  it's the first action Phase 0 performs so the orchestrator fails
  fast if auth can't be established. `claude mcp list` should
  report `dev-localhost: ... - ✓ Connected` after step 1, and the
  `deriva_ml_*` tools should be callable. Without this, Phase 0
  part E (catalog sanity check) and every persona's MCP-tool work
  is blocked.

### 5.2 Phase 0 steps (in order)

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
      Curator / Modeler / Analyst will lean on must match the
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

9. **Phase 0 part E — confirm the catalog is usable.** Quick
   sanity check that the bootstrap produced what was expected:
   the catalog exists at the expected name and id; the canonical
   dataset hierarchy is reachable; `Image_Classification` feature
   values are populated for the labeled partitions; the class
   distribution is approximately uniform across the 10 CIFAR-10
   classes (a severely skewed distribution — e.g., bird+ship
   dominating — indicates the loader has regressed).

   This is a fail-fast gate on bootstrap, not the parity check the
   evaluator will eventually run on the personas' work. If
   something at this step looks badly wrong (no datasets, all-null
   features, severe class skew), file a Phase 0 finding against
   `load-cifar10` and stop — running personas against a broken
   catalog learns nothing.

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

13. **Launch the Curator** in the shared e2e worktree with their
    persona prompt. (Modeler and Analyst launch later, sequentially,
    in the *same* worktree — there are no per-persona worktrees in
    this revision of the document; see §3.4.)

### 5.3 What's *not* Phase 0

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

## 6. Wrap-up (mechanical)

When the three personas have finished, the scenario is over. The
**evaluator** (a separate agent, driven by its own rubric document)
runs next: it reads the e2e branch, the catalog state, the
`tacit-knowledge.md` chain, the `findings/` directory, and any
reports the personas produced, and writes its own evaluation under
`docs/reports/<YYYY-MM-DD>-evaluation.md`. This scenario document
ends at "three personas done"; the evaluation is its own pass.

The mechanical wrap-up of the worktree happens after the
evaluator's pass, not before it — the evaluator needs the artifacts
in place to do its work:

1. **Cherry-pick template fixes** from the shared
   `e2e-test/<YYYY-MM-DD>` branch back to `main` of the model
   template. `[E2E-DROP]`-tagged commits (the catalog-id and RID
   repointing commits) stay on the branch; only genuine improvements
   to the template get picked back.
2. **Push or archive the e2e branch** for the historical record.
3. **Worktree teardown** with explicit user confirmation: `git
   worktree remove ../deriva-ml-model-template-e2e`, then
   `git branch -D e2e-test/<YYYY-MM-DD>`.
4. **Catalog disposition** with explicit user confirmation: delete
   or preserve.

---

## 7. Things this document deliberately does not cover

- **Evaluation criteria and findings classification.** Those belong
  to the evaluator's rubric (`evaluator.md`), not the scenario.
- **What a "good" run looks like.** Same — that's the evaluator's
  call. The scenario produces artifacts; the evaluator judges them.
- **Inline bug-fixing.** Findings flow to a fix-pass after the
  evaluator's report lands.
- **Multi-host scenarios, performance benchmarks, schema migration
  exercises.** Out of scope; tracked elsewhere.
- **Concurrent persona execution.** Sequential only. Concurrent
  execution is a future variant once the sequential baseline
  reveals the cross-persona friction patterns.
- **Personas other than the three named.** Platform integrator,
  reviewer, ops, etc. — each is worth a run, but not this run.

---

## Quick reference

| Question | Answer |
|---|---|
| Where does this scenario live? | `docs/test-plans/2026-05-20-e2e-multipersona.md` |
| Where does the evaluator's rubric live? | `docs/test-plans/evaluator.md` (separate document) |
| Where do persona findings go? | `findings/<persona>/<NN>-<slug>.md` in the shared e2e worktree |
| Where does the persona-to-persona handoff happen? | `tacit-knowledge.md` (project root, in the shared e2e worktree) |
| Who creates the catalog? | Phase 0 bootstrap (§5), via `load-cifar10` — *before* any persona runs |
| What's the catalog name? | `e2e-test-<YYYYMMDD>` (chosen at run start) |
| Mode flag? | Interactive (checkpoint per persona) or Autonomous; chosen at start |
| Branch / worktree? | Single shared branch `e2e-test/<YYYY-MM-DD>` cut from `main`, checked out at `../deriva-ml-model-template-e2e`. All three personas run sequentially in this one worktree (see §3.4). |
| Who writes the evaluation? | The evaluator agent, after the personas finish. Output: `docs/reports/<YYYY-MM-DD>-evaluation.md`. |
| Who fixes bugs surfaced? | A fix-pass agent, after the evaluator's report. Personas never fix mid-arc. |
