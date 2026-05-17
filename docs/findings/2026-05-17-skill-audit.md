# Skill audit — 2026-05-17

A comprehensive review of the **`deriva-ml-skills`** plugin (with
**`deriva-skills`** as its assumed companion), evaluated from three
perspectives:

- **The LLM** — would the right skill fire at the right time? Would
  the agent follow it correctly?
- **The ML developer** — do these skills carry me through realistic
  model-training workflows end-to-end?
- **The data manager** — do they cover catalog stewardship —
  vocabularies, schema evolution, lineage, ACL, handoff?

Method: three parallel persona audits dispatched as Explore agents,
each walking concrete workflows derived from the e2e test plan
(Phase 1–11) plus the personas the user named. Findings cross-
checked against the source code (skill file contents, MCP tool
signatures) to filter agent hallucinations.

## Snapshot

- **Skill count:** `deriva-ml-skills` 27 on `main`, +2 in open PRs
  (#16 `setup-ml-catalog`, #17 `using-deriva-mcp`); `deriva-skills`
  12. Total **39 → 41** after open PRs merge.
- **Overall assessment:** **Strong** on the forward ML path and on
  generic catalog operations. **Weak** on catalog stewardship work
  (schema evolution, vocabulary deprecation, ACL review, catalog
  health audit, project handoff).
- **Plugin balance:** appropriate. Generic catalog operations correctly
  route to `/deriva:*`; DerivaML-domain operations correctly route to
  `/deriva-ml:*`. No skill mis-routes between the plugins.

## Findings that survived cross-validation

Each finding below was raised by at least one persona audit and
verified by source inspection. Lower-confidence agent claims (notably
"no skill owns multirun design", which is wrong — `write-hydra-config`
and `configure-experiment` both cover it) were dropped.

### HIGH-IMPACT GAPS (affect multiple workflows; no skill owns the operation)

#### S1. Feature population from CSV
*Affects:* ML-developer W4 (add feature mid-project), DM1 (quarterly
data drop). Prior audit's #5.a from 2026-05-16 — still unresolved.

The `create-feature` skill teaches the *forward path* (define vocab →
create feature → add values via an Execution's Python API), but does
not document the common one-time pattern: read a CSV of ground-truth
labels, parse, and call `deriva_ml_add_feature_values` either in a
batch script or via the MCP tool directly. Real data managers and ML
developers hit this constantly — receiving a CSV from a domain expert
and loading it into a feature is bread-and-butter work.

**Where it lives now:** nowhere. `create-feature/SKILL.md` lines
76-93 cover Python API usage inside a running execution. There's no
"load values from a file" recipe.

**Recommended fix:** Add a Phase 3.1 subsection to `create-feature`:
"Bulk-populate feature values from a file." Worked example using
pandas to read CSV, validate rows against the feature's value-column
schema, and call `deriva_ml_add_feature_values` in batches. Note
provenance trade-off: a CSV-driven ingest should still happen inside
a tracked Execution (with a workflow describing the CSV source), not
as a raw MCP call — otherwise the resulting feature values lose their
provenance chain.

Effort: 1–2 hours of authoring, similar shape to the salvage skill.

#### S2. Schema evolution / migration workflow
*Affects:* DM2 (schema refactor), W5 cross-reference, ML-developer
schema-change downstream. Prior audit's WF-3 from 2026-05-16 — still
the worst-covered workflow.

Naming conventions are exhaustively documented in
`deriva-skills/skills/entity-naming/references/naming-conventions.md`
(lines 111-128 cover "what breaks" on rename), but **no skill walks
the migration execution end-to-end:** drop old FKs → add new columns
→ migrate row data → re-create FKs → validate that downstream
datasets and features still type-check.

**Why this is a real gap:** schema evolution is unavoidable in
long-lived catalogs. A data manager faced with "split Image into
Subject and Study" today has guidance on what to call the new tables
and what naming patterns to avoid, but no playbook for the operation
itself.

**Recommended fix:** New skill `evolve-schema` (deriva-skills, tier-1
— migration is a generic catalog operation, not an ML one). Checklist
form rather than narrative: (1) plan the cutover state, (2) write a
migration script (using `catalog-operations-workflow` as the template
for code provenance), (3) name the post-migration validation queries
(find rows with NULL on the new FK, find old-FK references that
should have been migrated), (4) feature-impact analysis (which
features reference columns that moved or were renamed?), (5)
rollback strategy.

Effort: 2–3 hours of authoring. Significant value across both
plugins because schema evolution touches everything.

#### S3. Catalog health audit
*Affects:* DM4 (quality audit), plus indirect cross-cutting impact.
Implied by every data manager workflow.

Fragments exist — `/deriva:generate-descriptions` enforces non-empty
descriptions, `/deriva:troubleshoot-deriva-errors` covers
permission-denied flows — but no unified "run this audit, get a
report on what's wrong with my catalog" exists. The kinds of issues
data managers want to surface: tables without descriptions, columns
without descriptions, vocabularies with terms missing descriptions,
datasets with no `Dataset_Type` tag, features whose value columns
disagree with declared types, dangling-FK association rows, orphaned
hatrac assets.

**Recommended fix:** Combination of (a) a runnable audit script in
`deriva-skills/scripts/` (or as a `deriva_audit_catalog` MCP tool if
upstream is willing) and (b) a new skill
`/deriva:audit-catalog-health` documenting what each check catches
and how to interpret findings. The script does the queries; the
skill names the checklist and explains what each result means.

Effort: 3–4 hours. Script is the harder part; skill body is
straightforward once the queries exist.

#### S4. Cross-execution lineage walking (asset → execution → dataset → commit → features)
*Affects:* ML-developer W5 (model comparison synthesis), DM6
(handoff audit).

`deriva_ml_get_lineage` is documented in `compare-model-runs` and
`troubleshoot-execution` as a tool that "returns a tree of producing
executions back to the root." But no worked example shows the full
chain: from a prediction-asset RID, extract the producing execution,
extract the consumed dataset RID + version, extract the workflow's
git commit hash, and (the still-missing piece) extract the feature
lineage — which features were attached to the consumed dataset's
members at the time of the execution.

The lineage tool exists; the **teaching example** doesn't. ML
developers picking winners across runs frequently need this exact
trace, and currently have to stitch it together from the lineage
tool's output without a guide.

**Recommended fix:** Add a "Full lineage example" worked block to
`troubleshoot-execution/SKILL.md` (the natural home — that skill
already covers the lineage tool in its "Trace an artifact's
provenance" section). Show: pick an asset RID, run `get_lineage`,
parse the tree shape, walk to producing execution, extract
`source_code_commit`, extract dataset version pin, render as a
table.

Effort: ~30 min once we agree the example. Builds on existing skill
content.

### MEDIUM-IMPACT GAPS (specific workflows blocked)

#### S5. Vocabulary term deprecation
*Affects:* DM3 (vocabulary curation).

`manage-vocabulary` documents adding terms and synonyms but is silent
on deprecation — how to mark a term as deprecated, how to find
feature values still referencing the deprecated term, how to
coordinate the migration to a replacement term.

**Recommended fix:** Extend `manage-vocabulary` with a "Deprecation
pattern" section: when to add a `Deprecated` column on the
vocabulary, the query for "find feature values still using
deprecated terms", and the migration recipe (point new feature
values at the modern term, leave historical values in place with the
deprecated FK target preserved for provenance).

Effort: ~1 hour.

#### S6. ACL / access-control review
*Affects:* DM5. Currently entirely out of scope.

The data manager workflow for "audit who has write access to what,
at what granularity" has no covering skill. The MCP tool surface
exposes ERMrest's ACL APIs; the skill set doesn't teach the audit
pattern.

**Recommended fix:** New tier-1 skill in `deriva-skills`,
`access-control-audit`, with the query patterns for listing
catalog-level / schema-level / table-level ACLs, identifying
divergences between policy and actual ACLs, and finding orphan rows
that violate scope.

Effort: ~2 hours; depends on the user actually needing this
workflow. Lower priority than S1–S4 unless governance is a current
pain point.

#### S7. Dataset versioning when feature values are added to existing members
*Affects:* DM1 (quarterly data drop), W4 (mid-project feature add).

`dataset-lifecycle` documents version flips for `add_dataset_members`
and `delete_dataset_members` but is silent on whether
**adding/updating feature values on existing dataset members** should
also flip the dataset to a dev label. A dataset whose member rows
gained new feature values is materially different from before; a
consumer reproducing an earlier experiment needs to know.

**Recommended fix:** Add a paragraph to `dataset-lifecycle` Phase 4
"Mutations land on dev" naming this case explicitly and stating the
team's policy (probably: yes, the dataset should flip to dev because
its observable content changed even though its member RID set
didn't).

Effort: ~20 min.

#### S8. Project handoff / onboarding (process, not technical)
*Affects:* DM6, W6. Cross-cutting.

The technical onboarding (clone, install, auth, run an existing
experiment) is reasonably covered by `setup-derivaml-project`,
`validate-project-setup`, and `setup-notebook-environment`. What's
missing is the **process documentation**: a project-specific
convention template (where local naming/vocab/versioning policies
live), a handoff checklist (what departing data managers should
ensure before leaving), and a runnable health-check (whatever S3
becomes, applied as a one-command verification).

**Recommended fix:** A new tier-1 skill `catalog-handoff` in
`deriva-skills` that wraps the S3 health-check script plus a
project-convention documentation template (markdown skeleton the
project fills in with their own choices). Could also extend
`getting-started` with a "if you inherited this catalog from
someone" path.

Effort: ~1 hour if S3 lands first; standalone it's ~2 hours.

### LOW-IMPACT FINDINGS (small fixes, no workflow blocking)

These came from the LLM-perspective audit. Each is line-cite-able;
none mislead the agent in serious ways but each marginally improves
clarity.

| ID | Skill | Issue | Fix |
|---|---|---|---|
| L1 | `configure-experiment/SKILL.md:54` | "Setup Steps" subsection trails off into the multirun section without bridging. | Add a one-line transition. |
| L2 | `execution-lifecycle/SKILL.md:14` | `--allow-dirty` use case underspecified. | Add: "for rapid local iteration only, never for production runs that get cited." |
| L3 | `troubleshoot-execution/SKILL.md:19-60` | The five problem sections lack a top-level symptom-to-section map. | Add a table at the top. |
| L4 | `create-feature/SKILL.md:80` | "Create vocabulary + terms" step doesn't show the `add_term(...)` call signature inline. | Add a worked example block. |

## What the audit confirmed is NOT a gap

Things that came up but, on verification, are adequately covered:

- **Multirun config design.** The ML-developer agent claimed no skill
  owns this; in fact `configure-experiment/SKILL.md:59` has a
  Multiruns section and `write-hydra-config/SKILL.md:75-80` covers
  the syntax and CLI invocation. **Adequately covered.**
- **Failed execution salvage.** Covered by PR #5 (`troubleshoot-execution`
  salvage workflow, with the four-branch decision tree). Verified
  end-to-end coverage of W3. **Done.**
- **Setting up a fresh ML catalog from scratch or from clone.** Covered
  by PR #16 (`setup-ml-catalog` with two branches). Verified W1 step
  1 and DM1 step 1. **Done.**
- **MCP cold-start orientation.** Covered by PR #17 (`using-deriva-mcp`
  skill teaching the agent to read upstream prompts/resources before
  first MCP call). Closes finding B1 from May 2026. **Done after #17
  merges.**
- **Resource-vs-tool routing.** Already in `deriva-ml-context`'s
  "Read-side questions" section plus reinforcement in individual
  skills from PR #18. **Done after #18 merges.**

## Cross-references and triggering

- **Cross-reference integrity:** spot-checked all `/deriva-ml:*` and
  `/deriva:*` references in the LLM audit — all targets exist and
  cover what's claimed. **Healthy.**
- **Description triggering:** no MAJOR collisions. The
  `dataset-lifecycle` vs `experiment-lifecycle` overlap on "training
  data" is actually correct — they cover complementary
  data-centric vs hypothesis-centric framings, and a user asking
  about both should reasonably get both fired. **Healthy.**
- **Always-on skills:** `deriva-ml-context`, `generate-descriptions`,
  `maintain-experiment-notes`, `troubleshoot-execution` (auto-fire),
  `using-deriva-mcp` (PR #17 pending). All scoped appropriately;
  descriptions specify firing conditions precisely. **Healthy.**

## Prior-audit follow-up

| Prior finding | Status as of 2026-05-17 |
|---|---|
| **#5.a** Feature population from CSV | Still unresolved → see **S1** above |
| **#5.b** Feature lineage after schema change | Still unresolved → folded into **S2** (schema migration) and **S4** (lineage walking) |
| **WF-3** Schema evolution | Still the worst-covered workflow → see **S2** above |
| **B1** `using-deriva-mcp` skill | **DONE** in PR #17 |
| **B2** Resource cross-references | **DONE** in PR #18 |
| **B3** `maintain-experiment-notes` triggers on entry-point choices | **DONE** in PR #18 |
| **B4** Clone catalog slice | **DONE** in PR #16 (`setup-ml-catalog`) |

## Most-served vs least-served personas

- **Best-served: ML developer.** 5 of 6 workflows covered FULL (W1
  train from scratch, W2 multirun comparison, W3 failed-run recovery,
  W5 lineage with partial gaps, W6 onboarding). The skill set was
  clearly designed around this persona.
- **Moderately-served: LLM.** No major triggering or correctness
  issues; small clarity fixes only. The plugin reads cleanly to the
  agent.
- **Worst-served: Data manager.** 3 of 6 workflows partially or
  wholly uncovered (DM2 schema evolution, DM4 health audit, DM5 ACL
  review). The plugin wasn't designed for this persona and it shows.

## Recommendations, ordered

If you address findings in order of (impact × ease), the priority
cut is:

1. **S1** — CSV feature population (~1–2 hr). Highest impact across
   workflows; small, surgical edit to `create-feature`.
2. **S4** — Lineage walking worked example (~30 min). Builds on
   existing skill content; high LLM-mechanical-correctness value.
3. **S7** — Dataset versioning when features are mutated (~20 min).
   Closes a real semantic ambiguity in `dataset-lifecycle`.
4. **L1–L4** — bundle the four LLM-perspective minor fixes into one
   PR (~30 min total).
5. **S2** — Schema evolution skill (~2–3 hr). Biggest persona-
   coverage win; the worst-covered workflow gets addressed.
6. **S3** — Catalog health audit (~3–4 hr). Pair with S8 onboarding
   so the script is reusable.
7. **S5** — Vocabulary deprecation (~1 hr). Small, isolated.
8. **S8** — Project handoff (~1–2 hr). Best done after S3.
9. **S6** — ACL audit (~2 hr). Defer unless ACL governance becomes
   an active pain point.

Items 1–4 are ~3 hours of focused work and address the worst of the
findings. 5–6 are real new skills (~5–7 hours combined) and improve
the data-manager persona coverage substantially. 7–9 are
incremental.

## Out of scope / explicitly not gaps

- **Data privacy / HIPAA compliance.** Mechanisms exist (ACLs,
  snapshots, audit logs); compliance is policy, not skill territory.
- **Cross-catalog federation.** Multiple Deriva instances orchestrated
  outside the skill set.
- **Domain-specific catalog design** (e.g., medical-imaging schema
  conventions). Belongs in domain plugins on top of these.

## Method notes for the next audit

- **Three persona walkthroughs in parallel** worked well — different
  agents independently flagged the same gaps (CSV ingest, schema
  migration, lineage walking), which lifts confidence.
- **One agent (ML developer) had partial hallucinations** —
  claimed a skill was missing that actually exists. The cross-
  validation pass (verifying agent claims against the actual skill
  files) is load-bearing. Always run it.
- **Skill-count discrepancies between agents** were a useful flag —
  one agent counted 27, another 28. The difference was whether they
  picked up open-PR additions. Future audits should explicitly tell
  the agent which state to audit (current main vs. main + open PRs).
- **The e2e plan's Phase 2–11 outlines** were a much richer source
  of realistic workflows than the prior audit's invented scenarios.
  Whenever a real test plan exists, use it.
