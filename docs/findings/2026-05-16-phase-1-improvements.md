# Phase 1 findings — improvement opportunities

**Source:** End-to-end DerivaML platform test session, Phase 1 (catalog bootstrap + direct/indirect verification).
**Date:** 2026-05-16.
**Audit refreshed:** 2026-05-17 against deriva-py `5cd01a25`, deriva-ml `1.36.5`, deriva-skills `1.2.1`, deriva-ml-skills `1.3.5`, deriva-ml-mcp `main`.
**Companion docs:**
- Session journal: `docs/e2e-test-2026-05-13-journal.md`
- Already-filed bug write-up: `docs/bugs/2026-05-16-bag-loader-stale-path-builder.md`

The items below are grouped by repo so each owner can pick up the
relevant set. Each item lists what was observed, why it matters, and
a concrete (suggested) fix shape.

## Audit summary (2026-05-17)

A re-audit against upstream after the May-16/17 burst of fixes. Status
key:

- ✅ **SHIPPED** — verified in the current pinned version.
- 🟡 **PARTIAL** — addressed but not fully complete or different shape than proposed.
- ⛔ **NOT SHIPPED** — no upstream change observed.
- 📦 **N/A** — owned by an external repo (uv, Claude Code) where filing happens separately.

| # | Repo | Item | Status |
|---|------|------|--------|
| B1 | deriva-ml-skills | New skill `using-deriva-mcp` | ✅ shipped in v1.3.5 |
| B2 | deriva-ml-skills | Cross-reference resource templates from existing skills | ✅ shipped in v1.3.5 (dataset-lifecycle, compare-model-runs, create-feature, work-with-assets all reference `ml_datasets` etc.) |
| B3 | deriva-ml-skills | `maintain-experiment-notes` should capture entry-point choices | ✅ shipped in v1.3.5 (description now explicitly triggers on "MCP entry-point choice") |
| B4 | deriva-ml-skills | New skill `clone-catalog-slice` (deferred) | ✅ shipped as `setup-ml-catalog` in v1.3.5 (broader scope: covers both from-scratch and clone-slice flows) |
| C1 | deriva-mcp-core | Surface guide prompts aggressively | 🟡 C1c done (prompts also as resources at `deriva://deriva-ml/concepts`, `.../getting-started`); C1a/b/d not planned per inline note added by user 2026-05-17 |
| C2 | deriva-mcp-core | Document resource-vs-tool decision | 🟡 Implicitly addressed by the resource-form prompts (C1c) carrying the rule; not a separate section yet |
| C5 | deriva-mcp-core | Per-server attribution in `<system-reminder>` | ⛔ Not shipped (and partly Claude Code side, not just server-side) |
| D1 | deriva-ml-mcp | Cross-link with model-template README | ✅ Shipped. `deriva-ml-mcp/README.md` §"Connecting Claude Code" now links to the model template's §4 (which mirrors the recipe and adds the NODE_EXTRA_CA_CERTS step). Two-way: model template links back to deriva-ml-mcp's §"Connecting Claude Code" as the canonical upstream definition |
| D2 | deriva-ml-mcp | Drop "pre-release" caveat | ✅ Shipped (caveat reworded to "This is the current canonical recipe... it is not a placeholder.") |
| D3 | deriva-ml-mcp | Expand `deriva_ml_getting_started` content | 🟡 Resource form shipped (C1c) — the body now lives where any resource-walking client finds it; "resources vs tools" content needs spot-check inside the prompt body itself |
| E1 | deriva-ml | `ml.refresh_schema()` doesn't suppress stale-cache warning | ⛔ Not shipped (warning still fires; needs deeper look at the warning emission path) |
| E2 | deriva-ml + deriva-ml-mcp | `find_*` vs `list_*` — actually a convention worth documenting | ✅ Fully shipped. MCP wire side: `deriva-ml-mcp/src/deriva_ml_mcp/prompts.py:508-544` "VERB CONVENTIONS ON THE WIRE" section. Python library side: new "Verb naming" subsection in `deriva-ml/CLAUDE.md` §"Key Patterns" + a new "API verb conventions" section in `deriva-ml/README.md`. All three documentation sites cross-reference each other. Live MCP server picks up the source-side prompt change on next container rebuild |
| E3 | deriva-ml | `feature_values()` returns a generator, not list | ⛔ Not shipped |
| E4 | deriva-ml | Schema-mutating methods don't invalidate path-builder cache | ✅ Shipped (deriva-py side — see F1; deriva-ml mutations now route through a binding that auto-invalidates) |
| E5 | deriva-ml | Bootstrapping ships opinionated vocab term set | ✅ Shipped 2026-05-17 in `deriva-ml/main` commit `e52ddc9c` (PR #164). Removed four domain-specific terms from default `Workflow_Type`: `VGG19`, `RETFound`, `Multimodal`, `Embedding`. Remaining 9 seeded terms are all platform-level workflow shapes (Training, Testing, Prediction, Feature_Creation, Visualization, Analysis, Ingest, Data_Cleaning, Dataset_Management). New regression test `tests/schema/test_seeded_vocab_terms.py` pins both the load-bearing platform terms and the absence of the removed domain-specific ones. Docstring now carries the platform-only term-selection principle |
| F1 | deriva-py | `getPathBuilder()` cache is invalidation-free | ✅ Shipped (same-instance auto-invalidation on `/schema` POST/PUT/DELETE) |
| F2 | deriva-py | Cheap "is the cache stale?" check | ✅ Shipped (`if_stale=True` parameter probes snaptime cheaply) |
| F4 | deriva-py | `InsecureRequestWarning` spam against localhost | ⛔ Not shipped |
| F5 | deriva-py | FK-cycle warning log spam | ✅ Shipped (targeted suppression via `_intentional_cycles` + `_reported_cycles` dedup; SAWarning wrapped in `catch_warnings`) |
| G1 | test plan | Resources as first-class in dual-channel verification | ⛔ Not shipped (the spec/plan haven't been updated — this is a model-template doc change waiting on session resumption) |
| G3 | agent process | Read MCP server's guide prompts first | ✅ Addressed via B1 (the `using-deriva-mcp` skill enforces this at the skill layer) |
| G4 | uv upstream | `uv lock --upgrade-package` silently no-ops on package-name mismatch | 📦 Upstream uv issue; not yet filed |
| G5 | Claude Code upstream | Mid-session `.mcp.json` edits don't propagate | 📦 Claude Code feature ask; not yet filed |
| H2 | deriva-ml-model-template | MCP setup section points at wrong recipe | ✅ Shipped. README §4 rewritten: split into two recipes (remote production server via stdio, dev-localhost via HTTP+OAuth using `claude mcp add -t http dev-localhost ...`). Uses the new meta-marketplace (`informatics-isi-edu/deriva-plugins`) and installs both `deriva@deriva-plugins` and `deriva-ml@deriva-plugins`. Stale `check-versions` reference removed |
| H3 | deriva-ml-model-template | `NODE_EXTRA_CA_CERTS` requirement isn't documented | ✅ Shipped. Bundled into the README §4 rewrite (H2): a "TLS trust for dev-localhost" subsection with the exact `docker cp` extraction command and the `.claude/settings.local.json` env block. Cross-referenced from deriva-ml-mcp's README too |

**Note (2026-05-17):** Auth-cluster findings (C3, C4, F3, F6, G2, H1)
were **removed from this doc** because they are all symptoms of an
underlying auth-stack bug currently being fixed in a separate
session. Once that bug fix lands, the user-facing auth surface
should reduce to one command (log in / log out) and the rest of the
opacity issues resolve as a consequence. The previous text covering
those items is gone from the audit table and the body sections
below.

**Headline (after the 2026-05-17 burst): 18 of 25 remaining
items shipped or partially addressed (~72%).** The 2026-05-16
burst closed B1/B2/B3/B4/Appendix 3/D2/F1/F2/F5/E4/G3 plus C1c
(partial). The 2026-05-17 work closed **E2** (all three doc
sites), **D1** (two-way cross-link), **H2** (README §4
rewritten), **H3** (TLS trust docs bundled into H2), and **E5**
(domain-specific terms removed from default Workflow_Type seeds
with a new regression test).

**Still open:**
- **deriva-ml E1/E3** — `refresh_schema()` warning still fires; `feature_values()` is a generator.
- **deriva-py F4** — `InsecureRequestWarning` against localhost.
- **deriva-mcp-core C5** — per-server `<system-reminder>` attribution.
- **Test plan G1** — resources as first-class in dual-channel verification.
- **External:** G4 (uv), G5 (Claude Code).

Per-item rationale is in each section below; the table is the summary view.

---

## A. Root cause of the original question: why didn't the MCP **resources** get used?

This was the user's flagged finding and the spine of the rest of this
document. Recording the diagnosis up front so the rest of the items
can be read as consequences.

The deriva MCP server's tool catalog surfaced with explicit
server-level instructions (delivered as a `<system-reminder>`):

> "DERIVA MCP server for catalog introspection, querying, entity
> CRUD, annotation management, and file operations. Call the
> relevant guide prompt (query_guide, entity_guide, annotation_guide,
> catalog_guide) before your first use of each tool group in a
> conversation."

The server exposes **six guide prompts** (`query_guide`, `entity_guide`,
`annotation_guide`, `catalog_guide`, `deriva_ml_concepts`,
`deriva_ml_getting_started`) and **10+ ML resource templates**
(`ml_datasets`, `ml_dataset_detail`, `ml_dataset_spec`,
`ml_dataset_members`, `ml_workflows`, `ml_workflow_detail`, ...).
The agent skipped both — read no prompts, used no resources, went
straight to tools.

The mechanical reasons:

1. **The instruction was buried.** The `<system-reminder>` arrived
   alongside a list of 122 tool names. The tool list dominates
   attention.

2. **Claude Code's tool-discovery affordances are tool-shaped, not
   resource-shaped.** `ToolSearch` and the deferred-tool catalog
   are explicitly for tools. There's no equivalent "resource
   search" or "prompt search." `ListMcpResourcesTool` exists but
   has to be remembered, not auto-surfaced.

3. **The deferred-tool name list looked complete.** With
   `deriva_ml_list_datasets`, `deriva_ml_get_dataset`,
   `deriva_ml_list_features`, etc. all visible, the resource
   templates feel redundant. They're not — resources are snapshot-
   shaped (single read returns up to 1000 rows + truncation hint),
   tools are pagination-shaped (preflight → page → advance).
   "Verify the state of a freshly-created catalog" wants snapshots.

4. **Stale habits.** The agent had spent the whole session using
   `deriva-ml` Python API (`ml.find_datasets()`, etc.). When
   switching to the indirect channel, it looked for tool-shaped
   analogs of the methods it already knew. Resources never
   entered the search space.

5. **No skill or prompt enforced the read-the-guide step.** No
   deriva-ml skill said "read `deriva_ml_getting_started` before
   you start." The MCP server said so; the server's instruction
   was easy to skip.

Everything below traces back to these mechanics. Each fix targets
a specific link in the chain.

---

## B. `deriva-skills` and `deriva-ml-skills`

**Repo:** `informatics-isi-edu/deriva-skills`,
`informatics-isi-edu/deriva-ml-skills`.

### B1. New skill: read-mcp-guide-first

> **Status: ✅ SHIPPED in deriva-ml-skills v1.3.5** as the
> `using-deriva-mcp` skill. Description matches the proposed
> trigger set; body encodes the cold-start discipline (read
> `deriva://deriva-ml/concepts` then `deriva://deriva-ml/getting-started`
> via `ReadMcpResourceTool` before the first MCP tool call,
> plus the per-tool-group guide prompts from deriva-mcp-core).
> Tier-2 in deriva-ml-skills as proposed.

Add a skill in `deriva-ml-skills` (or possibly tier-1 in
`deriva-skills` since the four core guides — `query_guide` etc. —
are tier-1) whose only job is:

> Before your first deriva-mcp call in a conversation, read the
> `deriva_ml_concepts` and `deriva_ml_getting_started` MCP prompts.
> When reaching for a `deriva_ml_*` tool, also read the relevant
> per-group guide prompt.

Trigger phrases should be high-confidence for any context that
involves a Deriva catalog operation — "list datasets", "check
catalog state", "look at this catalog", "verify schema", etc. The
skill body is short: it tells the agent to fetch the guide prompts
via the MCP protocol's `prompts/get` method (or whatever Claude
Code wrapper surfaces it), or to invoke `ListMcpResourcesTool` /
read each prompt by URI.

This mirrors the server's instruction at the layer where Claude
Code's auto-discovery affordances actually fire. The MCP server's
instruction is correct but invisible at use-time.

### B2. Cross-reference resource templates from existing skills

> **Status: ✅ SHIPPED in deriva-ml-skills v1.3.5.** Verified in
> `dataset-lifecycle`, `compare-model-runs`, `create-feature`,
> and `work-with-assets` — all reference `ml_datasets` and other
> `ml/...` resource templates as the preferred snapshot-read path.

Every existing skill that reaches for `deriva_ml_*` list-style tools
(`dataset-lifecycle`, `compare-model-runs`, `create-feature`,
`work-with-assets`, etc.) should mention the corresponding **resource
template** as the preferred path for snapshot reads, and the tool
as the path for paginated drilling.

Concrete: in `dataset-lifecycle/SKILL.md`, when the agent is
verifying or browsing existing datasets, prefer
`ReadMcpResourceTool(server="...", uri="deriva://catalog/{h}/{c}/ml/datasets")`
over `deriva_ml_list_datasets`. Same shape decisions for
workflows, executions, features.

### B3. `maintain-experiment-notes` should capture entry-point choices

> **Status: ✅ SHIPPED in deriva-ml-skills v1.3.5.** Skill
> description explicitly triggers on "making an MCP entry-point
> choice where more than one valid path existed (e.g., 'I used
> resource X rather than tool Y because the catalog was fresh
> and a snapshot was sufficient')." Plumbing decisions are now
> first-class triggers alongside content decisions.

Today this skill triggers on dataset / feature / workflow
*content* decisions. It should also trigger on **plumbing decisions
the future maintainer would need to understand** — "I read this via
resource X rather than tool Y because the catalog was fresh and a
snapshot was sufficient." These aren't experiment-design decisions
in the narrow sense, but they're tacit-knowledge decisions of the
same character.

Add a section to the skill's description / examples covering this
shape of decision: "MCP entry-point choices when there's more than
one valid path."

---

## C. `deriva-mcp-core` (the MCP server framework)

**Repo:** `informatics-isi-edu/deriva-mcp-core`.

### C1. Surface the guide prompts more aggressively

> **Status: PARTIALLY DONE (C1c) + remainder NOT PLANNED — closed 2026-05-17.**
> **C1c is shipped** — `deriva_ml_getting_started` and `deriva_ml_concepts`
> are exposed as both prompts AND resources at
> `deriva://deriva-ml/getting-started` and `deriva://deriva-ml/concepts`
> (resource handlers in `deriva-ml-mcp/src/deriva_ml_mcp/resources/ml.py`).
> Resource-walking clients now find them.
> **C1a, C1b, C1d are NOT PLANNED** — no current resourcing for further
> `deriva-mcp-core` changes. The user-facing impact of C1 (agents skipping
> the orientation) is mitigated in `deriva-ml-skills` by the
> `using-deriva-mcp` skill ([PR #17](https://github.com/informatics-isi-edu/deriva-ml-skills/pull/17))
> which encodes the cold-start discipline at the skill layer using the
> resource form C1c provides. Re-open C1a/b/d if non-Claude-Code clients
> need server-side enforcement.

Server-level instructions ("Call the relevant guide prompt...") are
correct but easily missed. Options, in increasing order of
intrusiveness:

- (C1a) Repeat the "read the guide prompt" reminder in each tool's
  `description` field. Per-tool repetition is ugly but actually-read.
- (C1b) Emit a structured `prompts` payload alongside the tool
  catalog on `initialize`. Today prompts require a separate
  `prompts/list` round-trip — almost no client does that
  automatically.
- (C1c) Expose `deriva_ml_getting_started` as both a `prompt` AND a
  `resource` (URI like `deriva://meta/getting-started`, always
  reachable). Resource-walking clients find it.
- (C1d) Make the server's `<system-reminder>` text itself include a
  preamble like "BEFORE USING ANY TOOL: read the
  `deriva_ml_getting_started` prompt." Doesn't help clients that
  collapse all server reminders, but helps the ones that show them
  verbatim.

C1c is the most universal — agents that don't read prompts often
do read resources.

### C2. Document the resource-vs-tool decision

> **Status: NOT PLANNED at server layer — mitigated at skill layer.** Closed
> 2026-05-17. No current resourcing for `deriva-mcp-core` / `deriva-ml-mcp`
> prompt edits. The teaching is now carried in `deriva-ml-skills`:
> `deriva-ml-context`'s "Read-side questions: fetch the resource first"
> section is the canonical statement, reinforced by individual skills
> (`dataset-lifecycle`, `compare-model-runs`, `create-feature`,
> `work-with-assets`) that cross-reference the resource alternative next
> to their `deriva_ml_list_*` examples ([PR #18](https://github.com/informatics-isi-edu/deriva-ml-skills/pull/18)).
> This addresses the user-visible problem (Claude Code agents defaulting
> to tools when a resource would serve) but leaves non-Claude-Code clients
> with no server-side guidance. Re-open if those clients matter.

In `deriva_ml_getting_started` (or a new `entry_points_guide`),
add a "when to use a resource vs a tool" section. The shape:

- **Resource** — single read, snapshot-shaped, suitable for
  verification / introspection / "show me what's there." Capped at
  some bound (1000 rows for `ml_datasets`); truncated reads return
  a hint to switch to the tool.
- **Tool** — paginated, cursor-based, suitable for full enumeration
  or interactive drilling. Pay the preflight cost (or skip it for
  fixed-shape queries).

Without this, agents like me default to tools because the deferred-
tool catalog is the affordance Claude Code surfaces.

### C5. Per-server attribution in `<system-reminder>` collapses

> **Status: NOT PLANNED — closed 2026-05-17.** Primarily a Claude Code
> rendering issue (per the finding's own note) rather than something
> `deriva-mcp-core` can fix definitively. No current resourcing for
> server-side mitigation either. If multiple-MCP-server agent confusion
> becomes a recurring debugging cost in real sessions, the server-side
> prefix workaround (`[deriva-mcp-core] ...` in its instruction text)
> is a one-line change worth re-opening for.

The aggregated `<system-reminder>` for "MCP Server Instructions"
concatenates all servers' instructions into one block. With one
server, fine. With multiple, the agent has to figure out which
server an instruction belongs to. Some lightweight per-server
attribution (`## <server-name>` headings) would help.

This is more a Claude-Code-side rendering issue than a server-side
one, but the server could mitigate by prefixing its own instruction
with `[deriva-mcp-core] ...`.

---

## D. `deriva-ml-mcp` (the deriva-ml plugin loaded by the core)

**Repo:** `informatics-isi-edu/deriva-ml-mcp`.

### D1. Cross-link to the model-template README

> **Status: ✅ SHIPPED 2026-05-17.** Two-way cross-link:
> - `deriva-ml-mcp/README.md` §"Connecting Claude Code to the
>   dockerized server" now ends with a paragraph pointing at
>   the model template's §4 (which mirrors the recipe and adds
>   the NODE_EXTRA_CA_CERTS step).
> - `deriva-ml-model-template/README.md` §4 closes with "See the
>   `deriva-ml-mcp` README for the canonical dev-localhost
>   recipe and HTTPS-transport notes." linking back.
> Together H2 is also resolved (the template's recipe is now
> correct).

The deriva-ml-mcp README §"Connecting Claude Code to the dockerized
server" has the correct recipe (`claude mcp add -t http
dev-localhost https://localhost/mcp --client-id deriva-mcp
--callback-port 8080`). The model template's README §"Set Up
Claude Code (Optional)" has a different, incorrect-for-dev recipe
(`docker run -i --rm ... ghcr.io/informatics-isi-edu/deriva-mcp`).

Either:
- Refer the model template README directly at the deriva-ml-mcp
  README for the dev recipe, OR
- Lift the deriva-ml-mcp recipe verbatim into the model template
  with attribution.

Today they diverge silently, and the agent (and any new dev) hits
the wrong recipe.

### D2. The "pre-release" caveat box in the README is load-bearing

> **Status: ✅ SHIPPED.** The caveat was reworded to: "This is
> the current canonical recipe for dev-localhost. ... it is not
> a placeholder." Future-superseded-by-deriva-docker context is
> kept but no longer reads as "skip this section."

Currently:

> "Pre-release status. This workflow exists because deriva-docker
> support for the deriva-ml-mcp plugin is currently pre-release.
> When the final deriva-docker release ships, this section and the
> helper script will be replaced by deriva-docker's canonical
> workflow."

For new users this reads as "skip this section, it'll be replaced."
It's actually the only working dev-localhost recipe. Either tag
it more visibly as "current canonical recipe" or remove the
pre-release language until deriva-docker's canonical workflow is
actually available.

### D3. Guide-prompt content for `deriva_ml_getting_started`

The prompt body (I read the first ~60 lines) covers the (hostname,
catalog_id) rule and the pagination contract. Two additions would
have helped me in Phase 1:

- A "resources vs tools" section (matches C2 above).
- A pointer to the deriva-mcp-core core guides (`query_guide`,
  `entity_guide`, `annotation_guide`, `catalog_guide`) for clients
  that haven't yet read them.

The prompt already says it's layered above core; making the
layering visible (which guides apply at which layer) would help an
agent navigate.

---

## E. `deriva-ml` (the Python library)

**Repo:** `informatics-isi-edu/deriva-ml`.

### E1. `ml.refresh_schema()` doesn't suppress the stale-cache warning

> **Status: ⛔ NOT SHIPPED.** Tested against deriva-ml v1.36.5:
> calling `ml.refresh_schema()` explicitly does not silence the
> "schema cache is at snapshot X; live catalog is at Y" warning
> on subsequent `find_datasets()` / `find_features()` calls.

> **Status: NOT A BUG — the cache round-trip is correct.** Investigated
> 2026-05-17 (deriva-ml PR #162). The 8 unit tests in
> `tests/core/test_refresh_schema_warning.py` pin the
> correctness invariants: `SchemaCache.write` persists the
> snapshot, the on-disk file authoritatively reflects the latest
> write, the warning has exactly one source in `core/base.py`,
> and `refresh_schema` is structurally wired to read the live
> snapshot and write it to the cache in that order. The reported
> behavior is one of three environmental cases (multi-tenant
> catalog advancement; silent `refresh_schema` exception swallowed
> by an outer try; `working_dir` mismatch between sessions). If
> reported again, ask the user which of those three applies
> before opening a bug. **No code change in deriva-ml.**

The warning fires:

```
WARNING:deriva_ml.core.base:schema cache is at snapshot
353-YMFJ-98W2; live catalog is at 353-YMJ5-C4KJ.
Using cached schema. Call ml.refresh_schema() to update.
```

When I explicitly call `ml.refresh_schema()` immediately after
construction, the warning still fires for subsequent operations.
Either the warning checks a different cache than `refresh_schema()`
updates, or `refresh_schema()` isn't actually invalidating the
right layer.

Repro:
```python
ml = DerivaML(hostname="localhost", catalog_id="8", check_auth=True)
ml.refresh_schema()  # explicit refresh
# Now use ml.find_datasets(), ml.find_features(...), etc.
# Warning still fires for these subsequent calls.
```

### E2. Inconsistent verb: `find_features` vs `list_*` — actually a convention worth documenting

> **Status: ✅ FULLY SHIPPED 2026-05-17.** Documented in all
> three planned places:
> - **MCP wire surface:** `deriva-ml-mcp/src/deriva_ml_mcp/prompts.py:508-544`
>   "VERB CONVENTIONS ON THE WIRE" section.
> - **Python library agent guide:** new "Verb naming" subsection
>   in `deriva-ml/CLAUDE.md` §"Key Patterns" with the full
>   convention + a worked example (`ml.list_features`
>   AttributeError → use `ml.find_features(table)`).
> - **Python library public README:** new "API verb conventions"
>   section in `deriva-ml/README.md` with the same rule and a
>   pointer back to CLAUDE.md for details.
> All three docs cross-reference each other so a reader who lands
> on any one finds the others. Live MCP server picks up the
> prompt-side change on the next container rebuild.
>
> Investigation against deriva-ml v1.36.5 found an
> **implicit convention**:
>
> - `find_*` — schema-introspection / discovery that walks the
>   catalog model with filtering logic. Used for things like
>   `find_features`, `find_datasets`, `find_workflows`,
>   `find_executions`, `find_assets`, `find_associations`. These
>   do non-trivial work to identify matching entities (predicates,
>   traversal, association detection).
> - `list_*` — straightforward enumeration of "what's there."
>   Used for `list_assets`, `list_executions`,
>   `list_dataset_members`, `list_dataset_children`,
>   `list_dataset_parents`, `list_vocabulary_terms`. These are
>   thin wrappers over a table read.
>
> So `find_features` is consistent with the convention (features
> are an association pattern that has to be *discovered* — they're
> derived from `find_associations(min_arity=3, max_arity=3,
> pure=False)`, see `deriva-ml/docs/design/deriva-ml-audit-2026-05-phase2-model.md`
> §2.2). Not an inconsistency — but the convention is not stated
> anywhere a new user would find it.
>
> **Revised recommendation:** rather than alias `list_features`
> to `find_features`, **document the convention** in three places:
>
> 1. **`deriva-ml/CLAUDE.md`** (the agent guide for code-side
>    work). One paragraph + the rule.
> 2. **`deriva-ml`'s public README / API reference.** Same.
> 3. **The `deriva_ml_getting_started` MCP prompt** in
>    `deriva-ml-mcp`. Verified against the live server
>    (v1.36.5 / current head, 2026-05-17): this prompt does
>    NOT currently explain the convention. It uses both verbs
>    throughout — `find_workflow_by_url`,
>    `find_workflow_executions` in the workflow/execution
>    domain summaries, plus 12+ `list_*` tools elsewhere — but
>    never says why one verb vs the other. An agent reading the
>    prompt sees both verbs in the tool inventory and has no
>    way to predict which to use. Add the rule to the prompt
>    body somewhere near the "THE FIVE ML DOMAINS" section.
>
> One paragraph each. Rule: "use `find_*` when the method has to
> discover/filter (predicate, traversal, association detection);
> use `list_*` for straight enumeration of what's there." This
> reframes the implementation work as **don't add aliases;
> document the existing rule.**
>
> Edge cases worth noting in the docs:
> - `list_dataset_element_types` does some traversal but is named
>   `list_*` — borderline; pre-existing.
> - `find_assets` is currently just an `is_asset(t)` filter (per
>   the audit doc); arguably should be `list_assets` (which also
>   exists separately, on a different layer — see the
>   audit's §1.4 about the unused `with_metadata` param). The two
>   are a naming-collision smell worth a follow-up cleanup.

> **Status: NOT A BUG — the convention is intentional.** Resolved
> 2026-05-17 by deriva-ml PR #163 (DerivaML class docstring) +
> deriva-ml-skills#15 (deriva-ml-context skill addition). The
> `find_*` vs `list_*` distinction is the project convention:
> `find_*` is a catalog-wide search of entities of a kind
> (optionally filtered); `list_*` is enumeration scoped to a
> specific parent entity passed as the first argument. The
> muscle-memory miss was a discoverability issue, not an
> inconsistency. `ml.list_features()` doesn't exist because
> features aren't scoped to a parent entity the way dataset
> members are scoped to a dataset — the catalog-wide search is
> `ml.find_features()`. **No code change in deriva-ml.**

- `ml.find_features(table)` ✓ exists
- `ml.list_features()` ✗ doesn't exist
- `ml.list_assets("Image")` ✓ exists
- `ml.list_workflows()` ✓ exists
- `ml.list_executions()` ✓ exists

The plan I wrote had `ml.list_features()` from muscle memory. Real
code crashed with `AttributeError: 'DerivaML' object has no
attribute 'list_features'. Did you mean: 'find_features'?`

**Resolution analysis:** This finding's premise (the inconsistency)
was incorrect. The library's full call surface uses the two
prefixes consistently:

| Prefix | Source | Examples |
|---|---|---|
| `find_*` | Live catalog search | `find_features`, `find_datasets`, `find_workflows`, `find_executions`, `find_experiments`, `find_assets`, `find_incomplete_executions` |
| `list_*` | Local workspace or scoped enumeration | `list_assets(table)`, `list_dataset_members(dataset)`, `list_dataset_children(dataset)`, `list_workflow_executions(workflow)`, `list_vocabulary_terms(table)`, `list_executions(...)` (local SQLite registry; distinct from `find_executions` which queries ERMrest) |

The two `*_executions` methods are the load-bearing example
that disambiguates the rule: `find_executions(...)` hits the
live catalog; `list_executions(...)` reads the local registry
in `~/.deriva-ml/.../workspace.db`. The bullets in this
finding's original text said "`ml.list_workflows()` ✓ exists"
and "`ml.list_executions()` ✓ exists" — `list_executions` is
real (it's the registry reader), and **`list_workflows` does
not exist** on `DerivaML` (the catalog-search is `find_workflows`).
So even the original finding's spot-checks were not what they
seemed.

The fix is to document the convention where LLMs and humans
will see it — done in two places (Python class docstring;
always-on skill). Closed as documentation, not as behavior
change.

Two reasonable fixes:
- Add `list_features` as an alias of `find_features`.
- Rename `find_features` → `list_features` (deprecate the old
  name on a release boundary).

### E3. `ml.feature_values(...)` returns a generator, not a list

> **Status: ⛔ NOT SHIPPED.** Still a generator; `list()` wrapper
> needed by callers that want length / slicing.

Every other `list_*` returns a list. `feature_values()` returns a
generator, which silently breaks code that does `len(values)` or
`values[:5]`.

Fix: either standardize to a list (and add an explicit
`iter_feature_values` for streaming), or rename the method to
something that signals lazy iteration (`iter_feature_values`,
`stream_feature_values`).

### E4. Schema-mutating methods don't invalidate the path-builder cache

> **Status: ✅ SHIPPED (root cause fixed in deriva-py, see F1).**
> deriva-py's `ErmrestCatalog` now auto-invalidates
> `_path_builder_cache` on any successful 2xx/3xx `POST` / `PUT`
> / `DELETE` to a `/schema/...` path on the same instance. Since
> deriva-ml's schema-mutating methods (`create_asset`,
> `create_vocabulary`, `add_term`, `create_feature`, etc.) all
> go through that instance, they inherit the fix for free —
> no deriva-ml-side changes needed.

> **Status: NOT NEEDED NOW — closed without code change.** Closed
> 2026-05-17 after auditing every ``getPathBuilder()`` consumer
> in ``deriva-ml/src/``. The load-bearing failure (fresh-catalog
> upload flow) was fixed upstream in deriva-py's
> ``BagCatalogLoader._ensure_path_builder()`` (calls
> ``getPathBuilder(refresh=True)`` on first build). Verified by
> deriva-ml PR #161, pinned by the unit tests in
> ``tests/execution/test_bag_loader_path_builder_refresh.py``.
> Every remaining ``getPathBuilder()`` consumer in deriva-ml
> reads from tables that exist at catalog creation time
> (``deriva-ml`` schema, ``public.ERMrest_RID_Lease``) — none of
> them are vulnerable to in-process schema mutations because the
> tables they touch don't change after ``create_ml_schema``. An
> invalidation hook on ``create_asset`` / ``create_vocabulary`` /
> etc. would be defensive code with no demonstrable consumer.
> **If a future deriva-ml caller is added that mutates the
> catalog schema and then queries the path-builder for the
> newly-added table in the same process, E4 should be revisited
> at that point.**

The bag-loader stale-path-builder bug (filed separately as
`docs/bugs/2026-05-16-bag-loader-stale-path-builder.md`,
fix landed in deriva-py `de19aaf1`) has its **root cause** in
deriva-ml: every method that mutates the catalog schema
(`create_asset`, `create_vocabulary`, `add_term`, `create_feature`,
`add_column`, etc.) leaves a stale `ErmrestCatalog._path_builder_cache`
behind.

The deriva-py fix is defensive (bag-loader passes `refresh=True`
unconditionally). It paid the cost of an extra `/schema` walk per
loader instance. A cleaner deriva-ml-side fix: after every schema
mutation, invalidate the cache:

```python
def create_asset(self, ...):
    # ... do the work ...
    self.catalog._path_builder_cache = None
    # or: self.catalog.getPathBuilder(refresh=True)
```

With that in place, the deriva-py defensive `refresh=True` would
still be correct but redundant.

**Audit conclusion (2026-05-17):** Of the 9
``getPathBuilder()`` call sites in ``deriva_ml/src/``:

- 3 are in ``catalog/clone_via_bag.py`` /
  ``catalog/localize.py`` / ``schema/create_schema.py`` —
  these run on freshly-connected catalogs (no prior schema
  mutations in the same process to be stale relative to).
- 3 are in ``execution/state_machine.py`` — they write to
  the ``Execution`` table (a ``deriva-ml`` schema table
  created at ``create_ml_schema`` time, never mutated after).
- 1 is in ``execution/rid_lease.py`` — reads from
  ``public.ERMrest_RID_Lease`` (system table, present from
  catalog creation).
- 1 is in ``dataset/bag_builder.py`` — reads from
  ``Dataset_Dataset`` (deriva-ml schema, present from
  ``create_ml_schema``).
- 1 is in ``core/mixins/path_builder.py`` — the public
  ``pathBuilder()`` accessor; the user is responsible for
  freshness here, and ``DerivaML.refresh_schema()`` is the
  documented escape hatch.

The bag-loader path was the only load-bearing consumer that
hit the stale-cache scenario, and that's fixed upstream.

### E5. Bootstrapping ships an opinionated vocab term set

> **Status: ✅ SHIPPED 2026-05-17 in `deriva-ml/main` commit
> `e52ddc9c` (PR #164).** Four domain-specific terms removed
> from the default `Workflow_Type` seeds: `VGG19`, `RETFound`,
> `Multimodal`, and `Embedding`. The first three were artifacts
> from an earlier deployment (specific neural-network
> architectures, a specific retinal-imaging foundation model,
> and a research-area category respectively); the fourth
> (`Embedding`) was reclassified during PR review as
> project-orientation rather than a platform-level workflow
> shape — workflows producing vector representations are
> structurally `Feature_Creation` or `Training`.
>
> Remaining 9 seeded terms are all platform-level workflow
> shapes: Training, Testing, Prediction, Feature_Creation,
> Visualization, Analysis, Ingest, Data_Cleaning,
> Dataset_Management.
>
> New regression test `tests/schema/test_seeded_vocab_terms.py`
> (8 tests, no live catalog) pins both:
> - the load-bearing platform terms (so a future cleanup that
>   drops Training/Testing/etc. breaks the test before consumers);
> - the absence of the four removed domain-specific terms (so a
>   future PR re-adding a project-specific name has to argue
>   it's genuinely platform-level).
>
> The `initialize_ml_schema` docstring now carries a
> "Term-selection principle" section: every seeded term must
> describe a platform-level concept, not a domain-specific one.
> Future contributors landing in the source see the rule before
> they propose a new term.
>
> Existing-catalog impact: none. `_ensure_terms` is
> skip-if-exists, so catalogs already initialized (eye-ai,
> model-template runs, etc.) keep whatever terms they have.
> Only new catalogs created after this commit get the cleaner
> default set.

> **Status: FIXED.** Resolved by deriva-ml PR #164 (released in
> v1.36.5, 2026-05-17). Four domain-specific terms removed from
> the ``Workflow_Type`` seeds in ``initialize_ml_schema``:
> ``VGG19``, ``RETFound``, ``Multimodal`` (specific architectures /
> model / research-area), plus ``Embedding`` (its description
> named "foundation models" — also project-orientation).
>
> The remaining 9 ``Workflow_Type`` seeds describe genuine
> cross-domain workflow shapes: ``Training``, ``Testing``,
> ``Prediction``, ``Feature_Creation``, ``Visualization``,
> ``Analysis``, ``Ingest``, ``Data_Cleaning``,
> ``Dataset_Management``.
>
> ``initialize_ml_schema``'s docstring now records a
> "Term-selection principle" that future contributors land on
> before proposing a new term: every seeded term must describe a
> platform-level concept, not a domain-specific one. The new
> regression test ``tests/schema/test_seeded_vocab_terms.py``
> (8 tests) pins both what stays seeded and what must not be.
> A future PR that re-adds ``VGG19``/``RETFound``/``Multimodal``
> fails the regression test and gets forced to either
> demonstrate platform-level relevance or move the term to a
> per-project initializer.
>
> Existing-catalog impact: none. ``_ensure_terms`` is
> skip-if-exists; already-initialized catalogs keep their terms.

A fresh catalog gets:
- 17 `Workflow_Type` terms including `RETFound`, `VGG19`,
  `Embedding`, `Multimodal`
- 8 `Dataset_Type` terms including `Validation`, `File`

Some of these are clearly project-specific. The CIFAR loader
adds three more (`CIFAR_Data_Load`, `Image Classification`,
`ROC Analysis Notebook`). A user adapting the template for their
own data inherits the platform's defaults, which may not match
their domain.

Options:
- Prune the defaults to a true minimum (probably:
  `Workflow_Type` = `Training`, `Testing`, `Prediction`,
  `Analysis`; `Dataset_Type` = `Training`, `Testing`, `Validation`,
  `Complete`, `Labeled`, `Unlabeled`, `Split`).
- Split into a "core" set and an optional "examples" extension that
  ships its own vocab terms when loaded.

The current behavior — load every term ever — is the same shape as
the "ship every config you've ever used" anti-pattern.

---

## F. `deriva-py` (the underlying client library)

**Repo:** `informatics-isi-edu/deriva-py`.

### F1. `ErmrestCatalog.getPathBuilder()` cache is invalidation-free

> **Status: ✅ SHIPPED in deriva-py `5cd01a25`.** The implementation
> now has both **same-instance auto-invalidation** (any successful
> non-error POST/PUT/DELETE to `/schema/...` on this instance
> clears `_path_builder_cache`) AND the explicit cheap-staleness
> check (F2 below). See `deriva/core/ermrest_catalog.py:388` for
> the new `getPathBuilder(refresh=False, if_stale=False)`
> signature and `_invalidate_path_builder_if_schema_mutation` on
> line 466 for the auto-invalidation hook.

This is the substrate beneath E4. Docstring acknowledges the
staleness window only for cross-process schema changes:

> "Schema rows added by another process between calls are not seen
> unless the caller passes refresh=True; this is the same staleness
> window every other catalog-model read has."

But the same-process scenario — schema mutation followed by read —
is much more common in test harnesses, fresh-catalog loaders, and
init scripts. The bag-loader fix (de19aaf1) handles one consumer's
exposure. The underlying library could:

- Auto-invalidate `_path_builder_cache` after any successful
  `POST /schema/...` made through the same `ErmrestCatalog`
  instance. This is the cleanest fix and prevents every future
  consumer from making the same mistake.
- Or: provide a callback registration so schema-mutating layers
  (deriva-ml, custom code) can wire their mutations into the
  invalidation.

### F2. Cheap "is the cache stale?" check

> **Status: ✅ SHIPPED in deriva-py `5cd01a25`** as the `if_stale`
> parameter on `getPathBuilder`. Probes `GET /` for the catalog's
> `snaptime` and rebuilds only when it has advanced past the
> cached value. Cold call always rebuilds (no snaptime recorded
> yet); subsequent calls settle into compare-and-rebuild-on-drift.

`getPathBuilder(refresh=True)` walks the full `/schema` every time
it's called. The bag-loader fix pays that cost once per loader
instance — acceptable. If other layers start sprinkling
`refresh=True` defensively, the cost compounds.

A cheaper check using the catalog's snapshot ETag or version
counter would let consumers conditionally refresh:

```python
def getPathBuilder(self, refresh=False, if_stale=False):
    if if_stale:
        current_snap = self._head_snapshot()  # cheap HEAD or GET /
        if current_snap == self._path_builder_snap:
            return self._path_builder_cache
        refresh = True
    # ... rest as today ...
```

### F4. `InsecureRequestWarning` spam against localhost

> **Status: ⛔ NOT SHIPPED.** Every HTTPS request to localhost
> still emits the warning. Worktree probes during this session
> piped `| grep -v InsecureRequestWarning` to strip them.

Every `requests` call to `https://localhost/...` emits an
`InsecureRequestWarning` from `urllib3`. The deriva-localhost cert
chain is self-signed (DERIVA Dev Local CA, valid through 2035) but
Python's `requests` doesn't see the `NODE_EXTRA_CA_CERTS` Claude
Code uses; it only honors `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE`.

Three reasonable fixes:
- deriva-py honors `NODE_EXTRA_CA_CERTS` (probably wrong scope
  but cheap).
- deriva-py looks for a deriva-installed CA bundle in a known
  location (`~/.deriva/ca-bundle.crt`?) and adds it to the
  `requests` session's trust store.
- deriva-py adds a one-time "to silence this for trusted dev
  hosts, set `REQUESTS_CA_BUNDLE=$HOME/.config/deriva/deriva-dev-ca.crt`"
  hint to its logger setup.

Even (c) — the documentation-only fix — would have saved me
multiple minutes of confusion.

### F5. FK-cycle warning log spam — fix in `deriva-py`

> **Status: ✅ SHIPPED in deriva-py `5cd01a25`.** The targeted
> suppression landed exactly as proposed:
> - `deriva/bag/schema_io.py:422` wraps `metadata.sorted_tables`
>   in `warnings.catch_warnings()` + `filterwarnings("ignore",
>   message="Cannot correctly sort tables")`.
> - `deriva/bag/loader.py:280-289` has an `_intentional_cycles`
>   allowlist (mirrors the proposed `_KNOWN_INTENTIONAL_CYCLES`)
>   and a `_reported_cycles` dedup set. Known cycles log at
>   `DEBUG`, unknown cycles still log at `WARNING` so legitimate
>   accidental cycles in user schemas still surface.

**The cycle is by design.** `Dataset ↔ Dataset_Version` is
intentional in the deriva-ml schema (a Dataset has a
`current_version` FK to Dataset_Version; a Dataset_Version has a
`Dataset` FK back). Each direction is operationally needed. The
schema is not changing.

This finding is **purely about output noise from the consumer
side** (`deriva-py`). The fix:

- `deriva-py`'s bag pipeline knows the `Dataset ↔ Dataset_Version`
  cycle is intentional in the deriva-ml schema and that its own
  manual cycle-handling logic handles it correctly. For **that
  specific known-and-handled cycle**, both the upstream SAWarning
  and the bag-loader's own follow-up log line should be silenced.
- Other cyclic FK warnings — schemas where the consumer doesn't
  *know* the cycle is intentional, or where deriva-py's
  cycle-breaker can't safely resolve them — should still surface
  normally. The cycle warnings exist for a reason; they're a
  legitimate signal that a schema design needs scrutiny. **A
  blanket filter would mask future, genuinely problematic
  cycles** in user schemas (e.g., a domain schema someone designs
  with an accidental cycle).

So the fix is **targeted suppression**, not blanket suppression:

| Where | What |
|---|---|
| `deriva-py/bag/schema_io.py:414` | Check whether the cycle being warned about is the known `(Dataset, Dataset_Version)` pair. If yes, silence the SAWarning at this call site. If no, let it propagate normally. |
| `deriva-py/bag/loader.py:248-250` | Same predicate: if `cycle` is the known `Dataset ↔ Dataset_Version` pair, log at `DEBUG`. Otherwise log at `WARNING` so the user/operator sees it. |

Concrete shape — define a small constant somewhere in the bag
module:

```python
# Cycles deriva-py knows how to handle silently because they're
# part of the deriva-ml core schema by design. Anything not in
# this list still warns normally — cycle warnings are how we
# notice accidental schema bugs.
_KNOWN_INTENTIONAL_CYCLES = frozenset({
    frozenset({"deriva-ml.Dataset", "deriva-ml.Dataset_Version"}),
})
```

Then at the two call sites, check the cycle against this set
before deciding whether to surface the warning. If/when the
deriva-ml schema gains another intentional cycle (or another
deriva-* library does), it's a one-line addition to the set.

**Trace of the noise chain (for the deriva-py maintainer):**

- `deriva-py/bag/schema_io.py:414` accesses `metadata.sorted_tables`.
  SQLAlchemy emits `SAWarning: Cannot correctly sort tables ...
  unresolvable cycles between tables "deriva-ml.Dataset,
  deriva-ml.Dataset_Version"`. The warning text contains the
  table names involved — usable for the cycle-identity check.
- `deriva-py/bag/loader.py:248-250` then emits
  `logger.warning(f"Breaking cycle in FK dependencies: {...}")`
  with the cycle list (e.g., `["deriva-ml.Dataset",
  "deriva-ml.Dataset_Version", "deriva-ml.Dataset"]`). The `cycle`
  variable at that point holds the cycle directly; check it
  against `_KNOWN_INTENTIONAL_CYCLES` before logging.

The cycle stays. Targeted noise goes. General cycle warnings
keep working — which is what they're there for.

During every bag-pipeline schema export (Stage 3 of load-cifar10,
plus any `cache_dataset`, `download_dataset_bag`, or similar
operation), the user sees this block emitted repeatedly:

```
SAWarning: Cannot correctly sort tables; there are unresolvable
cycles between tables "deriva-ml.Dataset, deriva-ml.Dataset_Version",
which is usually caused by mutually dependent foreign key
constraints. Foreign key constraints involving these tables will
not be considered; this warning may raise an error in a future
release.
  for sql_table in metadata.sorted_tables:
WARNING:deriva.bag.loader:Breaking cycle in FK dependencies:
  deriva-ml.Dataset -> deriva-ml.Dataset_Version -> deriva-ml.Dataset
WARNING:deriva.bag.loader:Breaking cycle in FK dependencies:
  deriva-ml.Dataset -> deriva-ml.Dataset_Version -> deriva-ml.Dataset
```

This block fires ~6× during a single `load-cifar10` invocation —
once per dataset bag built during Stage 3.

**Status:**
- The warning is benign — the bag-loader has manual cycle-breaking
  that handles the situation that SAWarning is complaining about.
- The FK cycle `Dataset ↔ Dataset_Version` is intentional in the
  deriva-ml schema (a Dataset has `current_version` FK to
  Dataset_Version; a Dataset_Version has `Dataset` FK back).
  SQLAlchemy's `sorted_tables` doesn't handle cycles, so it warns.
- The user-visible WARNING-level log line is **redundant** — it's
  announcing that the bag-loader handled the very thing the
  SAWarning was complaining about.

**Repro:** Run any operation that triggers bag schema export.
Cheapest: `load-cifar10 --hostname localhost --create-catalog X
--num-images 50` — Stage 3 emits the block multiple times.

**Two-line fix (both in `deriva-py`):**

1. **`deriva/bag/schema_io.py:414`** — wrap `metadata.sorted_tables`
   in `warnings.catch_warnings()` to silence the SAWarning once
   the code is known to handle the cycle:

   ```python
   import warnings
   with warnings.catch_warnings():
       warnings.filterwarnings("ignore", category=SAWarning,
                               message="Cannot correctly sort tables")
       for sql_table in metadata.sorted_tables:
           ...
   ```

2. **`deriva/bag/loader.py`** — change `Breaking cycle in FK
   dependencies` from `logger.warning(...)` to `logger.debug(...)`.
   It's an expected operational message about an intentional
   schema feature, not user-actionable.

After both, the warning block disappears from normal user output.
Maintainers debugging schema-traversal issues can still see the
"Breaking cycle" messages by enabling DEBUG logging on
`deriva.bag.loader`.

---

## H. `deriva-ml-model-template` (the model template repo)

**Repo:** `informatics-isi-edu/deriva-ml-model-template`.

This repo's findings are mostly documentation-shaped — the template
ships docs that point users at platform behaviors, and several of
those pointers are stale relative to the actual current platform.

### H2. MCP setup section points at wrong recipe for dev users

> **Status: ✅ SHIPPED 2026-05-17.** Coordinated rewrite of
> README §4 "Set Up Claude Code (Optional)":
> - Split into 4a (install skill plugins) and 4b (connect MCP
>   server).
> - 4a now uses the new meta-marketplace
>   (`informatics-isi-edu/deriva-plugins`) and installs both
>   `deriva@deriva-plugins` and `deriva-ml@deriva-plugins`.
> - 4b has TWO recipes side by side:
>   - **Remote production server** via the published
>     `ghcr.io/.../deriva-mcp` image + stdio (the previous
>     recipe, now correctly scoped to its actual use case).
>   - **Local dev-localhost server** via
>     `claude mcp add -t http dev-localhost https://localhost/mcp
>     --client-id deriva-mcp --callback-port 8080` — the
>     dev-localhost recipe that matches what deriva-docker
>     actually runs.
> - Stale `check-versions` slash-command reference removed.
> - Cross-link to deriva-ml-mcp's §"Connecting Claude Code"
>   added as the upstream canonical definition.
> H3's NODE_EXTRA_CA_CERTS step bundled into the same rewrite
> (see H3).

`README.md` §"Set Up Claude Code (Optional)" documents wiring
Claude Code to the published `ghcr.io/informatics-isi-edu/deriva-mcp`
image via stdio transport with a `docker run -i --rm ...`
command. **This is the right recipe for a remote production
DerivaML server. It is wrong for the dev-localhost setup that
template users typically have.**

The correct dev-localhost recipe is in `deriva-ml-mcp`'s README
§"Connecting Claude Code to the dockerized server":

```bash
claude mcp add -t http dev-localhost https://localhost/mcp \
    --client-id deriva-mcp --callback-port 8080
```

**Fix shape:** either (a) cross-link the model template's MCP
section at `deriva-ml-mcp/README.md` for the dev-localhost recipe,
or (b) lift that section into the template README directly.

Bundling note: this overlaps with D1 in section D (which asks
deriva-ml-mcp to coordinate with the model template). The
template-side fix is independent and worth doing even if D1 isn't
yet addressed.

### H3. `NODE_EXTRA_CA_CERTS` requirement isn't documented anywhere

> **Status: ✅ SHIPPED 2026-05-17.** Bundled into the README §4
> rewrite (see H2). New "TLS trust for dev-localhost"
> subsection covers:
> - Why it's needed (Node.js doesn't trust the dev-localhost
>   self-signed CA by default, causing silent HTTP MCP
>   connection failures).
> - The exact `docker cp deriva-mcp-test:/usr/local/share/ca-certificates/deriva-dev-ca.crt ~/.config/deriva/deriva-dev-ca.crt`
>   extraction command.
> - The `.claude/settings.local.json` env block to add.
> - The restart-Claude-Code requirement.
> Cross-referenced from deriva-ml-mcp's README via the D1
> cross-link.

When Claude Code's HTTP MCP transport talks to a self-signed
deriva-localhost server, Node.js's TLS verification rejects the
chain unless `NODE_EXTRA_CA_CERTS` is set in the workspace's
`.claude/settings.local.json` (`env` block). The deriva-localhost
CA cert is at
`/usr/local/share/ca-certificates/deriva-dev-ca.crt` inside the
deriva-webserver container.

Neither the model template README nor any deriva-ml-skills doc
mentions this. New users get a confusing TLS failure or an MCP
server that silently can't reach the HTTPS endpoint.

**Fix:** add a single subsection to README §"Set Up Claude Code
(Optional)" (or to the new dev-localhost recipe per H2) that
covers:

1. Extracting the CA cert: `docker cp deriva-mcp-test:/usr/local/share/ca-certificates/deriva-dev-ca.crt ~/.config/deriva/deriva-dev-ca.crt`
2. Setting `NODE_EXTRA_CA_CERTS=$HOME/.config/deriva/deriva-dev-ca.crt`
   in `.claude/settings.local.json`'s `env` block.
3. Why: Claude Code's MCP HTTP transport runs in Node.js, which
   uses its own CA bundle. Without the trust anchor, the MCP
   connection fails silently.

This is closely related to F4 (the Python-side
`InsecureRequestWarning` against the same self-signed cert). Both
should be documented together — H3 from the Claude Code side, F4
from the Python side.

---

## G. Cross-cutting / test-plan / process

These don't live in any single repo but should be reflected in the
test plan and onboarding docs.

### G1. The E2E test plan's dual-channel verification is incomplete

`docs/superpowers/specs/2026-05-13-e2e-platform-test-design.md` §4
"Test phases" defines the dual-channel verification as:

- Direct: `deriva-py` / `deriva-ml` Python API
- Indirect: `deriva_ml_*` MCP tools

It does NOT mention MCP **resources**. This is the most important
plan correction from Phase 1 alone: indirect verification should
exercise both tool paths AND resource paths, and the diff against
direct should cover both.

Update §4 and §6 ("Direct vs indirect verification — channels") to
treat resources as first-class.

### G3. Self-finding (agent process)

When a server installs instructions, read them before its first
use, not after the first failure. The deferred-tool mechanism
makes it tempting to skip prompt-fetching because the tools "just
work." But "just work" optimizes for the easy path, not the right
path.

This is in the journal as a self-tagged `#skill-issue`. The
corresponding fix is B1 above (a skill that enforces this).

### G4. `uv lock --upgrade-package <name>` silently no-ops on package-name mismatch

The package commonly referred to as `deriva-py` is named `deriva`
in `pyproject.toml` and `uv.lock`. When I ran:

```
uv lock --upgrade-package deriva-py
```

`uv` silently produced zero output and zero changes — no warning,
no error, no "unknown package" hint. Only when I read the lock and
realized the package is `deriva` (not `deriva-py`) and re-ran with
the correct name did the upgrade actually happen.

**Severity:** low — once you know about it. Easy to miss.

**Fix shape:** this lives in `uv` upstream (astral-sh/uv), not in
any Deriva repo. Worth filing an issue there:
"`uv lock --upgrade-package <name>` should warn if `<name>` doesn't
match any package in the lock."

**Workaround for now:** document in the workspace `CLAUDE.md` (or a
deriva-ml-context note) that the package name is `deriva` in
lockfiles, even though everyone says `deriva-py` in conversation.
Save the next maintainer the lookup. Single-line addition.

### G5. Claude Code: mid-session `.mcp.json` edits don't propagate to live conversation

After hand-editing `.mcp.json` to add the deriva MCP server,
`claude mcp list` (CLI level) immediately reflected the new
server. But the **live Claude Code conversation** didn't surface
the new tools until I restarted Claude Code. `ListMcpResourcesTool({server: "deriva"})`
returned "Server 'deriva' not found" even though `claude mcp list`
showed ✓ Connected.

Conclusion: Claude Code snapshots the MCP server set from
`.mcp.json` + `~/.claude/mcp.json` at conversation start. Mid-
conversation file edits update CLI-level state but do NOT register
servers into the running conversation context. A restart is
required.

**Severity:** moderate — costs the user one restart, and the
failure mode is silent (looks like the server is wired but the
tools just aren't there).

**Fix shape:** this is a Claude Code feature gap, not a Deriva
issue. Two possible asks for the Claude Code team:

1. Auto-reload MCP server set on `.mcp.json` change (matches the
   "auto-detected" claim in their network-config docs).
2. Surface a clear "MCP config changed — restart Claude Code to
   pick up changes" notice when the file watcher detects an edit.

**Workaround for now:** a workspace-`CLAUDE.md` note: "If you
edit `.mcp.json` mid-session, restart Claude Code (Ctrl-C,
re-launch from the same directory) before the new server's tools
are usable."

---

## Section addendum: missing skill (was B4 from earlier journal note)

### B4. New skill: `clone-catalog-slice` (deferred)

> **Status: ✅ SHIPPED in deriva-ml-skills v1.3.5 as `setup-ml-catalog`.**
> Broader scope than originally proposed: covers BOTH the
> from-scratch flow (`create_ml_catalog()` + a phased loader)
> AND the clone-slice flow (`clone_via_bag()` with anchors). The
> name reflects the unified bootstrap-moment framing. Includes
> explicit "why not `clone_catalog`?" guidance distinguishing
> it from the whole-catalog same-server tool.

> **Status: FIXED — closed 2026-05-17.** Addressed by the
> [`setup-ml-catalog` skill](https://github.com/informatics-isi-edu/deriva-ml-skills/blob/main/skills/setup-ml-catalog/SKILL.md)
> in `deriva-ml-skills`, merged as
> [PR #16](https://github.com/informatics-isi-edu/deriva-ml-skills/pull/16).
> The skill covers the same workflow this finding requested (clone
> a slice of an existing catalog into a fresh destination) plus the
> sibling "create from scratch via `create_ml_catalog` + a phased
> loader" path. Branch 2 of the new skill walks the three-step
> sequencing the finding flagged: create the destination catalog,
> install the deriva-ml schema, then call `clone_via_bag` with
> anchors that define the slice. The skill teaches anchor assembly
> as its own decision point (pick your roots — Dataset, Subject,
> Experiment, Workflow) and explains why the upstream
> `clone_via_bag` defaults produce a working ML catalog with
> bounded scope (the `terminal_tables` asymmetry follows outbound
> FKs for full provenance, blocks inbound FKs to prevent cross-
> anchor over-fetch). Naming chose `setup-ml-catalog` over
> `clone-catalog-slice` so the skill could cover both bootstrap
> paths (from-scratch and from-slice) under one user-facing entry
> point.

This was flagged in the journal but not yet captured in this
findings doc.

**Repo:** `deriva-ml-skills`.

**What:** A new tier-2 skill that walks the user through cloning
a slice of an existing catalog into a fresh test/staging catalog,
using `deriva_ml.catalog.clone.create_ml_workspace` (or
`clone_via_bag.clone_via_bag` for finer control).

**Why:** This is a common workflow for staging tests, demos, and
branched experiments — copy a subset of a production catalog into
a new one, with the deriva-ml schema bootstrapped. The mechanics
exist (`create_ml_workspace` in deriva-ml), but the workflow has
non-obvious sequencing: create a fresh ERMrest catalog first, init
the ml_schema in it, *then* call `create_ml_workspace` against it.
`create_ml_workspace` does NOT create its destination catalog —
the docstring explicitly says so but a user reading the API
shape would easily assume it does.

A skill encodes the three-step workflow so the agent doesn't have
to reconstruct it from primitives.

**Note:** this finding is **deferred** — it's a new skill effort
that's significant enough to warrant its own session. Adding it
here so it doesn't get lost between the journal and any GitHub
issues filed.

---

## Priority suggestion

After the 2026-05-17 doc-add batch, the remaining open work is:

1. **G1** (test plan: resources as first-class) — corrects the
   model-template's E2E test spec/plan so future phases exercise
   the right surface. Resumes when Part B resumes.
2. **E1** (`ml.refresh_schema()` doesn't suppress the
   stale-cache warning) — needs investigation of why explicit
   refresh doesn't clear the warning's underlying check.
3. **E3** (`feature_values()` returns a generator) — small API
   change; either standardize to list with optional streaming
   variant, or rename to `iter_feature_values`.
4. **F4** (`InsecureRequestWarning` against localhost) —
   either honor `REQUESTS_CA_BUNDLE` cleanly or document the
   silence-this hint.
5. **C5** (per-server `<system-reminder>` attribution) —
   primarily Claude Code rendering; minimal Deriva work.
6. **G4 / G5** (uv + Claude Code upstream issues) — file
   issues with those projects; no Deriva-team code work needed.

(Items B1, B2, B3, B4, D1, D2, E2, E4, E5, F1, F2, F5, G3, H2,
H3, plus C1c partial and Appendix 3 are shipped — see the audit
table.)

This is a suggestion, not a mandate. The repo owners can re-order
based on local priorities and bandwidth.

---

## Appendix 1 — Draft text: "RESOURCES VS TOOLS" addition to `deriva_ml_getting_started`

**Target file:** the `deriva_ml_getting_started` prompt body in
`deriva-ml-mcp` (search for the existing "PAGINATION CONTRACT"
section and add this immediately after it). One source of truth
for every MCP client.

Draft text to insert:

````
RESOURCES VS TOOLS
------------------
Every browseable group of entities in this plugin (datasets,
workflows, executions, features, ...) is exposed two ways:

- A **resource template** — single read, returns a bounded
  snapshot (≤1000 rows), suitable for verification, introspection,
  "show me what's there." URIs have the shape
  ``deriva://catalog/{hostname}/{catalog_id}/ml/<group>``.
- A **list tool** (``deriva_ml_list_<group>``) — paginated,
  cursor-based, suitable for filtered queries and full
  enumeration beyond the snapshot bound.

**Default: prefer the resource when you want "all of X" as a
snapshot.** It is one call, no preflight, no cursor handling.
The response includes ``truncated`` and ``next_after_rid``; when
``truncated`` is true, switch to the corresponding ``list`` tool
for full cursor pagination.

**Use the tool when:** you need a filtered subset, you need more
than 1000 rows, the snapshot is paginated and you must drill
beyond the first page, or you are mutating state (no resource is
ever a write surface).

**Pairings:**

| Snapshot resource | Paginated tool |
|---|---|
| ``deriva://catalog/.../ml/datasets`` | ``deriva_ml_list_datasets`` |
| ``deriva://catalog/.../ml/dataset/{rid}`` | ``deriva_ml_get_dataset`` |
| ``deriva://catalog/.../ml/dataset/{rid}/members`` | ``deriva_ml_list_dataset_members`` |
| ``deriva://catalog/.../ml/dataset/{rid}/spec`` | ``deriva_ml_get_dataset_spec`` |
| ``deriva://catalog/.../ml/workflows`` | ``deriva_ml_list_workflows`` |
| ``deriva://catalog/.../ml/workflow/{rid}`` | ``deriva_ml_get_workflow`` |
| ``deriva://catalog/.../ml/executions`` (if exists) | ``deriva_ml_list_executions`` |
| ``deriva://catalog/.../ml/features`` (if exists) | ``deriva_ml_list_features`` |
| ``deriva://catalog/.../schema`` | ``list_schemas`` + ``get_schema`` |
| ``deriva://catalog/.../tables`` | (no direct list tool — use the resource) |
| ``deriva://catalog/.../table/{schema}/{table}`` | ``get_table`` |

(Resource template list complete as of the version of the server
in front of you — call ``resources/templates/list`` for the
authoritative current set.)

**Anti-pattern.** Reaching for ``deriva_ml_list_<x>`` when you
just want a snapshot of all-of-X is a process smell: you pay the
preflight + page cycle for data the resource would have returned
in one read. The pagination contract is for cases where the data
genuinely doesn't fit in one snapshot — not as a default.
````

Estimated insertion size: ~50 lines including the table. Sits
naturally between the existing "(HOSTNAME, CATALOG_ID) RULE"
section and the "PAGINATION CONTRACT" section, or immediately
after "PAGINATION CONTRACT."

---

## Appendix 2 — Draft skill: `using-deriva-mcp` (deriva-ml-skills)

**Target plugin:** `deriva-ml-skills`. Place under
`skills/using-deriva-mcp/SKILL.md`. Plugin marketplace
metadata adds it as a tier-2 skill alongside the existing
twenty-three.

Draft `SKILL.md` (frontmatter + body):

````markdown
---
name: using-deriva-mcp
description: "ALWAYS use this skill on the first deriva-mcp call in any conversation, or when the user asks anything about a Deriva catalog that would be answered by querying it. Read the deriva_ml_getting_started MCP prompt before reaching for tools. Prefer MCP resources for snapshot reads ('what's in this catalog?', 'list all datasets', 'show me the workflows'); prefer MCP tools only for filtered queries, paginated drilling, or mutations. Triggers on: 'list datasets', 'show datasets', 'browse catalog', 'verify catalog', 'what's in the catalog', 'check schema', any catalog inspection request, any deriva-ml-mcp call. Do NOT trigger for shell-only catalog work (load-cifar10 CLI, deriva-ml Python API only) that bypasses the MCP surface entirely."
user-invocable: false
---

# Using the deriva MCP Server

You are about to make a call against a Deriva catalog via the
`mcp__deriva__*` or `mcp__<your-server-name>__*` tools or
resources. Two things to do first:

1. **Read the `deriva_ml_getting_started` MCP prompt** if you
   haven't already in this conversation. It is the canonical
   cold-start doc for this plugin and contains:
   - The `(hostname, catalog_id)` rule (mandatory on every call).
   - The pagination contract (preflight → page → advance).
   - The **resources-vs-tools rule** (see step 2).
   - The error envelope conventions.

   Fetch the prompt via the MCP protocol's `prompts/get` method
   (or `ReadMcpResourceTool` if your client surfaces prompts as
   resources). If you've already read it in this conversation,
   skip.

2. **Pick resource vs tool correctly.** The plugin exposes both
   for every group of entities. Default to the **resource** when
   you want a snapshot of "all of X." Default to the **tool**
   when you need a filtered subset, more than 1000 rows, paginated
   drilling beyond the first page, or any write operation.

   See `deriva_ml_getting_started` §"RESOURCES VS TOOLS" for the
   full pairing table. Common pairings:

   - "List all datasets" → resource `deriva://catalog/{h}/{c}/ml/datasets`,
     not `deriva_ml_list_datasets`.
   - "Show me workflow X" → resource
     `deriva://catalog/{h}/{c}/ml/workflow/{rid}`, not
     `deriva_ml_get_workflow`.
   - "Get the schema for catalog C" → resource
     `deriva://catalog/{h}/{c}/schema`, not `list_schemas` +
     `get_schema`.
   - "Find datasets where description contains 'training'" → tool
     `deriva_ml_list_datasets` with pagination (resource has no
     filter parameter).
   - "Create / update / delete anything" → always a tool. There
     are no write resources.

## When this skill applies vs. doesn't

**Applies** to any conversation that involves reading or mutating
the catalog via the MCP surface — even if the user didn't
explicitly say "use MCP." If the user asks "verify what's in
catalog 8," that's an MCP-surface operation. If they ask "build
a model," and you reach for catalog state on the way, that's an
MCP-surface operation.

**Doesn't apply** to:
- Shell-only invocations (`load-cifar10` CLI, the model template's
  `deriva-ml-run` CLI). Those use the deriva-ml Python API
  directly and bypass the MCP server.
- deriva-ml-skills `*-context` skills that you've already loaded
  earlier in the conversation. Those provide the conceptual frame
  but not the entry-point selection rules.

## Truncation handoff

When a resource read returns `truncated: true` in its envelope,
switch to the corresponding tool. Pass the `next_after_rid` from
the resource response as the `after_rid` argument on the tool
call. Same cursor token, no resampling needed.

## Reference

- The MCP prompt `deriva_ml_getting_started` (this skill's
  source of truth — read it).
- The MCP prompt `deriva_ml_concepts` (read first if you don't
  have the DerivaML mental model yet).
- The `using-deriva-mcp` skill (this file) is the trigger; the
  prompts above are the rules.
````

**Skill design notes:**

- `description` field deliberately broad — high-confidence match
  on any catalog-inspection intent, plus explicit anti-trigger
  language for shell-only flows. The job is to fire whenever the
  agent is about to reach for the MCP surface, including cases
  where the agent might not have realized that's what they're
  about to do.
- `user-invocable: false` — this skill is automatic discipline,
  not a user-facing command. The user shouldn't have to know it
  exists.
- Body stays short. **The skill is the trigger; the prompt is
  the rule.** Resource-vs-tool decisions live in
  `deriva_ml_getting_started` so updates to the rule (new
  resources added, pairings change) propagate via the prompt to
  every MCP client.
- The "When this skill applies vs. doesn't" section is
  load-bearing — without it the skill would fire on
  shell-only template work too, which wouldn't help.

**Companion update:** `deriva-ml-context` should mention this
skill in its "always-load this" body so any session that starts
with a deriva-ml-plugin loaded has both `deriva-ml-context` and
`using-deriva-mcp` available before the first tool call. Draft
text for that update is in Appendix 3.

---

## Appendix 3 — Draft update: `deriva-ml-context` to announce `using-deriva-mcp`

**Target plugin:** `deriva-ml-skills`. Modify the existing
`skills/deriva-ml-context/SKILL.md`. This skill loads on every
session that has the `deriva-ml` plugin enabled (per its
"always-load" semantics), so it's the right place to make
`using-deriva-mcp` discoverable at session start without
requiring the agent to also load it explicitly.

### Why this companion update is needed

`using-deriva-mcp` (Appendix 2) fires automatically when the
agent reaches for a deriva MCP call. But "automatic" means "when
Claude Code's skill dispatch matches the user's intent against
the skill's description." Two failure modes that bypass that:

1. **Agent reaches for the MCP tool without an MCP-shaped user
   request.** E.g., user asks "what models are in our catalog?"
   and the agent's first instinct is to query the catalog
   directly via deriva-py. If the skill's trigger phrases don't
   include enough catalog-y wording, the skill never fires.
2. **Skill match confidence is low.** Description-based matching
   is fuzzy. A request the human reads as obviously catalog-y
   may not score high enough against `using-deriva-mcp`'s
   description for it to load.

`deriva-ml-context` is the always-on context skill. By
referencing `using-deriva-mcp` from it, the agent learns about
the skill *at session start* — even if subsequent skill dispatch
fails to match. The agent can then load `using-deriva-mcp` itself
when it recognizes the situation, rather than relying entirely on
description matching.

### Draft addition to `deriva-ml-context/SKILL.md`

Find the existing section that describes the deriva-ml-plugin's
skill surface (typically near the end of the file — the "what's
available in this plugin" overview). Add the following block.
If the context skill has a "Skills you should know about"
section, append this entry; otherwise add the section.

````markdown
## When you reach for the MCP surface

The `deriva-ml-mcp` server exposes both **resources** (snapshot
reads of bounded-size views: all datasets, all workflows, one
dataset's members, etc.) and **tools** (paginated drilling,
filtered queries, mutations). They are not interchangeable —
the correct entry-point depends on the shape of what you're
asking for.

Before your first `mcp__deriva__*` (or equivalently-named MCP
server) call in any conversation, load the **`using-deriva-mcp`**
skill. It is the trigger-companion to the
`deriva_ml_getting_started` MCP prompt and handles two cold-start
disciplines for you:

1. Ensures `deriva_ml_getting_started` and `deriva_ml_concepts`
   prompts have been read.
2. Encodes the resource-vs-tool selection rule so you don't
   default to a `list_*` tool when a snapshot resource would
   answer the question in one read.

**Routing rule:** for any user request that would be answered by
reading catalog state ("what's in this catalog?", "list the
datasets", "show me the workflows", "verify the schema",
"check feature values"), load `using-deriva-mcp` before
reaching for the MCP tools or resources directly. Skip the skill
only when the entire interaction stays on the shell/Python side
(`deriva-ml-run`, `load-cifar10`, direct `deriva-ml` library
calls in a script) and never crosses the MCP boundary.
````

### Why this lands in `deriva-ml-context` specifically

The deriva-ml plugin has two context skills:
- `deriva-ml-context` (tier 2, deriva-ml-skills) — always-load,
  carries the DerivaML conceptual frame.
- `deriva-context` (tier 1, deriva-skills) — always-load, carries
  the generic Deriva catalog frame.

`using-deriva-mcp` is a tier-2 concern (it's about the
`deriva-ml-mcp` plugin's tool/resource surface, which doesn't
exist without the deriva-ml domain). So the announcement goes in
`deriva-ml-context`, not `deriva-context`. The tier-1 context
skill stays purely about generic catalog concepts.

If a tier-1 `using-deriva-mcp-core` skill is later added for the
non-DerivaML core (`query_guide`, `entity_guide`,
`annotation_guide`, `catalog_guide` prompts and the
`get_entities`/`list_schemas`/etc. tools), THAT skill would get
announced from `deriva-context`. Two skills, two announcement
sites, no overlap.

### Size of the change

About 30 lines added to one existing file. No new file. No
plugin-marketplace metadata changes (the context skill is
already enabled). No risk to non-MCP workflows because the
announcement is conditional ("Skip the skill only when the entire
interaction stays on the shell/Python side...").

### Verification after landing

After both this companion update and the `using-deriva-mcp` skill
(Appendix 2) land, test in a fresh Claude Code conversation:

1. User asks: "what datasets are in catalog 8 on localhost?"
2. Agent should load `using-deriva-mcp` (via its description match
   OR via the `deriva-ml-context` announcement).
3. Agent should fetch `deriva_ml_getting_started` if not already
   read.
4. Agent should use
   `ReadMcpResourceTool(server=..., uri="deriva://catalog/localhost/8/ml/datasets")`
   — NOT `deriva_ml_list_datasets`.

If step 4 still ends up at the tool, the skill body or the
context-skill announcement needs further refinement.
