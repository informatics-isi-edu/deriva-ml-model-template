# E2E Platform Test Session Journal — 2026-05-19

This is the running journal for the end-to-end platform test session
that begins with a clean restart of Phase 1. Each phase appends a
new section. Findings are tagged inline; see
`docs/findings/2026-05-16-phase-1-improvements.md` for cumulative
findings and `docs/bugs/` (workspace-level) for filed bug write-ups.

---

## Session setup

- **Workspace:** `/Users/carl/GitHub/DerivaML/deriva-ml-model-template-e2e`, branch `e2e-test/2026-05-18`
- **Plan:** `docs/superpowers/plans/2026-05-13-e2e-platform-test.md` (in the main repo, not this worktree)
- **deriva-ml version:** `1.36.5.post2+g1369b34ef` (check_auth removed)
- **deriva-mcp-core version:** `0.1.0` at `08bb642` (remap_hostname public API)
- **deriva-ml-mcp version:** `3.3.5.dev13+gf60f2d54b`
- **MCP server:** dev-localhost, OAuth via credenza, anonymous mode disabled
- **deriva-mcp-test container:** rebuilt and healthy with the above pins

Pre-flight fixes that landed today before Phase 1 started:

| Repo | Commit | Change |
|------|--------|--------|
| deriva-mcp-core | `08bb642` | Expose `remap_hostname()` and `remap_url()` as public plugin API |
| deriva-ml | `1369b34e` | Drop `check_auth` and the `get_authn_session()` probe |
| deriva-mcp (legacy) | `df4dd64` | Drop `check_auth=True` kwarg |
| deriva-ml-mcp | `f60f2d5` | Use public `remap_hostname()` in `get_ml()` |
| deriva-ml-model-template (template) | `4a55da8` | Drop `check_auth=True` + uv.lock bump |
| deriva-ml-model-template-e2e | `116092b` | Drop `check_auth=True` |
| deriva-docker (local edit, not committed) | — | `DERIVA_MCP_ALLOW_ANONYMOUS=false` so server returns 401+WWW-Authenticate |
| ~/.claude.json | — | Consolidated `dev-localhost` MCP config to user scope; cleared 5 project-scope duplicates |

---

### 2026-05-19 13:38 — Phase 1: catalog bootstrap

**Skill tried:** `deriva-ml:setup-ml-catalog` (1.3.5 replacement for the
plan-referenced `route-project-setup` from 1.2.1, which no longer
exists). Trigger description matches the user-style phrasing
"Set up a fresh CIFAR-10 catalog on localhost for end-to-end
testing" exactly.

**Routed to:** none — skill is marked `disable-model-invocation: true`,
so the agent cannot invoke it; only an explicit `/deriva-ml:setup-ml-catalog`
slash command from the user can. Fell back to CLI per the plan's
"whether routed by skill or by fallback" branch. **Design observation
(not bug):** the gate creates tension between autonomous test runs
and the plugin's user-confirmation model. Worth thinking about a
designated "test-mode" override or an agent-invocable companion skill
that explicitly delegates to `setup-ml-catalog`'s underlying tool calls.

**MCP tools used:**
- `deriva_ml_list_datasets` (preflight + full page)
- `deriva_ml_list_features` (with `table="Image"`)
- `list_vocabulary_terms` (deriva-mcp-core, for `e2e-test-20260520.Image_Class`)
- `list_schemas` (deriva-mcp-core)

**Plan vs reality (MCP tool surface):** the plan referenced
`deriva_ml_list_vocabularies` but no such tool exists. Equivalents are
`list_vocabulary_terms` + `lookup_term` from deriva-mcp-core. The plan
also names the resource path; for Phase 1 verification we used the
tool path only (resources tested separately in earlier diagnostics).
`#plan-drift` — plan needs a refresh against the current MCP surface.

**Catalog created:** id `20`, domain schema `e2e-test-20260520`,
500 images (250 train + 250 test), 500 features (Image_Classification
per image), 13 datasets (RIDs 84W, 856, 85E, 85R, 868, 86G, 86T, B68,
B6G, B6T, BQW, BR4, BRE), 10 Image_Class vocab terms.

**Direct/indirect diff:**

| Check | Direct | Indirect | Agreement |
|-------|--------|----------|-----------|
| Schemas | `public, WWW, deriva-ml, e2e-test-20260520` | same | ✓ |
| Dataset count | 13 | 13 | ✓ |
| Dataset RIDs + descriptions | (above) | match | ✓ |
| `Image_Class` vocab terms | 10 (airplane…truck) | 10 same | ✓ |
| `Image_Classification` feature | present BUT `find_features()` returned **3 duplicates** | `count: 1` | ✗ **#diff** (deriva-ml bug) |

**Findings:**

- **`#bug` (deriva-ml `find_features` duplicates):** `ml.find_features()`
  (no `table=` arg) returns one Feature per traversal direction in
  the association graph. For an `Image.Image_Classification` feature
  it returns 3 copies. Filed at
  `/Users/carl/GitHub/DerivaML/docs/bugs/2026-05-19-find-features-duplicates.md`.
  MCP-indirect path is unaffected (always passes a `table=` arg).
  Severity: low — needs a dedup pass before return.
- **`#bug` (deriva-ml schema-cache stale after load):** Cache file at
  `~/.deriva-ml/<host>/<catalog>/schema-cache.json` written by
  `load-cifar10` reflects an intermediate snapshot, not the final
  post-load state. Next `DerivaML()` construction loads the stale
  cache, warns, and proceeds — user has to call `refresh_schema()`
  to see Image table, vocab, datasets. Filed at
  `/Users/carl/GitHub/DerivaML/docs/bugs/2026-05-19-schema-cache-stale-after-load.md`.
  Severity: medium — silent staleness behind a warning.
- **`#skill-issue` (1.2.1 → 1.3.5 plugin migration):** Plan still
  references skills (`route-project-setup`) that no longer exist;
  the 1.3.5 plugin replaced them with the gated `setup-ml-catalog`
  pair. Plan refresh needed.
- **`#plan-drift` (MCP tool name vs plan):** Plan references
  `deriva_ml_list_vocabularies` which is not in the current MCP
  surface. Use `list_vocabulary_terms` (deriva-mcp-core).

**Decisions:**

- Used `refresh_schema()` inline in the direct check to work around
  the stale cache. Did NOT change `load-cifar10` to fix it — that's a
  follow-up after the test session.
- Skipped the `deriva_ml_list_features` `preflight_count=True` step
  since the count is small (1 feature on Image). Indirect call with
  `table="Image"` returned the full record directly.

**Repoint:**

- `deriva_localhost.py`: `localhost_1407` → `localhost_20` (catalog 20,
  schema `e2e-test-20260520`).
- `datasets_localhost.py`: all 13 dataset RIDs updated, version
  `0.1.0.post1.dev1`.
- `assets_localhost.py` + `roc_analysis_localhost.py`: cleared stale
  RIDs (populated by Phase 2 / 4). All `deriva_ml` refs point at
  `localhost_20`.
- `tests/test_configs_load.py` passes after the repoint.
- DROP commit: `4769eba` on branch `e2e-test/2026-05-18`.

---

### 2026-05-19 16:30 — Inter-phase: bug-fix sweep + clean-baseline rebuild

After Phase 1 surfaced bugs B1–B7 (find_features dup, stale schema
cache, bag-loader DatabaseModel collision, etc.), we paused Phase 2
to fix all root-cause bugs rather than work around them. Direct
quote from user: "Lets fix all bugs. No work arounds." Then "I want
the code as clean and simple as possible" — drove removal of two
older interim commits (`de19aaf` defensive `refresh=True` and
`2f8f3e0` snaptime-based path-builder cache).

**Architectural fixes that landed:**

| Repo | Commit | Change |
|------|--------|--------|
| deriva-py | `3a6a7bb` | Binding-layer `purge_cache_by_prefix("/schema")`; identity-tied schema + path-builder caches; removed snaptime tracking |
| deriva-py | `9d6daae` | `BagDatabase.model` regular attribute (unblocks `DatabaseModel(BagDatabase, DerivaModel)` MRO collision) |
| deriva-py | `ed5ee69` | `Model.fromcatalog` uses `getCatalogSchema()` |
| deriva-ml | `1f2e722` | B1+B2: `find_features` dedup; removed all `catalog.get("/schema").json()` |
| deriva-ml | `0f14de7` | T1: route all schema reads through `getCatalogSchema()` |
| deriva-ml-mcp | `2116130` | uv.lock bump to deriva-py `ed5ee69` |
| deriva-ml-model-template-e2e | `3656198` | uv.lock bump to deriva-py `ed5ee69` + deriva-ml `0f14de7` |
| deriva-ml-model-template (skill-audit) | `fe408f2` | uv.lock bump (same SHAs) |

**Cache-invalidation invariant** (verified by 17 unit tests in
`deriva-py/tests/deriva/core/test_ermrest_catalog.py`):

- Schema-mutation calls (`catalog.post/put/delete` to `/schema/...`)
  invalidate the binding HTTP cache for `/schema*` URLs.
- Data writes (`/entity/...`) do NOT invalidate the schema cache.
- `getCatalogSchema()` is identity-memoized: returns the same parsed
  dict for the same `Response` object; only re-parses when the
  binding fetches a new response (which happens on schema mutation
  or 304-driven re-fetch).
- `getPathBuilder()` ties to the parsed-dict identity: rebuilds only
  when `getCatalogSchema()` returns a new dict.
- **No cache state is visible to deriva-ml.** All invalidation lives
  in deriva-py at the binding + ermrest_catalog layer.

**Localhost catalog cleanup:** Deleted orphan catalogs 2, 3, 4
(remnants of prior debugging sessions). Catalog 1 (system) and
catalog 20 (current Phase 1 session) preserved.

**Container rebuild:** `deriva-mcp-test` rebuilt via
`deriva-ml-mcp/scripts/rebuild-deriva-docker-mcp.sh`. Verified
installed versions in the container:

- `deriva` → `ed5ee69` (deriva-py)
- `deriva-ml` → `0f14de7e` (main)
- `deriva-mcp-core` → `08bb642` (file copy)
- `deriva-ml-mcp` → `2116130` (main)

**No workarounds preserved.** All previous interim fixes
(`de19aaf`, `2f8f3e0`) removed from history; replaced with a
single clean commit in deriva-py.

**Validity caveat for catalog 20.** Catalog 20 was created by
`load-cifar10` running against the **pre-fix** pins
(`5cd01a25` / `1369b34e`). The schema doc is correct (no bugs land
in the catalog state) but the load path itself was not exercised
against the cache-fixed code. For a fully valid clean-state
assessment, Phase 2 should recreate the catalog with the rebuilt
container + the new pins via `uv run load-cifar10`.

---
