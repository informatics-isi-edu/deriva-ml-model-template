# Tacit Knowledge

This file records **tacit knowledge** — the *why*, the *intent*, and the
*background* behind decisions made about this project's models and data.

The **catalog** is the source of record for everything else: data contents,
RIDs, dataset versions, workflow URLs and checksums, executions, lineage.
Don't replicate catalog-stored facts here. Don't ask this file what's in
the catalog — query the catalog directly (resources first, tools next).
When this file *needs* to reference a catalog entity, link to it
(`deriva://catalog/{host}/{cat}/ml/...`) instead of inlining its contents.

Each entry captures a decision: what was chosen, what alternatives were
considered, what was rejected and why, and any background context a future
reader would need to evaluate whether the decision still holds.

---

## tk-001 — Bootstrap

**When:** 2026-05-27 (multipersona rerun, post-#246 row-completeness fix)
**By:** Orchestrator (Phase 0)
**Supported by:** —

### What was set up

Fresh catalog **`e2e-test-20260527`** (catalog id `93`) on `localhost`. Bootstrapped
end-to-end via `load-cifar10`:

| Phase | Command | Output |
|---|---|---|
| schema | `load-cifar10 --hostname localhost --create-catalog e2e-test-20260527 --phase schema` | catalog 93 created, domain schema in place |
| images | `load-cifar10 --hostname localhost --catalog-id 93 --num-images 1500 --phase images` | 1500 Image rows (750 train / 750 test), 1500 Image_Classification features with perfectly balanced class distribution (150 per class × 10 classes) |
| datasets | `load-cifar10 --hostname localhost --catalog-id 93 --num-images 1500 --phase datasets` | 13 datasets (see deriva://catalog/localhost/93/ml/Dataset for the full list) |

### Why `--num-images 1500`

The small-variant guard (`SMALL_TRAIN_SIZE`/`SMALL_TEST_SIZE` = 500/500)
needs `--num-images >= 1002` to keep the Toronto small-split distinct
from the full split. 1500 gives headroom; matches the catalog scale
intended for the 2026-05-27 platform-test run.

### Why this run

Validates the **deriva-ml v1.39.4 PagedFetcher row-completeness fix** (PR #246):
the prior 2026-05-26 run used `--num-images 500` and never exercised the
URL-length guard. The 2026-05-27 first attempt at `--num-images 1500`
exposed a fetcher bug where oversized GET requests were silently dropping
RIDs. PR #246 lands the chunk-loop replacement; this run is the e2e validation.

### Sibling versions pinned for this run

- deriva-ml: `v1.39.4` (`4ed88122`) — contains the #246 fix
- deriva-py: pinned via `@deriva-ml` branch (`e944ad8e`)
- deriva-ml-mcp: `v0.5.3`
- deriva-mcp-test container: rebuilt against above

### Handoff to Curator

- The catalog id, schema, image counts, and dataset RIDs are now
  visible via the standard MCP resources — query them rather than
  reading them out of this file.
- Default Hydra connection (`default_deriva`) and dataset config
  (`default_dataset` → small labeled split WD2) already point at
  catalog 93. Just `uv run deriva-ml-run --info` from this worktree.
- Class distribution is perfectly balanced (150 per CIFAR-10 class
  × 10 classes = 1500). This is post-fix #15 behaviour; if the
  Curator finds skew, that's a finding against the loader.

---
