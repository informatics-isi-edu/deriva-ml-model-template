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

**When:** 2026-05-27 (third-pass multipersona run, post-path_walker + cifar10_cnn refactor)
**By:** Orchestrator (Phase 0)
**Supported by:** —

### What was set up

Fresh catalog **`e2e-test-20260527c`** (catalog id `95`) on `localhost`.
Bootstrapped via `load-cifar10` in three phases (schema → images →
datasets). 1500 Image rows (750 train + 750 test), 1500
`Image_Classification` features, perfectly balanced class
distribution (150 per CIFAR-10 class × 10 classes). 13 datasets
created — the canonical hierarchy:

- `JZ8` Complete (1500) — root container
- `JZJ` Split → `JZT` Training (750) + `K04` Testing (750)
- `K0M` Small_Split → `K0W` Small_Training (500) + `K16` Small_Testing (500)
- `TX0` Labeled_Split → `TX8` Labeled_Training (600) + `TXJ` Labeled_Testing (150)
- `WD2` Small_Labeled_Split → `WDA` (400) + `WDM` (100)

All RIDs wired into `src/configs/datasets.py`. Default Hydra
connection (`default_deriva`) and dataset config (`default_dataset`
→ `WD2`) already point at catalog 95.

### Sibling versions

- deriva-ml: `v1.39.4` (`4ed88122`) — includes PR #246 PagedFetcher
  row-completeness fix, PR #243 resolver dedup, PR #237 defensive
  one-liners
- deriva-py: pinned via `@deriva-ml` branch at `e944ad8e` —
  includes `SchemaPathWalker` (PR #263)
- deriva-ml-mcp: `v0.5.4`
- deriva-ml-skills: `v1.4.8` (latest e2e polish + concepts.md
  cleanup)
- deriva-mcp-test container: rebuilt against above

### Notable platform changes since 2026-05-27 (first pass)

- **cifar10_cnn refactor (PR #37)**: the runner is now ~730 lines
  with three clearly-labeled sections (ML primitives / DerivaML
  harness / entry point). Eval logic collapsed into one
  `evaluate()` primitive; predict logic into `predict_batch`;
  bag-to-loader dispatch into one `build_loaders` function. The
  Developer arc will exercise this refactored surface.
- **PagedFetcher fix (PR #246)** prevents the 50% row-loss bug
  that blocked the first 2026-05-27 attempt at `--num-images 1500`.
- **path_walker pin (PRs #38/#59)** unblocks the denormalize
  planner's runtime import; was the blocker on this run's second
  attempt.

### Handoff to Curator

- Catalog id, schema, image counts, and dataset RIDs are visible
  via the standard MCP resources — query them, don't read them
  out of this file.
- Class distribution is perfectly balanced (150 per class × 10).
  Any skew the Curator finds is a finding against the loader.
- `default_dataset` is `WD2` (small labeled split) — the typical
  fast-run choice. Override with `datasets=cifar10_labeled_split`
  (or others) for larger runs.

---
