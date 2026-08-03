---
type: Log
title: Tacit Knowledge — DerivaML Model Template
description: >
  The why behind this project's DerivaML decisions — rationale, dead ends, and
  cross-discipline consequences that the catalog records but does not explain.
  Append-only; each entry is a dated tk-… decision record.
tags: [tacit-knowledge, provenance, deriva-ml]
---

# Tacit Knowledge

**Tacit knowledge** is knowledge gained through experience and practice —
context-dependent, and hard to codify or transfer through documentation
(Polanyi). Its fully embodied core (intuition, pattern-recognition earned by
doing) cannot be written down; what *can* be captured is the **externalizable
shell** around it — the decisions made, the alternatives weighed and rejected,
and the *why* a future teammate would otherwise have to reconstruct or re-learn
the hard way. This file captures that shell and points at the rest.

It records the *why*, the *intent*, and the *background* behind decisions made
about this project's models and data.

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

## tk-2026-08-03-bump-deriva-ml-1.55.1

**Decision.** Bumped the pinned deriva-ml in `uv.lock` from `1.45.0.post1`
to `1.55.1` (git HEAD `d2c5bc8`). Only `uv.lock` changed — no template
source edits were required.

**Why no source changes.** The template consumes deriva-ml through a fixed,
narrow surface: 14 imported symbols (`DerivaML`, `DerivaMLConfig`,
`DerivaMLModel`, `multirun_config`, `run_notebook`, `BaseConfig`,
`notebook_config`, `load_configs`, `Workflow`, `with_description`,
`DerivaBaseConfig`, `base_defaults`, `run_model`, `DatasetSpecConfig`) plus
the `deriva-ml-run` / `deriva-ml-run-notebook` CLI entry points. All 14
symbols and both entry points still exist at their same import paths in
1.55.1 — verified against a fresh checkout, not assumed. All 10 config
smoke tests pass and `deriva-ml-run --list-configs` resolves cleanly on
1.55.1.

**The one breaking change in range doesn't touch us.** Between 1.45 and
1.55.1, `split_dataset` gained a required `partition_by` parameter (it now
raises `ValueError` on the ambiguous `row_per != element_table` shape that
used to cause silent train/test leakage). The template never *calls*
`split_dataset` — dataset splitting is a user operation driven by the
`/deriva-ml:dataset-lifecycle` skill, not baked into the skeleton — so the
signature change has no effect on template code. A future teammate wiring
`split_dataset` into a model repo built from this template must pass
`partition_by="element"` (safe per-element) or `"row"` (legacy) when
`row_per` differs from `element_table`.

**Pin floor left at `>=1.42,<2.0`.** The `pyproject.toml` floor documents
when `subsample()` / `Split_Partition` landed; it's still a valid minimum.
Not raised, because the template doesn't require any 1.46+ behavior at the
source level — the lock, not the floor, is what pins the concrete version.

**Dirty-tree docs checked, left as-is.** 1.55.1 auto-excludes
`findings/`, `outputs/`, `.scratch/` from the provenance dirty-tree check
(and adds `DERIVA_ML_DIRTY_CHECK_IGNORE`). The template already git-ignores
`outputs/` and `multirun/`, so those never appeared in `git status` anyway
— the new auto-exclusion changes nothing for this skeleton, and the
`DERIVA_ML_ALLOW_DIRTY` guidance in README/CLAUDE.md stays correct.
