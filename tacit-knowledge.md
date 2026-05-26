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

### tk-001 — Bootstrap: e2e-test-20260526 catalog 18 created for 3-persona multipersona run
**When:** 2026-05-26T10:55:00-07:00
**By:** Carl Kesselman (https://localhost/auth/realms/deriva/ec40f483-26ae-4a8b-aa24-5155ca94cb22)

Created localhost catalog **id 18** (alias `e2e-test-20260526`) for the
2026-05-26 multipersona e2e run (Curator → Developer → Analyst).
Bootstrap was scripted via `load-cifar10` in three phases:

```
uv run load-cifar10 --hostname localhost --create-catalog e2e-test-20260526 --phase schema
uv run load-cifar10 --hostname localhost --catalog-id 18 --num-images 500 --phase images
uv run load-cifar10 --hostname localhost --catalog-id 18 --num-images 500 --phase datasets
```

500-image CIFAR-10 sample — `--num-images 500` chosen for fast iteration
across all three persona arcs; matches the prior 2026-05-25 run for
cross-run comparability. Class distribution is balanced 50-per-class
across all 10 CIFAR-10 classes (post-#15 fix, no bird+ship skew).

13 datasets created by the loader (Toronto split + small variant +
training-derived labeled split via `split_dataset(seed=42)` and a small
variant via `seed=123`):

| Group | Complete | Split | Training | Testing |
|---|---|---|---|---|
| Toronto (Complete=`96E`) | 96E | 96R | 970 | 97A |
| Toronto small | — | 97T | 982 | 98C |
| Training-derived labeled (seed=42) | — | C7Y | C86 | C8G |
| Training-derived labeled small (seed=123) | — | CRR | CS0 | CSA |

All RIDs wired into `src/configs/datasets.py` at v0.1.0.post1.dev1
(dev versions — release via `ml.lookup_dataset(rid).release(minor=True)`
if downstream wants pinned snapshots). `src/configs/deriva.py`
`default_deriva.catalog_id=18`. Both edits are `[E2E-DROP]` commits
on `e2e-test/2026-05-26` and drop out at wrap-up.

Cross-channel verification (direct deriva-ml + MCP `deriva_ml_*` tools)
agreed on dataset count (13), feature count (1: `Image_Classification`),
feature value count (500), and class distribution.

**Sibling versions for reconstructability:**
- deriva-ml v1.39.2 (39d88f39)
- deriva-ml-mcp v0.5.1 (5c0390a)
- deriva-mcp-core 376df57 (no tag — running 0.1.0 series HEAD)
- deriva-skills v1.2.3 (ecdb9ab)
- deriva-ml-skills v1.4.7 (de96fac) — includes tk-NNN/**When**/**By**/**Supported by** entry headers + semantic-awareness bridge
- Docker `deriva-mcp-test` container rebuilt 2026-05-26T17:34, runs deriva-ml 1.39.2 + deriva-ml-mcp 0.5.1.

**Implications for collaborators:** Curator inherits a clean catalog
with all built-in datasets, no synthetic/curated subsets, no model
training runs, no analysis assets. The Curator's first job is to audit
this state, decide which dataset(s) the Developer will actually train
on, and add at least one curated variant. Open questions left for the
Curator (not directives — questions to answer through their own work):
why is `default_dataset` pointed at `CRR` rather than `C7Y`? Is the
seed=42 vs seed=123 split duplication intentional or vestigial? Are
all 13 datasets going to be used, or are several of them dead branches
that should be pruned/deleted before downstream consumption?
