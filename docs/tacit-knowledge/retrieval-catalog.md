---
type: RetrievalCatalog
title: Tacit Knowledge — retrieval catalog
description: >
  Derived lookup over tacit-knowledge.md — one greppable row per entry.
  Cache, not record: rebuilt whole by the capture side-effect; every row
  mirrors the Log and originates no authority. Queried by tk_lookup.py
  (hand-grep is the fallback); never loaded whole. Not an OKF index.md.
generated_from: tacit-knowledge.md
generated_at: (not yet built)
generator: capture-tacit-knowledge rebuild
covers_through:
  id: (none)
  offset: 0
tags: [tacit-knowledge, retrieval-catalog, deriva-ml]
---

# Tacit Knowledge — Retrieval Catalog

_No entries indexed yet. Rebuilt whole as a silent side-effect of capture once
entries accumulate past the threshold. Queried by `tk_lookup.py`; hand-grep is
the fallback. See `skills/capture-tacit-knowledge/references/index-and-retrieval.md`._

## Rows

**One entry per line, greppable.** Each row carries the entry's `tk-NNN`, **all
anchor scopes** it applies at (instance RID *and* type *and* abstraction *and*
process/skill — so the generalization walk's widened greps all hit), and its
**keywords including topic-CV synonyms** (so a query using a synonym still
matches). `superseded-by` mirrors the entry's tombstone edge (D2). Cost of
finding candidates is O(matches), not O(entries).

| tk-NNN | anchors (all scopes) | keywords (+ synonyms) | superseded-by |
|---|---|---|---|

_Example populated row (illustrative):_
`[tk-042](../../tacit-knowledge.md#tk-042)` | execution 8KG · Dataset_Type=Animal_Subset · Dataset · execution-lifecycle | model-configuration · label-smoothing · regularization | (none)

## candidate-terms (proposed, awaiting human review)

_none_
