# `roc_analysis.ipynb` template needs an explicit `gt_execution=` knob, not just a heuristic

**Persona:** Evaluator
**Severity:** Medium
**Category:** Missing feature
**Phase:** Cross-arc synthesis (consumes Analyst-02; separable platform-shape ask)

## What happened

The Analyst had to abandon the shipped `roc_analysis.ipynb`
template's natural invocation path and write
`scripts/build_joined_wide_table.py` from scratch — not because
the notebook is unfixable, but because the *abstraction* the
notebook offers is wrong for a catalog with multiple
ground-truth-shaped executions.

The notebook's design assumes the catalog has a unique
ground-truth execution, identifiable by the row-shape heuristic
`with_confidence == 0`. On a catalog with one GT execution that
heuristic is correct. On any catalog with **either** a
loader-retry-orphaned GT execution (the case here, see
`findings/evaluator/01`) **or** legitimately multiple GT executions
(e.g., human-relabeled-after-correction, multi-annotator workflows,
gold-vs-silver), the heuristic silently picks one — and the
analysis proceeds without telling the user which.

The Analyst's one-line fix in `findings/analyst/02` (pick the
GT-candidate with the most rows) makes the heuristic better on this
catalog but doesn't fix the underlying shape: the notebook is still
*guessing* which execution is ground truth, just with a better
prior. A future catalog where a partial GT execution legitimately
has more rows than the canonical one (because the canonical one
was deliberately a subset) breaks the new heuristic the same way.

The durable fix is to take ground-truth identity as **an explicit
configuration input**, not infer it.

## Suggested shape

In `src/configs/roc_analysis.py` (or wherever the notebook config
class lives), add an optional `gt_execution: str | None = None`
field. In the notebook:

- If `gt_execution` is set, filter feature rows to that execution.
- If `gt_execution` is `None`, run the current heuristic *and*
  surface the picked execution prominently (the notebook already
  does the latter; add a warning when more than one GT-candidate
  exists).
- Make the chosen execution one of the recorded inputs of the
  notebook's analysis Execution (so the catalog's provenance shows
  *which* ground truth was used).

Then `src/configs/roc_analysis.py` could ship a `roc_toronto_hsr`
config that wires `gt_execution="HSR"` explicitly, and the
Analyst arc could use it without writing a separate script.

## Why this is separable from analyst/02 and evaluator/01

- **analyst/02** is the as-shipped notebook bug ("first by index
  order") — a one-line code fix. File as Bug.
- **evaluator/01** is the upstream loader cause that creates
  multi-GT-execution catalogs in the first place. File as Bug.
- **This finding** is the *template-shape* gap: even if the loader
  is fixed *and* the heuristic is fixed, the notebook still
  silently does the right thing when there's no ambiguity and
  silently does *something* when there is. Explicit > implicit for
  identifying ground truth.

## Why Medium / Missing feature

- The platform doesn't *fail* without this knob — the Analyst's
  workaround (standalone script) proves a path exists. But the
  notebook is the documented user-facing entry point, and "you
  have to abandon the notebook template and write Python" is a
  significant ergonomic regression for any user without a Curator
  arc to characterise the catalog first.
- The lift is modest: one optional config field + a handful of
  notebook-cell changes + a guard-and-warn block in the heuristic.

Detected during the 2026-05-28 e2e run with sibling versions:
deriva-ml v1.40.2, deriva-ml-mcp v0.5.9, deriva-mcp-core latest main,
deriva-skills v1.2.4, deriva-ml-skills v1.4.11.
