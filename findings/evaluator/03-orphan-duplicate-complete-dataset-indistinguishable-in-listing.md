# Aborted-then-retried `--phase datasets` leaves an indistinguishable full-duplicate `Complete` dataset

**Severity:** Medium
**Category:** Bug (template loader idempotency) + Polish (Chaise/listing distinguishability)
**Component:** `deriva-ml-model-template` `load-cifar10 --phase datasets` idempotency
**Persona findings this consolidates:** `findings/phase0/02`, `findings/curator/01`

## What the evaluator found

The catalog holds **two** `Complete,Labeled` datasets with byte-identical
descriptions and identical 1100-image membership:

| RID | Image members | Lineage |
|---|---|---|
| H8M | 1100 | root of the entire split family (KE0/KEA/RQ*/QM*/...) |
| F2J | 1100 | root of nothing; no execution consumes it |

Evaluator verified directly: `H8M image set == F2J image set` (full
duplicate), and `get_lineage(KE0)` traces to execution H7M which consumed
**H8M**, never F2J. The associated failed execution **F1J** (workflow 474,
status `Failed`, `upload_duration: null`) is the aborted first
`--phase datasets` run that raised the `Image.filename` KeyError — it is
the debris that produced the orphan.

The Phase-0 note originally framed F2J as "0 children, unreferenced"
(implying an empty husk); the Curator correctly corrected this to "full
1100-image duplicate" (`findings/curator/01`) and tk-003 carries the
right disposition: **use H8M, never F2J; do not delete without explicit
authorization.** So the *team handled it correctly* — the finding is
about the **platform behavior that created the hazard**, not the team's
response to it.

## The real defect

The `--phase datasets` step is documented as idempotent against partial
state, but its idempotency did not reuse the `Complete` dataset created by
the aborted first run (because that run's execution never committed, and
the reuse path keys off committed state). So the successful retry created a
second `Complete` dataset rather than adopting the orphan. The net effect:
a 50/50 trap for any user who picks "the Complete dataset" by description
alone — and pinning F2J would silently sever provenance for anything built
on it, because nothing in a bare `deriva_ml_list_datasets` listing
distinguishes the live root from the orphan.

## Severity rationale

Medium: it created a real correctness hazard (provenance-severing if the
wrong RID is pinned) but did not block this run — the team identified the
live root and documented the disposition. It would be Low if the two were
distinguishable in a listing; the indistinguishability is what keeps it at
Medium.

## Suggested hardening (out of scope for the run)

- **Loader fix:** on `--phase datasets` re-run, detect a pre-existing
  unreferenced `Complete` dataset and either (a) reuse it, or (b) tag its
  description (e.g. append " (superseded — do not use)") so a bare listing
  distinguishes the two.
- **Cheaper alternative:** emit a warning when re-running over a partial
  `Complete` so the operator knows an orphan will result.

## Reproduction

See `findings/phase0/02` steps 1-4; the duplicate-membership proof is in
`scripts/curator_verify_splits.py` (`F2J image set == H8M image set` PASS)
and reconfirmed by the evaluator via direct `Dataset_Image` set arithmetic.
