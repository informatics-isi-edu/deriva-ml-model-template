# Orphan Complete dataset F2J is a full 1100-image duplicate, not an empty 0-children husk

**Persona:** Curator
**Phase:** Catalog characterization — auditing the dataset hierarchy on catalog 69

## What happened

tk-001 and `findings/phase0/02` describe the orphan Complete dataset
**F2J** as "0 children, unreferenced" — implying it is a harmless empty
shell left over from the partial bootstrap retry. That framing is
**incomplete and potentially misleading for a downstream user.**

Direct member count (MCP `count_table` on `Dataset_Image`):

| Dataset | Image members |
|---|---|
| H8M (live Complete)   | 1100 |
| F2J (orphan Complete) | **1100** |

And set arithmetic (`scripts/curator_verify_splits.py`) proves the two
image sets are *identical*:

```
[PASS] F2J image set == H8M image set (full duplicate)
```

So F2J is **not** an empty husk. It is a fully-populated, second
`Complete,Labeled` dataset holding the exact same 1100 CIFAR-10 images
as H8M. A user browsing `deriva_ml_list_datasets` sees two datasets
with byte-identical descriptions ("Complete CIFAR-10 dataset: 1,100
labeled images (550 train + 550 test).") and the same member count —
nothing in the listing distinguishes the live one from the orphan.

## Why "0 children" is the wrong distinguisher

The prior framing keyed on dataset *children* (nested datasets). But
neither H8M nor F2J has any dataset children — verified:

```
deriva_ml_list_dataset_relations(H8M, direction="children", recurse=True)
  -> {"children": []}
deriva_ml_list_dataset_relations(F2J, direction="both",     recurse=True)
  -> {"parents": [], "children": []}
```

The split hierarchy (KE0/KEA/QM*/RQ*/...) does **not** hang off H8M via
dataset nesting. It hangs off H8M via *provenance* (split/subsample
executions consumed H8M and produced the splits). So "0 children" is
true of H8M too and cannot be what makes F2J an orphan.

What actually distinguishes F2J: **nothing in the catalog's lineage
graph derives from it.** `deriva_ml_get_lineage(KE0)` traces back to
execution H7M, which consumed **H8M** — not F2J. F2J is the lineage
root of *nothing*; H8M is the lineage root of the entire split family.
That is the real orphan signal, and it is not visible from member
counts or child counts.

## Reproduction

1. `mcp__mcp-localhost__deriva_ml_list_datasets(localhost, 69)` — note
   two `Complete,Labeled` datasets (H8M, F2J) with identical descriptions.
2. `count_table(Dataset_Image, filters={"Dataset":"F2J"})` -> 1100.
3. `count_table(Dataset_Image, filters={"Dataset":"H8M"})` -> 1100.
4. `deriva_ml_get_lineage(KE0)` -> producing execution consumed H8M.
5. `DERIVA_ML_ALLOW_DIRTY=true uv run python scripts/curator_verify_splits.py`
   -> `F2J image set == H8M image set (full duplicate)` PASS.

## Notes

- I did **not** delete F2J. Deleting a dataset is a destructive op
  requiring explicit user authorization (and the e2e plan reserves F2J
  as already-documented). This finding only corrects the *description*
  of the hazard, not the disposition.
- Concrete hazard for downstream personas: a user who picks "the
  Complete dataset" by description alone has a 50/50 chance of pinning
  F2J, which carries no lineage and would silently sever provenance for
  anything derived from it. The Modeler/Analyst handoff names **H8M**
  explicitly to avoid this.
- Possible template hardening (separate from this run): the idempotent
  `--phase datasets` retry could either reuse an existing unreferenced
  `Complete` dataset or tag the orphan's description (e.g. append
  "(superseded — do not use)") so the two are distinguishable in a bare
  listing. Filed here as an observation; not fixing mid-arc.
