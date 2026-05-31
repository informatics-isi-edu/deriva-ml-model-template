# Labeled-split datasets leak against F2T but the catalog surfaces no guard

**Persona:** Evaluator (cross-arc synthesis of `findings/curator/01` + `findings/curator/02`)
**Severity:** Medium
**Category:** Bug (bootstrap-loader split-registration gap, plus a latent
catalog-modeling hazard)

## What happened

The Curator filed two findings that, read together, describe a single
data-modeling hazard with platform implications. I verified both directly
against catalog 168 and am promoting the combined platform angle to a
first-class finding.

**The data facts (re-verified directly, deriva-ml Python over
`Dataset_Image`):**

- `NEJ` (`Dataset_Type` = `Testing`+`Labeled`, 110 images): **110/110
  members are in `F2T`** (the canonical training partition); 0 in `F34`.
- `PJ4` (`Testing`+`Labeled`, 100 images): **100/100 in `F2T`**; 0 in
  `F34`.
- `NE8 ∪ NEJ` reconstructs `F2T` exactly; `PHT ∪ PJ4 ⊆ F2T`.
- `deriva_ml_list_dataset_relations(F2T, direction="both", recurse=true)`
  → parent `F2J`, **children: []**. The labeled-split roots `NE0` and
  `PHJ` are standalone — not children of `F2T`.

**The hazard.** A consumer dispatching on `Dataset_Type` (the
authoritative "what is this for" signal) reads `NEJ`/`PJ4` as held-out
test sets. They are not — they are internal validation splits *of the
training pool*. The CIFAR-10 harness dispatches each input bag to a lane
by `Dataset_Type`: a `Testing`-typed bag is held out and scored. So
feeding `NEJ` as the eval set for an `F2T`-trained model would produce a
**totally leaked** accuracy (every eval image was a training image) with
**no warning from the pipeline**. The only catalog-side record of the
`NEJ→F2T` derivation is free-text description prose; it is not a walkable
lineage edge, so neither `get_lineage` nor `list_dataset_relations` can
answer "what was derived from the training partition?"

The canonical splits register correctly (`F2J`→F2T/F34,
`F3M`→F3W/F46 both show parent→child), so the registration mechanism
works — the labeled-split creation path simply does not use it.

## Why this is a finding even though the run was clean

The team navigated the trap perfectly: the Curator caught it, the Modeler
cited it (tk-004) and chose the leakage-free F2J pair, and the Analyst's
held-out numbers are honest (I confirmed F2T∩F34=0). So this did **not**
compromise any deliverable in *this* run. But the hazard is latent in the
catalog for the *next* user, and the platform offers no guard:

1. **Loader gap (Bug):** `split_dataset` / the bootstrap loader does not
   register labeled-split roots as children of their source partition.
2. **Type-tag overstatement (judgment call):** `Dataset_Type=Testing` on
   training-pool-derived splits invites the leak. The `Validation` term
   exists in the vocab and would describe these correctly — but retagging
   is a catalog mutation that flips the dataset version, and one can defend
   "testing = the held-out half of *this* split." Left to the user, per
   the Curator.

## Reproduction

Read-only, against localhost catalog 168 (deriva-ml Python or MCP
`query_attribute` on `Dataset_Image`):

```python
from deriva_ml import DerivaML
ml = DerivaML(hostname="localhost", catalog_id="168")
di = ml.pathBuilder().schemas["e2e-test-20260530"].tables["Dataset_Image"]
rows = list(di.entities().fetch())
from collections import defaultdict
m = defaultdict(set)
for r in rows: m[r["Dataset"]].add(r["Image"])
assert m["NEJ"] <= m["F2T"] and m["PJ4"] <= m["F2T"]      # total leakage
assert not (m["F2T"] & m["F34"])                           # F34 is safe
```

```
deriva_ml_list_dataset_relations(hostname="localhost", catalog_id="168",
    dataset_rid="F2T", direction="both", recurse=true)
# -> {"parents": [F2J], "children": []}   (NE0/PHJ invisible)
```

## Suggested disposition

GitHub issue on the bootstrap loader: register labeled-split roots as
catalog children of F2T so the derivation is a walkable lineage edge.
Optionally, revisit `Dataset_Type` for training-pool-derived splits
(`Testing` → `Validation`). The catalog data itself is internally sound
(perfectly balanced, no within-pair leakage); the gap is in how the
derivation is *recorded* and how the type tag would mislead a future
consumer.
