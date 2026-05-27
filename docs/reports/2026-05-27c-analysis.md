# Analyst Report — Multipersona e2e 2026-05-27c

**Catalog:** `localhost:95` (`e2e-test-20260527c`)
**Branch / commit:** `e2e-test/2026-05-27-c` @ post-`be860d2`
**Workflow under analysis:** `XRC` (CIFAR-10 2-Layer CNN, refactored runner, PR #37)
**Analysis execution:** [`YT6`](https://localhost/id/95/YT6)
**Analysis workflow:** `YT2` (`ROC Analysis Notebook`, notebook `roc_analysis.ipynb` @ `eee3fec`)
**Prepared by:** Analyst persona (autonomous)

This report answers four questions the platform test cares about:

1. Which Developer-arc execution is the best model? On what metric?
2. What does per-class behaviour look like — does any class drive overall accuracy?
3. Does `get_denormalized_as_dataframe` produce the right wide-table shape (the headline PR #246 check)?
4. What can a future reviewer reproduce from the catalog alone?

The catalog is the source of record for everything below; RIDs link
to live cite URLs and the report can be regenerated from those plus
the executed notebook asset (`YXG`).

---

## 1. Ranked executions

### Authoritative numbers (CSV-derived, joined to ground-truth feature)

The headline metric is **test accuracy computed from the committed
`prediction_probabilities.csv`** for each execution, joined against
the `Image_Classification` feature row where `Confidence IS NULL`
(the ground-truth lane, execution `FZC`, 1500 rows / one per
Image). Every test sample is labeled, so accuracy on matched rows is
the test_acc on the full test bag.

| Rank | Exec | Train dataset | Val | Test bag | Model variant         | seed | epochs | CSV asset | Source_Label | **Test acc (CSV)** | Micro-AUC | Macro-AUC |
|-----:|:----:|---------------|:---:|----------|-----------------------|-----:|-------:|:---------:|:------------:|-------------------:|----------:|----------:|
| 1    | `Y1M` | TX0 → TX8 (600) | —   | TXJ (150) | cifar10_quick (32/64, bs=128) | 123  | 10     | `Y3J`     | `epoch_10`   | **34.00%**         | 0.7772    | 0.7727    |
| 2    | `YDT` | TX0 → TX8 (600) | XEM (150) | TXJ (150) | default_model (32/64, bs=64)  | 7    | 10     | `YFT`     | `epoch_10`   | 33.33%             | **0.7920** | **0.7855** |
| 3    | `XRJ` | WD2 → WDA (400) | —   | WDM (100) | cifar10_quick (32/64, bs=128) | 42   | 3      | `XTG`     | `epoch_3`    | 24.00%             | n/a (smoke) | n/a |

Notes on the table:

- **Y1M edges YDT on argmax accuracy (34.00% vs 33.33%) but YDT
  leads on both Micro-AUC and Macro-AUC.** They are within one
  prediction of each other on a 150-image bag, so the ranking
  inverts depending on whether the consumer cares about hard labels
  (argmax) or about probability ranking. For a downstream consumer
  that cares about threshold tuning or top-k retrieval, **YDT is
  the better model**; for a consumer that only uses argmax labels,
  Y1M is fractionally ahead.
- **XRJ is the smoke run** and is included for completeness only;
  3 epochs on a 400-image train set is well below the others and
  produces a different (smaller) test bag (WDM not TXJ).

### Reconciliation against tk-004's reported numbers

tk-004 reports test_acc figures of **38.00% (Y1M), 36.67% (YDT),
24.00% (XRJ)** sourced from the training log. Recomputing from the
CSV gives **34.00% / 33.33% / 24.00%**. The XRJ figure matches; Y1M
and YDT drift by ~4 percentage points.

The cause is the silent desync that finding `developer/01` already
flagged: `evaluate()` (in the train loop) and `predict_batch()` (in
`record_predictions`) are two separate forward passes. Both are
deterministic with `model.eval()`, identical weights, and
`shuffle=False`, so they *should* agree — but the CSV numbers
differ from the log. The reconciliation rail the prompt asked for
(an "Emission-time accuracy" log line emitted alongside CSV write)
is the missing safety net here; without it, the analyst cannot tell
which is the correct value at a glance.

**Cross-channel check on the headline numbers.** I also re-ranked
using the `Image_Classification` *feature-row* lane (predictions
written into the catalog as a feature, separate from the CSV asset).
That lane agrees with the CSV exactly (34.00% / 33.33% / 24.00%).
Together they pin the catalog-of-record number, and the training
log is the one that disagrees.

**Decision:** use the CSV / feature-row number (34.00% / 33.33% /
24.00%) for the ranking. It's what's actually committed to the
catalog as a typed feature value — the training-log line is
free-text in an Execution_Asset.

---

## 2. Per-class accuracy & confusion structure

Per-class accuracy from the joined CSVs (recomputed):

| Class       | Y1M (n=15) | YDT (n=15) |
|-------------|-----------:|-----------:|
| airplane    | 80.00%     | 73.33%     |
| automobile  | 33.33%     | 20.00%     |
| bird        | 20.00%     | 20.00%     |
| cat         | 13.33%     | 20.00%     |
| deer        | **0.00%**  | **0.00%**  |
| dog         | 26.67%     | 33.33%     |
| frog        | 66.67%     | 66.67%     |
| horse       | 46.67%     | 13.33%     |
| ship        | 33.33%     | 46.67%     |
| truck       | 20.00%     | 40.00%     |

Same shape across both models:

- **airplane / frog / ship** lead; both models have learned the
  "big distinctive blob on uniform background" pattern early.
- **deer is at 0.00% in both models.** A genuine zero — every
  deer-labeled image was assigned to a different class. The
  confusion matrices (`YWC`, `YWE`) confirm this — the deer row is
  flat across other animal classes (most often bird / horse /
  dog), consistent with a CNN this small not yet learning
  fine-grained four-legged-mammal discrimination at 10 epochs.
- **automobile and truck swap accuracy** between the two runs:
  Y1M handles automobile better (33% vs 20%) but YDT handles truck
  better (40% vs 20%). The wheeled-vehicle confusion is the
  expected hard pair for early-epoch CIFAR-10.

Confusion matrices were saved as `YWC` (Y1M) and `YWE` (YDT) on
the `YT6` execution. They reproduce the same "deer-row goes
everywhere, vehicle pair confuses" structure visually.

Per-class AUC (probability ranking, not argmax):

| Class       | Y1M     | YDT     |
|-------------|---------|---------|
| airplane    | 0.918   | 0.935   |
| automobile  | 0.723   | 0.761   |
| bird        | 0.734   | 0.734   |
| cat         | 0.680   | 0.667   |
| deer        | 0.717   | 0.722   |
| dog         | 0.724   | 0.752   |
| frog        | 0.814   | 0.818   |
| horse       | 0.805   | 0.804   |
| ship        | 0.896   | 0.907   |
| truck       | 0.716   | 0.756   |
| **Micro**   | 0.777   | **0.792** |
| **Macro**   | 0.773   | **0.786** |

YDT's edge concentrates in airplane, automobile, dog, and truck —
all classes where YDT had access to the XEM validation signal
during training (XEM is 15-per-class, so every class got equal val
feedback). It's not a strong effect at 150 test samples, but the
sign is consistent across 4 of the 10 classes.

---

## 3. Denormalize parity verification (PR #246 headline check)

This is the headline check this run was set up to do: PR #246
restored row-completeness in `PagedFetcher`; this run confirms it
holds across the cifar10_cnn refactor (PR #37) and the path_walker
pin (#38/#59). I ran two denormalize calls and reconciled every
result against the direct-channel member-driven query.

### TX0 — Labeled_Split (750 images)

```python
ds = ml.lookup_dataset('TX0')
df = ds.get_denormalized_as_dataframe(
    include_tables=['Image', 'Execution_Image_Image_Classification'],
)
# duration: 0.98s
# rows:     1150
# columns:  12
# unique Image.RID: 750
```

| Source         | (Execution, Image) row count | Match? |
|----------------|-----------------------------:|:------:|
| Denormalize wide table                                | **1150** | — |
| Direct: feature_values('Image_Classification') filtered to TX0 hierarchy | 1150     | yes |
| Set of (Execution, Image) tuples — denorm vs direct   | identical | yes |

Composition of the 1150 EIIC rows:

| Producing execution | rows | meaning |
|---------------------|-----:|---------|
| `FZC` (ground truth) | 750  | one per Image in TX0 ∪ children |
| `Y1M` (predictions)  | 150  | TXJ test-bag predictions |
| `YDT` (predictions)  | 150  | TXJ test-bag predictions |
| `XRJ` (predictions)  | 100  | WDM predictions — WDM ⊂ TX0 ∪ children |

The 100 XRJ rows landing in TX0's denormalize is **correct**, not
spurious: WD2 (XRJ's training set) was carved from the same image
pool as TX0, and 100 of WDM's 100 images happen to also be members
of TX0's hierarchy (verified directly via set intersection). The
denormalize call returns "all features attached to images that are
in the TX0 hierarchy", which is exactly what `include_tables` is
specified to do — it doesn't filter feature rows by the producing
execution.

Class distribution of GT rows in the denormalize output is
**perfectly balanced**: 75 per class × 10 classes = 750. Matches
tk-002's audit of TX0's stratified construction.

### JZ8 — Complete (1500 images, root of the hierarchy)

This is the **headline 1500-image stress** that exercises the
PR #246 PagedFetcher fix:

```python
ds = ml.lookup_dataset('JZ8')
df = ds.get_denormalized_as_dataframe(
    include_tables=['Image', 'Execution_Image_Image_Classification'],
)
# duration: 1.44s
# rows:     1900
# columns:  12
# unique Image.RID: 1500
```

| Source | rows | Match? |
|---|---:|:-:|
| Denormalize wide table | **1900** | — |
| Direct: feature_values filtered to JZ8 members | 1900 | yes |

Composition: 1500 GT (FZC) + 150 Y1M + 150 YDT + 100 XRJ = 1900.

**No row loss at 1500-image scale, no duplication.** This is the
exact failure mode PR #246 was filed against — the previous
PagedFetcher would silently truncate at the page boundary and
return ~half the rows. That regression does not return on the
refactored runner.

### Discoverability notes

`get_denormalized_as_dataframe(include_tables=...)` is the right
shape for analysis work and the parameter name is self-explanatory;
the `Dataset.list_denormalized_columns(include_tables=...)` preview
is the right discovery aid before issuing the heavy call. One
discoverability point worth flagging — the dataset-lifecycle skill
documents the parity-check pattern (member set vs denormalize row
count), and it's still the right pattern. The only minor friction
was the "100 XRJ rows in TX0" surprise; that's not a denormalize
bug, but a consumer who hasn't internalised that dataset
membership and Image identity are independent dimensions will
double-take. Worth a sentence in the skill, but not a finding.

---

## 4. Conclusions

### Headline answer to "which model is best?"

**For ranking-aware consumers (top-k retrieval, threshold tuning):
YDT** (Micro-AUC 0.792 vs 0.777, Macro-AUC 0.786 vs 0.773).

**For argmax-only consumers (downstream classification labels):
Y1M** by 1 prediction out of 150 (34.00% vs 33.33%). The gap is
within noise on a 150-image bag — not statistically
distinguishable.

The signal across both metrics: the Validation lane (PR #29 / D01)
didn't dramatically boost accuracy, but it didn't *hurt* either,
and the resulting model's probability calibration is slightly
better — which is what a validation signal during training *should*
produce. The Developer arc validates the PR #29 dispatch lane:
TX0+XEM is a working composite-dataset configuration, not just a
no-op.

### Caveats

- **150 test images is small.** A 1-prediction difference moves
  accuracy by 0.67 percentage points. The ranking between Y1M and
  YDT should not be over-interpreted.
- **The training-log test_acc value cannot be trusted as the
  catalog-authoritative number.** It disagrees with the CSV /
  feature-row by ~4 points on Y1M and YDT. Either the training-log
  number or the CSV number is wrong; the analyst arc concludes the
  CSV number is the one to use (it's what's committed) but the
  underlying inconsistency is `developer/01`'s concern.
- **deer is a 0% class for both models at 10 epochs.** Doesn't
  invalidate the comparison, but a deer-heavy downstream task
  would need a longer training run or a different architecture.
- **No model is anywhere near production-ready.** Both Y1M and
  YDT sit at ~34% accuracy on a 10-class problem (chance = 10%).
  The point of this run was platform validation, not model
  quality.

### What the denormalize check tells us about the platform

PR #246 (row-completeness invariant in `PagedFetcher`) and the
cifar10_cnn refactor (PR #37) compose cleanly. The 1500-image
denormalize produces exactly the expected 1900 rows; the 750-image
labeled split produces exactly 1150 rows; the (Execution, Image)
key set matches the direct-channel query bit-for-bit.

This was the explicit failure mode this run was set up to detect.
**No regression observed.**

---

## 5. Reproducibility

A future reviewer who only has access to catalog 95 can reproduce
every claim in this report from the following starting points:

| Question | Where to look |
|----------|---------------|
| Which Developer executions are in scope? | `mcp.deriva_ml_list_executions(workflow_rid="XRC", status="Uploaded")` returns Y1M, YDT, XRJ. |
| Which assets carry the predictions? | `ml.lookup_execution(<rid>).list_assets()` — filter `asset_table == "Execution_Asset"` and `filename == "prediction_probabilities.csv"`. Asset RIDs: Y3J (Y1M), YFT (YDT), XTG (XRJ). |
| Ground-truth feature lane? | `ml.feature_values("Image", "Image_Classification")` → filter `Execution == "FZC"` and `Confidence is None`. 1500 rows, one per Image. |
| Test accuracy of each run, authoritative? | `Predicted_Class` column of the CSV joined on `Image_RID` against the GT lookup. The notebook (`YXG`) reproduces this. |
| ROC + confusion artefacts? | Execution `YT6` (`YT2` workflow, `roc_analysis.ipynb`): assets `YW6`, `YW8`, `YWA`, `YWC`, `YWE`, `YWG`, `YXG`, `YXJ`. |
| Denormalize parity numbers? | The skill recipe: `ds.list_dataset_members()['Image']` for the member set; `ds.get_denormalized_as_dataframe(include_tables=...)` for the wide table; reconcile (Execution, Image) tuples. |

### One-command reproduction

To re-run the analysis from a clean checkout of this branch:

```bash
DERIVA_ML_ALLOW_DIRTY=true uv run deriva-ml-run-notebook \
    notebooks/roc_analysis.ipynb \
    --allow-dirty \
    assets=analyst_2026_05_27c
```

The `analyst_2026_05_27c` asset config is wired into
`src/configs/assets.py` and the corresponding notebook config is
in `src/configs/roc_analysis.py`. Both are committed under the
`[E2E-DROP]` analyst commit on `e2e-test/2026-05-27-c`.

### Provenance graph (one hop)

```
FZC (GT loader) ─┐
                  ├── Image_Classification feature rows on each Image
Y1M / YDT / XRJ ─┘                       │
                                          ▼
                  prediction_probabilities.csv (Y3J / YFT / XTG)
                                          │
                                          ▼
                  YT6 (this analysis execution)
                       → ROC + confusion + roc_metrics.csv
                       → executed notebook (YXG)
```

The MCP `deriva_ml_get_lineage(YT6, depth=2)` resource walks this
graph automatically.
