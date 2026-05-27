# Analyst report — CIFAR-10 Family-A triplet (e2e 2026-05-27-e)

**Catalog:** [`localhost` / 2 (`e2e-test-20260527e`)](https://localhost/chaise/recordset/#2)
**Analyst persona arc:** 2026-05-27, autonomous mode
**Analysis execution:** [11AY](https://localhost/id/2/11AY) — `ROC curve analysis [overrides: assets=modeler_familyA_triplet]`
**Joined wide table:** [`findings/analyst/wide_joined_K16.csv`](../../findings/analyst/wide_joined_K16.csv) (500 rows × 35 cols)

This report is one of three deliverables. The companions are:

- The **joined wide table** above — one row per K16 image; columns for
  ground truth, each model's top-1 prediction, and each model's
  per-class probability vector. Any number in this report can be
  re-derived from it without re-querying the catalog.
- The **catalog-resident artifacts** committed by execution `11AY`:
  three ROC-curve plots, three confusion matrices, the cross-model
  comparison plot, and `roc_metrics.csv`. All cite their source
  prediction CSV by Asset_RID, so a follow-up analyst can re-enter
  the lineage from any of them.

---

## 1. What was on the table

The Modeler ([tk-004](../../tacit-knowledge.md#tk-004)) handed me three training
executions trained on the **same** data — Family-A dataset
[K0M](https://localhost/id/2/K0M@355-KW8K-DXSC) (K0W train / K16 test,
500/500, seed=42) — varying only model capacity and training budget:

| Run | Config | Epochs | Channels | Predictions CSV |
|-----|--------|--------|----------|-----------------|
| [XZP](https://localhost/id/2/XZP) | `cifar10_quick` | 3 | 32→64 | [Y1M](https://localhost/id/2/Y1M) |
| [Z1R](https://localhost/id/2/Z1R) | `default_model` | 10 | 32→64 | [Z3P](https://localhost/id/2/Z3P) |
| [103T](https://localhost/id/2/103T) | `cifar10_large` | 20 | 64→128 | [105R](https://localhost/id/2/105R) |

The K16 test partition is genuinely held out — it lives on the
Toronto test_batch side of the canonical split (see
[tk-002](../../tacit-knowledge.md#tk-002)). That makes it the right
substrate for ROC; using a Family-B partition (WDA/WDM, TX8/TXJ) for
this kind of comparison would have been wrong because those "test"
partitions are drawn from the *training* image pool, not the held-out
side.

Ground truth came from execution [FZC](https://localhost/id/2/FZC)
(the loader), filtered by `Execution == FZC AND Confidence IS NULL`
per [tk-003](../../tacit-knowledge.md#tk-003). At analysis time
`Image_Classification` carried 3200 rows total — 1500 GT plus 1700
prediction rows from five training executions; without the
execution-RID filter the "ground truth" would have silently been
contaminated with predictions.

## 2. Ranking the three runs

Cross-channel verification: the numbers below were independently
derived two ways — once from a standalone Python script
[`findings/analyst/rank_and_join.py`](../../findings/analyst/rank_and_join.py)
that joins predictions and GT in pandas, once from the
catalog-resident `roc_metrics.csv` produced by execution
[`11AY`](https://localhost/id/2/11AY). The two agree to all digits.

| Rank | Execution | Config | Top-1 Acc | Micro-AUC | Macro-AUC |
|------|-----------|--------|-----------|-----------|-----------|
| 1 | [103T](https://localhost/id/2/103T) | `cifar10_large` (20 ep, 64→128) | **36.8%** | **0.817** | **0.817** |
| 2 | [Z1R](https://localhost/id/2/Z1R) | `default_model` (10 ep, 32→64) | 36.0% | 0.795 | 0.801 |
| 3 | [XZP](https://localhost/id/2/XZP) | `cifar10_quick` (3 ep, 32→64) | 25.2% | 0.722 | 0.732 |

The Micro-AUC ranking is the cleaner picture than Top-1: it
discriminates more strongly between Z1R and 103T (0.795 vs 0.817)
than accuracy does (36.0% vs 36.8%), because it integrates over
ranking quality rather than just argmax correctness. The two larger
models are clearly above the 3-epoch baseline; the comparison
between them is where the interesting story is.

## 3. Per-class behavior — where capacity helps, where it doesn't

This is the part the Modeler couldn't see at training time. The
per-class AUC pivot tells a clean story:

| Class | quick | default | large | range | best |
|-------|-------|---------|-------|-------|------|
| airplane | 0.821 | 0.847 | 0.816 | 0.032 | default |
| automobile | 0.731 | 0.874 | 0.888 | **0.157** | large |
| bird | 0.606 | 0.724 | 0.742 | 0.137 | large |
| cat | 0.680 | 0.731 | 0.735 | 0.055 | large |
| deer | 0.702 | 0.735 | 0.793 | 0.091 | large |
| dog | 0.709 | 0.783 | 0.802 | 0.093 | large |
| frog | 0.777 | 0.827 | 0.844 | 0.067 | large |
| horse | 0.704 | 0.822 | 0.823 | 0.119 | large |
| ship | 0.786 | 0.832 | 0.858 | 0.072 | large |
| truck | 0.807 | 0.837 | 0.872 | 0.066 | large |

Two observations:

- **`airplane` is the one class where extra capacity hurts.** The
  large model is slightly *worse* than the default on airplane AUC
  (0.816 vs 0.847). I read this as a capacity-vs-data signal: with
  only 50 airplane training images, the larger model has more room to
  latch onto background features (sky vs water vs land) that don't
  generalize from K0W to K16. Worth a follow-up Modeler run with
  augmentation if airplane recall matters.
- **Capacity helps most where the texture is busy.** `automobile`,
  `bird`, and `horse` show the biggest AUC jumps from
  quick→large (0.157, 0.137, 0.119). These are exactly the classes
  whose silhouettes are richer or whose canonical poses are less
  consistent — they reward the larger feature pyramid. The smooth
  shapes (`airplane`, `ship`, `truck`) show smaller jumps because the
  shallow features are already most of the signal.

## 4. Confusions — what the models are actually confused about

Per-class top-1 recall paints a less polished picture than AUC:

| Class | XZP | Z1R | 103T |
|-------|-----|-----|------|
| airplane | 0.62 | 0.32 | 0.32 |
| automobile | 0.10 | 0.34 | 0.56 |
| bird | 0.00 | 0.22 | 0.30 |
| cat | 0.00 | 0.48 | 0.26 |
| deer | 0.14 | 0.42 | 0.22 |
| dog | 0.24 | 0.12 | 0.32 |
| frog | 0.68 | 0.42 | 0.42 |
| horse | 0.08 | 0.46 | 0.34 |
| ship | 0.06 | 0.58 | 0.46 |
| truck | 0.60 | 0.24 | 0.48 |

The **3-epoch model is degenerate**: it learns only `airplane`, `frog`,
and `truck` reliably, and gets `bird` and `cat` to zero recall — it's
collapsing those classes into a confidence-low rejection bucket. This
is consistent with its 25% accuracy: at three epochs, only the
loudest visual classes have crossed the decision boundary.

The two larger models trade per-class strength. Z1R does better on
`cat` and `ship`; 103T does better on `automobile`, `deer`, `dog`,
and `truck`. Neither dominates per-class, which is consistent with
their close micro-AUC numbers.

**Top systematic confusions (truth → prediction):**

- **`ship` → `airplane`** (XZP only: 25 cases). Smooth horizontal
  silhouettes against blue background — exactly what a 3-epoch model
  would lump together by background statistics.
- **`bird` and `deer` → `frog`** (XZP: 24 cases each). Small
  green-ish thing in foliage; the quick model has not yet learned
  body plan.
- **`dog` → `cat`** (Z1R: 19, 103T: 13). The canonical CIFAR
  confusion; survives both models.
- **`truck` ↔ `automobile`** (103T: 12 each direction). Wheeled
  vehicles with similar silhouettes; capacity isn't fixing it
  because the canonical CIFAR images of these classes are visually
  near-identical at 32×32.

**Cross-model agreement:** of 500 K16 images, only 96 (19.2%) get the
same prediction from all three models, and only 46 (9.2%) get the
same *correct* prediction. **207 (41.4%) are missed by all three
models.** Those 207 are the genuinely hard images — the ones a
larger Family-A capacity bump alone won't recover. Top systematic
errors among them: `automobile → truck` (5), `deer → frog` (5),
`dog → frog` (4), `horse → dog` (4).

This is a useful signal: roughly 40% of the K16 difficulty is
shared across the triplet, which means the next modelling lever is
not "more capacity in the same architecture" but **augmentation,
representation, or simply more training images**.

## 5. Overfitting signature — the 103T story

The Modeler called out ([tk-004](../../tacit-knowledge.md#tk-004))
that 103T's training accuracy climbs from 16% to 100% over 20 epochs
while test accuracy plateaus around 37% and test loss starts rising
after epoch 9. Pulled into this analysis, that overfitting *doesn't*
hurt the AUC ranking — 103T still wins, because:

- AUC measures ranking quality, not argmax. A model that's
  *overconfident on memorised training-set features* still keeps the
  probability ordering right on held-out images, as long as the
  features it's overfitting to correlate with the true class on
  K16. They mostly do (K0W and K16 are both sampled from the
  Toronto canonical distribution).
- The 103T overfitting story is about *confidence calibration*, not
  about discriminative power. If a downstream consumer needs
  calibrated probabilities (selective prediction, threshold-based
  alerts), 103T is the *worst* of the three — the per-class probs
  are likely to be miscalibrated upward. For pure ranking work
  (top-K retrieval, score-based ranking), 103T is still the right
  choice.

The catalog's emission-time accuracy log records training-time
test-set accuracy at epoch boundaries — peak was ~39.8% at 103T
epoch 9, which suggests the right operating point would have been
"train to epoch 9 and stop." Early stopping is a clean follow-up
Modeler arc with `early_stop_patience=` if it were a knob; today
it'd be a `model_config.epochs=9` override.

## 6. Recommendation for the next round

If the team is doing one more Modeler arc with this catalog:

1. **Add augmentation, not capacity.** Random crop / horizontal
   flip / color jitter would directly address the 41.4%
   all-three-wrong floor. Capacity has hit its plateau on K0M.
2. **Run a JZJ-scale comparison.** K16 at 500 images is enough to
   *rank* models but tight for absolute numbers. Re-running the
   triplet on the canonical [JZJ](https://localhost/id/2/JZJ@355-KW8K-DXSC)
   split (750/750) would either confirm or refute the
   capacity-helps-on-textured-classes story with more statistical
   power.
3. **Stop 103T at epoch 9.** The textbook overfitting curve says
   the model has more peak performance than the final epoch shows.

If the team is moving on from CIFAR-10: the three-run triplet is
a sufficient platform-fitness demonstration. The ranking is
defensible, the per-class breakdown tells a coherent story, the
catalog supports the analysis end-to-end, and the joined wide
table is the durable artifact that survives a re-analysis.

## 7. Catalog touchpoints (provenance summary)

- **Inputs** (Modeler-produced prediction CSVs): [Y1M](https://localhost/id/2/Y1M),
  [Z3P](https://localhost/id/2/Z3P), [105R](https://localhost/id/2/105R).
- **Ground truth feature rows**: 1500 rows in `Image_Classification`,
  filtered by `Execution=FZC AND Confidence IS NULL` (loader execution
  [FZC](https://localhost/id/2/FZC)).
- **Analysis execution**: [11AY](https://localhost/id/2/11AY)
  (workflow: ROC Curve Analysis).
- **Output assets** (catalog-resident): per-model ROC plots
  [1184](https://localhost/id/2/1184) /
  [1186](https://localhost/id/2/1186) /
  [1188](https://localhost/id/2/1188); per-model confusion matrices
  [118C](https://localhost/id/2/118C) /
  [118E](https://localhost/id/2/118E) /
  [118G](https://localhost/id/2/118G); cross-model comparison plot
  [11D8](https://localhost/id/2/11D8); aggregated metrics CSV
  [118J](https://localhost/id/2/118J); committed notebook
  source/markdown [11ER](https://localhost/id/2/11ER) /
  [11ET](https://localhost/id/2/11ET).
- **Worktree-resident derivations**:
  [`findings/analyst/wide_joined_K16.csv`](../../findings/analyst/wide_joined_K16.csv),
  [`findings/analyst/ranking.csv`](../../findings/analyst/ranking.csv),
  [`findings/analyst/per_class_recall.csv`](../../findings/analyst/per_class_recall.csv),
  [`findings/analyst/roc_metrics_from_catalog_11AY.csv`](../../findings/analyst/roc_metrics_from_catalog_11AY.csv)
  (downloaded copy of asset 118J), the standalone derivation script
  [`findings/analyst/rank_and_join.py`](../../findings/analyst/rank_and_join.py).

## 8. Friction encountered

One finding filed during this arc:

- [`findings/analyst/01-run-notebook-config-derivation-fails-under-papermill.md`](../../findings/analyst/01-run-notebook-config-derivation-fails-under-papermill.md) —
  `run_notebook()` auto-derivation of the Hydra config name from the
  notebook filename works interactively but fails when launched via
  `deriva-ml-run-notebook` (it calls papermill programmatically and
  never sets `PAPERMILL_INPUT_PATH` in `os.environ`). Workaround
  applied to `notebooks/roc_analysis.ipynb`: pass `"roc_analysis"`
  explicitly to `run_notebook()`. The auto-derivation docstring's
  claim about `PAPERMILL_INPUT_PATH` being "the most reliable signal"
  is wrong for the only headless runner the project ships.

No other friction was load-bearing for the analysis. The denormalize
work the Curator did in earlier runs ([tk-001](../../tacit-knowledge.md#tk-001))
was not needed here — the wide joined table for ROC didn't require
materializing a denormalized dataset bag; reading from
`feature_values` + the prediction CSVs was enough. A future analysis
that wanted per-image filename / Hatrac path / etc. would lean on
the denormalize path; this one didn't.
