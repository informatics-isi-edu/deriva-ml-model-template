# Learning-Rate Sweep on the Full Dataset — Design

**Date:** 2026-06-05
**Status:** DRAFT — in brainstorming; Phase 1–2 detailed, Phases 3–6 outlined, not yet approved
**Author:** Carl Kesselman (with Claude)
**Catalog (target for runs):** localhost / id 69 / e2e-test-20260605
**Mode:** Planning only — no implementation until the spec is approved

---

## 1. Summary

Add a new multirun, `lr_sweep_full`, that measures the impact of learning rate on the `cifar10_extended` model's generalization, using the full (KDT canonical) dataset. It is a parallel companion to the existing small-dataset `lr_sweep` — the existing sweep is left untouched so small-vs-full behavior can be compared later.

## 2. Decisions (settled)

| Axis | Decision | Rationale |
|---|---|---|
| Relationship to existing `lr_sweep` | New parallel sweep `lr_sweep_full`; leave `lr_sweep` intact | Isolates the dataset variable; preserves small-vs-full comparison |
| LR grid (independent variable) | `1e-4, 1e-3, 1e-2, 1e-1, 3e-1` (5 points) | Maps the full learning regime incl. the divergence boundary at the high end |
| Base architecture | `cifar10_extended` (64→128 ch, 256 hidden, dropout 0.25, wd 1e-4) | A representative "real" model rather than the tiny quick arch |
| Epochs | 50, held constant across all 5 runs | Full extended schedule; late-epoch behavior visible |
| Dataset | `cifar10_split` (KDT: 550 train / 550 test) | 5× larger test set → low-noise accuracy-vs-LR curve; leak-free + class-balanced; more train data for the bigger model |
| Seed | 42, fixed across all 5 runs | LR is the only variable; differences attributable to LR, not RNG |
| Success measure | Best test accuracy per LR + training-curve shape (converged/diverged/unstable/underfit) | Directly answers "impact of LR"; reuses the Analyst's scoring approach |

## 3. Phase 1 — Hypothesis

> **What is the impact of learning rate on the `cifar10_extended` model's generalization, on the full (KDT) dataset?**

- **Independent variable:** learning rate — `1e-4, 1e-3, 1e-2, 1e-1, 3e-1`.
- **Held constant:** `cifar10_extended` architecture, 50 epochs, `cifar10_split` (KDT), seed 42, extended-config batch size.
- **Evidence:** best held-out test accuracy per LR on the 550-image KDT test partition, plus per-LR training-curve classification (converged / diverged / unstable / underfit).
- **Success criterion:** succeeds (regardless of winner) if it produces a clean accuracy-vs-LR curve across all 5 points and a stability verdict for each. Expected shape: divergence at 1e-1/3e-1, sweet spot near 1e-3, slow-but-stable underfit at 1e-4 under a fixed 50-epoch budget.
- **Caveat (recorded up front):** 550 test images ≈ 0.18 pp per prediction — good for curve shape and gross ranking, not for a hairline winner. Single seed → no run-to-run variance estimate.
- **Cost budget:** 5 runs × 50 epochs of the extended model on 550 training images (minutes/run at this catalog's scale). Still gated: dry-run → one smoke run → full sweep before committing all 5.

## 4. Phase 2 — Configuration

One new multirun in `src/configs/multiruns.py` + its description in `src/configs/multirun_descriptions.py`:

```python
multirun_config(
    "lr_sweep_full",
    overrides=[
        "+experiment=cifar10_extended_full",   # extended arch + full labeled split...
        "datasets=cifar10_split",              # ...retargeted to the KDT canonical split
        "model_config.epochs=50",
        "model_config.learning_rate=0.0001,0.001,0.01,0.1,0.3",
    ],
    description=LR_SWEEP_FULL_DESCRIPTION,
)
```

Notes:

1. Builds on the existing `cifar10_extended_full` experiment and overrides `datasets=cifar10_split` — reuses an experiment rather than inventing one.
2. No new `model_config` / `experiment` presets needed; LR values swept as inline Hydra overrides (same pattern as the existing `lr_sweep`). Non-`[E2E-DROP]` template addition; only the dataset RIDs it resolves through (in `datasets.py`) are catalog-specific.

Inter-phase gate (no catalog writes): `--list-configs` shows `lr_sweep_full`; `+multirun=lr_sweep_full --cfg job` renders the resolved config.

## 5. Phase 3 — Identify assets *(OUTLINE — not yet expanded)*

- Pin `cifar10_split` (KDT) RID + version in `configs/datasets.py` (verify it's the version the runs should consume).
- No pretrained weights needed (training from scratch).
- Verify `Image_Class` vocab terms exist (they do — 10 terms).
- Gate: `dry_run=true` validates all references resolve.

## 6. Phase 4 — Run model *(OUTLINE — not yet expanded)*

- Progression: dry-run → one smoke run at a single safe LR (e.g. 1e-3) → full 5-point sweep.
- Runs as one parent multirun execution with 5 children.

## 7. Phase 5 — Update assets *(OUTLINE — not yet expanded)*

- Each run writes weights + training log + prediction CSV + feature rows. Note the 5 execution RIDs for the evaluation step.

## 8. Phase 6 — Evaluate *(OUTLINE — not yet expanded)*

- Accuracy-vs-LR curve + per-LR stability verdict from training logs.
- Reuse the Analyst's join (predictions vs ground truth on KDT test).
- Record the LR-impact reading in `tacit-knowledge.md`.

## 9. Open items / not yet done

- Phases 3–6 need full expansion.
- Phase-1 framing pending confirmation.
- Targets catalog 69 (the e2e catalog); a different catalog changes the dataset RIDs in §5.
