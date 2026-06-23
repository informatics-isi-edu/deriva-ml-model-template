# Task A3 Report — Remove CIFAR Implementation Files

**Status: DONE_WITH_CONCERNS**

**Summary:** 25 files staged for deletion (all expected files removed successfully);
`model_config` group is now completely unregistered — a generic `default_model` is
required in A4.

---

## git status --short (full)

```
D  CIFAR10.md
D  Experiments.md
D  notebooks/roc_analysis.ipynb
D  scripts/assets.toml
D  scripts/test_bag_fk_traversal.py
D  src/configs/cifar10_cnn.py
D  src/models/cifar10_classes.py
D  src/models/cifar10_cnn.py
D  src/scripts/_cifar10_assets.py
D  src/scripts/_cifar10_datasets.py
D  src/scripts/_cifar10_schema.py
D  src/scripts/_cifar10_source.py
D  src/scripts/analyst_join.py
D  src/scripts/load_cifar10.py
D  tests/test_analyst_join.py
D  tests/test_cifar10_assets.py
D  tests/test_cifar10_cnn_loaders.py
D  tests/test_cifar10_datasets.py
D  tests/test_cifar10_schema.py
D  tests/test_cifar10_source.py
D  tests/test_configs_load.py
D  tests/test_load_cifar10_retry.py
D  tests/test_load_cifar10_split_no_leakage.py
D  tests/test_runner_bag_dispatch.py
D  tests/test_runner_seed.py
```

Total: **25 files staged for deletion.** No git rm failures — every listed
file existed in the index and was removed cleanly.

Note: `tests/test_notebook_examples.py` (not in the deletion list) remains.

---

## model_config group / default_model status

**`src/configs/cifar10_cnn.py` was the SOLE file registering the `model_config`
Hydra-zen store group.** It registered:

- `default_model` (REQUIRED default — referenced by `BaseConfig` defaults list)
- `cifar10_quick`
- `cifar10_extended`
- `cifar10_large`
- `cifar10_test_only`
- and others

After this deletion, **the `model_config` group has zero registered entries.**
`src/configs/base.py` line 32 still contains `{"model_config": "default_model"}`
in the defaults list, which will cause a Hydra config resolution failure at
runtime until a new generic `default_model` is registered.

**A4 must create a generic scaffold that registers at minimum `default_model`
in the `model_config` group.**

---

## Remaining CIFAR references in src/ / tests/ / scripts/

The deleted files are gone. CIFAR references remain only in the **config modules
that A4 will convert to scaffolds**:

- `src/configs/datasets.py` — CIFAR dataset definitions (to be scaffolded)
- `src/configs/experiments.py` — CIFAR experiment combinations including
  `{"override /model_config": "cifar10_quick"}` etc. (to be scaffolded)
- `src/configs/multiruns.py` — references `model_config.epochs`, etc. (to be scaffolded)
- `src/configs/multirun_descriptions.py` — CIFAR-specific sweep docs (to be scaffolded)
- `src/configs/workflow.py` — registers `Cifar10CNNWorkflow` with `name="cifar10_cnn"` (to be scaffolded)
- `src/configs/assets.py` — CIFAR asset RIDs (to be scaffolded)

Also in `src/deriva_ml_model_template.egg-info/` (untracked build artifact —
egg-info was already in .gitignore; not in index).

No cifar refs in `src/models/`, `src/scripts/`, or `tests/` (all CIFAR files there deleted).

---

## Concerns

1. **`model_config` group is now empty** — `base.py` will fail to resolve
   `default_model` until A4 adds a generic scaffold. This is expected and by design.

2. **`src/configs/workflow.py`** directly imports/references `Cifar10CNNWorkflow`
   and registers `name="cifar10_cnn"`. This will cause an import error if the
   config package is loaded before A4 converts it. Expected breakage during strip.

3. **`tests/test_notebook_examples.py`** was not in the deletion list and remains.
   It may reference CIFAR notebooks — a later task should verify or delete it.

4. **egg-info** was already tracked correctly (not in git index), `.gitignore`
   already contained `*.egg-info/` — no change needed.
