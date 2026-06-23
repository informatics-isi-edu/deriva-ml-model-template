# Task A5 Report — Config Smoke Test + Strip Commit

**Status:** DONE
**Commit SHA:** `6f4e7ae`
**Test summary:** 10 passed, 0 failed, 6 warnings (pydantic/globus deprecations from upstream libs)
**Branch:** `chore/strip-cifar-to-skeleton`

---

## Test file

`tests/test_configs_load.py` — 10 tests:

| Test | Assertion |
|------|-----------|
| `test_configs_package_imports` | `import configs` succeeds |
| `test_model_config_imports` | `import configs.model` succeeds |
| `test_datasets_config_imports` | `import configs.datasets` succeeds |
| `test_experiments_config_imports` | `import configs.experiments` succeeds |
| `test_base_config_imports` | `DerivaModelConfig` exported and not None |
| `test_default_model_registered` | `store.get_entry("model_config", "default_model")` returns correct entry |
| `test_default_dataset_registered` | `store.get_entry("datasets", "default_dataset")` returns correct entry |
| `test_required_dataset_sentinels_registered` | `no_datasets` and `none` present in datasets group |
| `test_default_experiment_registered` | `store.get_entry("experiment", "default")` returns correct entry |
| `test_no_cifar_entries_registered` | 6 known CIFAR names raise `KeyError` — strip verified |

---

## pytest output

```
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/carl/GitHub/DerivaML/deriva-ml-model-template-strip
configfile: pyproject.toml
plugins: cov-7.1.0, hydra-core-1.3.2, anyio-4.13.0
collecting ... collected 10 items

tests/test_configs_load.py::test_configs_package_imports PASSED          [ 10%]
tests/test_configs_load.py::test_model_config_imports PASSED             [ 20%]
tests/test_configs_load.py::test_datasets_config_imports PASSED          [ 30%]
tests/test_configs_load.py::test_experiments_config_imports PASSED       [ 40%]
tests/test_configs_load.py::test_base_config_imports PASSED              [ 50%]
tests/test_configs_load.py::test_default_model_registered PASSED         [ 60%]
tests/test_configs_load.py::test_default_dataset_registered PASSED       [ 70%]
tests/test_configs_load.py::test_required_dataset_sentinels_registered PASSED [ 80%]
tests/test_configs_load.py::test_default_experiment_registered PASSED    [ 90%]
tests/test_configs_load.py::test_no_cifar_entries_registered PASSED      [100%]

======================== 10 passed, 6 warnings in 2.10s ========================
```

---

## --list-configs output (no CIFAR entries)

```
Available Hydra Configuration Groups:
==================================================

Top-level configs:
  - analysis
  - deriva_base
  - deriva_model

assets:
  - default_asset
  - no_assets

datasets:
  - default_dataset
  - no_datasets
  - none

deriva_ml:
  - default_deriva

experiment:
  - default

model_config:
  - default_model

workflow:
  - default_workflow

multirun:
  - example_sweep: Example Sweep
```

---

## ruff output

```
All checks passed!
```

---

## Concerns

None. The 6 pytest warnings are upstream pydantic/globus-sdk deprecations inside deriva-ml itself — not caused by this change.
