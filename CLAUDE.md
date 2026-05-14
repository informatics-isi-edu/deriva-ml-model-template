# CLAUDE.md

Agent instructions for working in this DerivaML model template.

**For usage** (setup, running models, loading data, configuring
catalogs, project layout): see [README.md](README.md). Don't
duplicate that material here.

This file covers what an AI agent needs to know to work *in* the
template — conventions, gotchas, where things live.

## Project context

This is a template for ML models integrated with DerivaML. As
shipped it contains a CIFAR-10 CNN example with 7 model variants.
Users typically clone it, replace the example with their own model
and data, and ship.

The platform underneath:
- **deriva-ml** — core Python library for reproducible ML on
  Deriva catalogs.
- **Hydra-zen** — Python-first configuration (no YAML).
- **uv** — dependency management, script execution.

## Source layout

- `src/configs/` — Hydra-zen configuration (Python, no YAML).
  - `base.py` — `BaseConfig` dataclass.
  - `cifar10_cnn.py` — model configs (architectures,
    hyperparameters).
  - `datasets.py` — `DatasetSpecConfig` per dataset.
  - `deriva.py` — Deriva connection configs.
  - `workflow.py` — Workflow definitions.
  - `assets.py` — Asset RID configs for model weights and
    predictions.
  - `experiments.py` — model + dataset combinations.
  - `multiruns.py` — parameter sweep configs.
  - `multirun_descriptions.py` — rich markdown for multirun parent
    executions.
  - `roc_analysis.py` — ROC notebook asset configs.
  - `dev/` — per-environment overrides
    (`deriva_<env>.py`, `datasets_<env>.py`, etc.).
- `src/models/` — model implementations.
  - `cifar10_cnn.py` — CNN model, training loop, prediction
    recording.
  - `model_protocol.py` — Protocol/interface model functions
    implement.
- `src/scripts/` — data loading scripts (importable Python
  package).
- `scripts/` — standalone shell/CLI utilities (not a Python
  package).
- `notebooks/` — analysis notebooks.
- `tests/` — pytest smoke tests for configs.

## Conventions

- **Use `uv` for everything.** Always `uv run <cmd>` — never
  invoke `pytest`, `ruff`, `python`, or `bump-version` directly.
- **Google-style docstrings** on every function, method, and class.
  Include `Args:`, `Returns:`, `Raises:`, and a runnable `Example:`
  block.
- **No backwards-compat shims.** If something is unused, delete it.
  No "removed" comment placeholders, no dead exports.
- **No over-engineering.** Only add what the current task requires.
- **TDD when adding new code.** Write a failing test, make it pass,
  refactor. Existing tests in `tests/test_configs_load.py` are
  configuration smoke tests — add a similar file when introducing
  a new module.

## Standard commands

See [README.md](README.md) §3–8 for the user-facing command list.
The agent should reach for these:

```bash
uv sync                                  # install/update deps
uv sync --group=jupyter                  # + Jupyter
uv sync --group=torch                    # + PyTorch

uv run python -m pytest tests/ -v        # run tests (see gotcha below)
uv run ruff check src tests              # lint
uv run ruff format src tests             # format
uv run bump-version patch|minor|major    # release (clean tree required)

uv run deriva-ml-run --info              # list configs
uv run deriva-ml-run dry_run=true        # dry run (no catalog writes)
```

## Gotchas

- **Use `uv run python -m pytest`, not `uv run pytest`.** The venv's
  `pytest` shim has a stale shebang pointing at system Python 3.10.
  `uv sync --reinstall` fixes it if you hit this.
- **Two `scripts/` dirs:** `src/scripts/` is an importable Python
  package; `scripts/` is for standalone shell/CLI utilities (not a
  package). When adding new code, pick the right one.
- **`num_workers=0` in DataLoaders on macOS.** `fork()` + MPS/GPU
  threads deadlock. Keep DataLoaders single-worker on macOS.
- **Commit before running.** DerivaML records the git commit hash
  for provenance; dirty-tree warnings appear when running with
  uncommitted changes. For fast iteration during development:
  `DERIVA_ML_ALLOW_DIRTY=true uv run <command>`. Don't set this in
  production runs — provenance is what it protects.

## Key rules when modifying configs

- **The defaults in `src/configs/datasets.py` ship with RIDs from a
  previous demo catalog and will not work in a fresh checkout
  until the user runs `load-cifar10` and updates them.** README §7
  documents the update procedure for users; the agent should
  follow the same procedure when configuring a new environment.
- **Use labeled datasets for evaluation.** `cifar10_small_labeled_split`
  or `cifar10_labeled_split` carry ground truth on both train and
  test partitions and are the right choice for ROC analysis,
  accuracy metrics, or any evaluation work. The `*_split` configs
  (without `_labeled`) are for training-only flows.
- **`Execution_Asset`** is for model outputs (weights, predictions,
  plots). `Execution_Metadata` is auto-managed; don't write to it
  directly.
- **Test with `dry_run=true`** before any catalog-writing run.

## Notebook runner specifics

- **`--config` on `deriva-ml-run-notebook` does NOT override the
  `run_notebook()` config name.** Use positional Hydra overrides
  (e.g., `assets=my_assets_prod`).
- **`--host` / `--catalog` are papermill parameters, NOT Hydra
  overrides.** They set the notebook's connection target but
  don't change which `deriva_ml=` config is resolved. To target a
  non-default catalog, pass `deriva_ml=<config_name>` as a Hydra
  override AND register the connection in
  `src/configs/dev/deriva_<env>.py`.

## Related docs

- [README.md](README.md) — user-facing setup and usage.
- [CIFAR10.md](CIFAR10.md) — end-to-end CIFAR-10 walkthrough.
- [Experiments.md](Experiments.md) — experiment configuration
  reference.
- [experiment-decisions.md](experiment-decisions.md) — design
  rationale and decision log for the example model.
