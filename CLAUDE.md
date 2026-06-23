# CLAUDE.md

Agent instructions for working in this DerivaML model template.

**For usage** (setup, running models, loading data, configuring
catalogs, project layout): see [README.md](README.md). Don't
duplicate that material here.

This file covers what an AI agent needs to know to work *in* the
template — conventions, gotchas, where things live.

## Project context

This is a template for ML models integrated with DerivaML. As
shipped it is a **runnable skeleton**: each `src/configs/*.py` is a
self-documenting scaffold (a docstring, a live default, and one
commented example to uncomment), and `src/configs/model.py` registers
a no-op placeholder `default_model` so a fresh clone resolves and
dry-runs end to end. Users clone it, fill in the scaffolds with their
own catalog, datasets, and model, and ship. A complete worked example
lives in a separate repo,
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example) —
do not re-add the worked example here.

The platform underneath:
- **deriva-ml** — core Python library for reproducible ML on
  Deriva catalogs.
- **Hydra-zen** — Python-first configuration (no YAML).
- **uv** — dependency management, script execution.

## Source layout

- `src/configs/` — Hydra-zen configuration (Python, no YAML). Each
  module is a scaffold: docstring + live default + one commented
  example.
  - `base.py` — `BaseConfig` / `DerivaModelConfig`.
  - `model.py` — model function + hyperparameter configs; registers
    `default_model` (a placeholder the user replaces).
  - `datasets.py` — `DatasetSpecConfig` per dataset.
  - `deriva.py` — Deriva connection configs.
  - `workflow.py` — Workflow definitions.
  - `assets.py` — Asset RID configs (model weights, predictions).
  - `experiments.py` — model + dataset combinations.
  - `multiruns.py` — parameter sweep configs.
  - `multirun_descriptions.py` — rich markdown for multirun parent
    executions.
  - `analysis.py` — analysis notebook config (consumes asset RIDs).
  - `dev/` — per-environment overrides
    (`deriva_<env>.py`, `datasets_<env>.py`, etc.).
- `src/models/` — model implementations (user-authored).
  - `model_protocol.py` — Protocol/interface model functions
    implement (re-exported from deriva-ml).
- `src/scripts/` — data loading scripts (importable Python
  package).
- `scripts/` — standalone shell/CLI utilities (not a Python
  package).
- `notebooks/` — analysis notebooks.
- `tests/` — pytest smoke tests for configs.
- `docs/design/` — the design docs (Specify-phase contracts),
  authored **before** the work via `/deriva-ml:design-experiment`,
  in per-entity subdirs (`<slug>.md` each):
  - `docs/design/experiment/` — per-experiment designs (the contract
    the config implements); see its `README.md`.
  - `docs/design/dataset/` — per-dataset designs (before the build).
  - `docs/design/feature/` — per-feature designs (before create).
  - `docs/design/model/` — per-model designs (before authoring the
    model fn + config).

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

uv run deriva-ml-run --list-configs      # list configs
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

- **`src/configs/datasets.py` ships with NO live dataset configs** —
  only the required empty sentinels (`default_dataset`, `no_datasets`,
  `none`). The empty `default_dataset` passes config validation but
  fails at execution time ("Dataset '' not found") by design: a fresh
  clone must not silently run against someone else's RIDs. The user
  registers their own `DatasetSpecConfig(rid=..., version=...)` by
  uncommenting the example in that file. README "Customizing this
  template" and `docs/customization.md` document the procedure.
- **Each config scaffold has ONE commented example** — point users
  (and yourself) at uncommenting and filling it in rather than writing
  config from scratch.
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
- [docs/customization.md](docs/customization.md) — the per-file
  walkthrough for turning the skeleton into a project.
- [tacit-knowledge.md](tacit-knowledge.md) — design
  rationale and decision log.
- [docs/design/](docs/design/) — design docs (the plan, written
  before the work) in per-entity subdirs: `experiment/`, `dataset/`,
  `feature/`, `model/`.
- [`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example) —
  the complete worked example (separate repo).

## Test plans

- When the user asks to run an end-to-end platform test (any
  variant of "let's do an e2e run", "exercise the stack", or
  "test the platform"), the current plan of record is
  [docs/test-plans/2026-05-20-e2e-multipersona.md](docs/test-plans/2026-05-20-e2e-multipersona.md).
  Read it before kicking off the run. The prior single-agent
  platform-fitness plan (May 2026) is superseded but kept in
  `docs/superpowers/specs/` for historical reference.

- **Don't tear down the e2e worktree at wrap-up.** After the
  Analyst arc commits and the evaluation report lands, push the
  branch to `origin/archive/e2e-test-<YYYY-MM-DD>` for the
  historical record, but **leave the local worktree at
  `../deriva-ml-model-template-e2e/` in place**. The user needs to
  be able to open the evaluation report, the per-persona findings
  files, the executed notebooks, and the tacit-knowledge entries
  *without* having to check out an archived branch or read GitHub.
  Only remove the worktree when the user explicitly asks (or when
  setting up the next run's worktree, which would otherwise
  conflict). The pre-bootstrap step 0 of the next run is the
  right place to ask "the prior run's worktree is still here; OK
  to archive and remove it?"
