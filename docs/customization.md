# Customizing this Template

This template ships as a **runnable skeleton**. Out of the box it resolves and
dry-runs end to end: a no-op placeholder model (`example_model`) on an empty
default dataset, wired through every config group. Your job is to fill in the
scaffolds in `src/configs/`.

Each config module follows the same shape: a module docstring, a **live
default** (so a fresh clone always resolves), and **one commented example** you
uncomment and fill in. You never write a config group from scratch — you copy
the example that is already there.

Work through the six steps below in order. Each one names the exact file, what
the fields mean, the block to uncomment, and how to verify before moving on.

Before you start, sanity-check that the skeleton resolves:

```bash
uv run deriva-ml-run --list-configs                # the config-group menu
uv run deriva-ml-run +experiment=default --cfg job # the resolved default config
```

---

## 1. Point at your catalog — `src/configs/deriva.py`

This module registers the `deriva_ml` config group: which Deriva catalog a run
connects to. The shipped `default_deriva` is a placeholder pointing at
`hostname="localhost"`, `catalog_id=0` — which is intentionally not a real
catalog.

Edit the two fields in the `deriva_store(...)` call:

```python
deriva_store(
    DerivaMLConfig,
    name="default_deriva",
    hostname="localhost",          # <- your Deriva host
    catalog_id=0,                  # <- your catalog ID
    use_minid=False,
    ...
)
```

You can also leave the file alone and pass `--host <hostname> --catalog <id>` on
every CLI invocation, but setting `default_deriva` once is less error-prone.

For multi-environment work (a dev catalog and a prod catalog, say), register
additional `DerivaMLConfig` entries — one per host/catalog — in
`src/configs/dev/deriva_<env>.py` under the same `deriva_ml` group, then select
with `deriva_ml=<name>` on the CLI.

**Verify:**

```bash
uv run deriva-ml-run --cfg job | grep -A3 deriva_ml   # hostname/catalog_id look right
```

See the `/deriva-ml:execution-lifecycle` skill for the connection/pre-flight
details.

---

## 2. Declare your datasets — `src/configs/datasets.py`

This module registers the `datasets` group: named sets of catalog datasets (by
RID + version) that a model or notebook consumes. The runner selects one with
`datasets=<name>` and the model reads it from `execution.datasets`.

RIDs are catalog-specific, so the template ships with **no live dataset
configs** — only the required empty sentinels (`default_dataset`, `no_datasets`,
`none`). The empty `default_dataset` passes config validation but fails at
execution time ("Dataset '' not found"), by design: a fresh clone must not
silently run against someone else's RIDs.

Uncomment and fill in the example near the top of the file:

```python
datasets_store(
    with_description(
        [DatasetSpecConfig(rid="<your-rid>", version="<ver>")],
        "<one-line description of this dataset>",
    ),
    name="<your_dataset>",
)
```

- **`rid`** — the dataset RID in your catalog. Discover RIDs with
  `ml.find_datasets()` after loading data.
- **`version`** — the released dataset version (also visible via
  `ml.find_datasets()`).
- **`with_description(...)`** is optional but recommended: it surfaces a
  description in `deriva-ml-run --info`.

Once you have a primary dataset, point `default_dataset` at it (replace its empty
`[]` with a real `DatasetSpecConfig` list) so a plain `uv run deriva-ml-run`
uses it.

For multi-environment setups, register parallel `<name>_<env>` configs in
`src/configs/dev/datasets_<env>.py` rather than editing the defaults, and select
with `datasets=<name>_<env>`.

**Verify:**

```bash
uv run deriva-ml-run --list-configs           # your dataset name appears under "datasets"
uv run deriva-ml-run datasets=<your_dataset> --cfg job
```

The `/deriva-ml:dataset-lifecycle` skill covers creating, splitting,
subsampling, and wiring datasets into this file.

---

## 3. Add your model — `src/models/model_protocol.py` + `src/configs/model.py`

A model is any callable that implements the `DerivaMLModel` interface
(re-exported in `src/models/model_protocol.py` from `deriva_ml.execution`). It
receives its Hydra-configured fields as keyword arguments, plus two injected
arguments:

- **`ml_instance`** — the `DerivaML` connection.
- **`execution`** — the run context. Use it to read input datasets, register
  output assets, and update run status.

It returns `None` — models report results through `execution` (registered
assets, created datasets, status updates), never via a return value.

`src/configs/model.py` ships a placeholder `example_model` that does no real
training; replace it with your own function, then update the config build:

```python
ExampleModelConfig = builds(
    example_model,                 # <- your model function
    learning_rate=1e-3,            # <- your default hyperparameters
    epochs=10,
    populate_full_signature=True,
    zen_partial=True,              # leaves ml_instance/execution unbound
)

# REQUIRED: ``default_model`` must exist — base.py lists it in its defaults.
model_store(ExampleModelConfig, name="default_model")
```

`zen_partial=True` is required so hydra-zen leaves `ml_instance` / `execution`
unbound — the runner supplies them when the execution starts. The config named
`default_model` **must** exist; `src/configs/base.py` lists
`{"model_config": "default_model"}` in its Hydra defaults, so the whole config
tree fails to resolve without it.

To add alternate hyperparameter sets, uncomment the example at the bottom of the
file — each variant reuses the base build and overrides only the fields that
differ:

```python
model_store(
    ExampleModelConfig,
    name="<your_variant>",
    learning_rate=1e-2,
    epochs=30,
    zen_meta={"description": "Higher learning rate, longer training."},
)
```

**Verify:**

```bash
uv run python -m pytest tests/ -q                 # config smoke tests still pass
uv run deriva-ml-run --list-configs               # default_model + your variants appear
uv run deriva-ml-run model_config=<your_variant> --cfg job
```

The `/deriva-ml:new-model` and `/deriva-ml:experiment-lifecycle` skills walk
through authoring the function and its config.

---

## 4. Define experiments — `src/configs/experiments.py`

An *experiment* is a named, pre-wired combination of config groups — typically a
model config paired with a dataset config — so a meaningful run is one name
instead of a long string of CLI overrides. Experiments are registered under the
`experiment` group at the `_global_` package so their defaults override config
groups at the root.

The template ships one live `default` experiment (the placeholder model on the
default dataset). Uncomment and fill in the example to add your own:

```python
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model_config": "<your_model_config>"},
            {"override /datasets": "<your_dataset_group>"},
        ],
        description="<what this experiment evaluates>",
        bases=(DerivaModelConfig,),
    ),
    name="<your_experiment>",
)
```

- The names in `override /model_config` and `override /datasets` must match
  configs you registered in `model.py` (step 3) and `datasets.py` (step 2).
- When overriding optional fields (e.g. `script_config`), set them to `MISSING`
  in `make_config` so Hydra fills them from the defaults list rather than
  shadowing the resolved value with the base's `None` default.

**Verify:**

```bash
uv run deriva-ml-run --list-configs                       # your experiment appears
uv run deriva-ml-run +experiment=<your_experiment> --cfg job
uv run deriva-ml-run +experiment=<your_experiment> dry_run=true   # validates vs catalog
```

---

## 5. (Optional) Sweeps, assets, and workflows

Three more scaffolds carry a commented example each. Reach for them as needed.

### Sweeps — `src/configs/multiruns.py`

A *multirun* bundles a set of Hydra override axes with a rich markdown
description under one name. Running `+multirun=<name>` launches the whole sweep —
no `--multirun` flag needed. Comma-separated values in an override become the
sweep axes. The template ships one live `example_sweep`. Add your own:

```python
multirun_config(
    "<your_sweep>",
    overrides=[
        "+experiment=<your_experiment>",
        "model_config.<param_a>=<v1,v2>",
        "model_config.<param_b>=<w1,w2>",
    ],
    description=MY_SWEEP_DESCRIPTION,  # from multirun_descriptions.py
)
```

Define `MY_SWEEP_DESCRIPTION` in `src/configs/multirun_descriptions.py` (it lands
on the parent execution recorded for the sweep). **Verify** with
`uv run deriva-ml-run --info` (lists multiruns) and
`uv run deriva-ml-run +multirun=<your_sweep> --cfg job`.

### Input assets — `src/configs/assets.py`

An *asset config* names a list of asset RIDs (model weights, prediction files,
plots) that a model or notebook consumes as inputs. Asset RIDs are produced *by*
prior executions — a training run that uploaded a weights file, say — so the
template ships only the empty `default_asset` / `no_assets` sentinels. After an
execution prints its output RIDs, register them:

```python
asset_store(
    with_description(
        ["<your-rid>"],
        "<what this asset is, e.g. trained weights from run X>",
    ),
    name="<your_assets>",
)
```

Select with `assets=<your_assets>`. The `/deriva-ml:work-with-assets` and
`/deriva-ml:execution-lifecycle` skills cover producing and wiring asset RIDs.

### Workflow metadata — `src/configs/workflow.py`

A `Workflow` describes the computational pipeline behind a run (name,
description, one or more workflow types) and automatically captures Git
provenance when an execution starts. The required `default_workflow` ships with
generic placeholder values; replace them, or add named workflows:

```python
MyWorkflow = builds(
    Workflow,
    name="<Pipeline Name>",
    workflow_type=["<Type A>", "<Type B>"],   # a string or a list
    description="<what the pipeline does>",
    populate_full_signature=True,
)
workflow_store(MyWorkflow, name="<your_workflow>")
```

Select with `workflow=<your_workflow>`.

### Analysis notebooks — `src/configs/analysis.py`

To add an analysis notebook, drop `notebooks/<your_notebook>.ipynb` and register
a matching config. By DerivaML convention, notebook `X.ipynb` uses the config
registered as `notebook_config("X", ...)`; `run_notebook()` derives the name
from the calling notebook's filename. The template ships a generic `analysis`
config — copy it:

```python
notebook_config(
    "<your_notebook>",
    config_class=AnalysisConfig,
    defaults={"assets": "<your_assets>", "datasets": "no_datasets"},
    description="<what this analysis does>",
)
```

**Verify** with
`uv run deriva-ml-run-notebook notebooks/<your_notebook>.ipynb --list-configs`.
See the `/deriva-ml:run-notebook` skill.

---

## 6. Rename the project — `pyproject.toml`

Finally, make the project yours: set `name` and `description` (and `authors`,
`urls`, etc. as appropriate) in `pyproject.toml`. This is also where you adjust
dependencies and dependency groups (e.g. drop `pytorch` if your model does not
need it).

After renaming, re-sync and re-run the smoke tests:

```bash
uv sync
uv run python -m pytest tests/ -q
```

---

## Putting it together

A typical first customization touches steps 1–4: point at a catalog, declare one
dataset, drop in a model, and define one experiment pairing them. Then:

```bash
# Commit first — DerivaML records the git commit hash for provenance.
git add -A && git commit -m "Configure my model + dataset"

# Validate against the catalog without writing.
uv run deriva-ml-run +experiment=<your_experiment> dry_run=true

# Run for real.
uv run deriva-ml-run +experiment=<your_experiment>
```

For a complete worked reference — every step above filled in with a real model,
datasets, sweeps, and an analysis notebook — see
[`deriva-ml-cifar-example`](https://github.com/informatics-isi-edu/deriva-ml-cifar-example).
