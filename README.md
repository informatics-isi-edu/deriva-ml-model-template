# DerivaML Model Template

This repository provides a template for using DerivaML and EyeAI to develop a machine learning (ML) model.

DerivaML is a Python library that helps you create and run reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs, so you can recreate or audit results later. Because provenance depends on versioned source code, this template assumes your project lives in GitHub.

This template includes:
- A basic project configuration
- A simple Python script entrypoint
- An example model and configuration modules
- An equivalent Jupyter notebook
- A sample parameter and environment setup

## Creating a new repository
This repository is set up as a template. Its intended use is to create a new repository using the template and then customize it for your specific model.

To create a repository from the template, follow the instructions here: [using templates](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

Templates for models set up as runnable Python scripts and Jupyter notebooks are provided.

## Project layout
```
.
├─ pyproject.toml                  # Project metadata, dependencies, uv config
├─ README.md                       # This guide
├─ coding-guidelines.md            # Operational & coding guidelines (see link below)
├─ src/
│  ├─ deriva_run.py                # Script entrypoint (Hydra main)
│  ├─ model_runner.py              # Helper for running models in DerivaML env
│  ├─ models/
│  │  └─ simple_model.py           # Example model function
│  └─ configs/
│     ├─ deriva.py                 # DerivaML configs (host, catalog, etc.)
│     ├─ datasets.py               # Dataset collection configs
│     ├─ assets.py                 # Asset ID list configs (e.g., model weights)
│     ├─ simple_model.py           # Model config definitions and variants
│     └─ experiments.py            # Placeholder for experiment presets
├─ notebooks/
│  └─ notebook_template.ipynb      # Example notebook
└─ .github/workflows/ci.yml        # GitHub Actions workflow
```

## GitHub Actions
This template uses GitHub Actions to automate the versioning of the model. GitHub Actions are configured in the `.github` directory, which you may not see by default in your file browser.

## Working with uv (setup, environments, and common tasks)
This template uses `uv` as the project management tool for environments and dependencies. If you haven’t installed `uv` yet, see the official docs: https://docs.astral.sh/uv/

### 1) Initialize your environment
Run this from the repository root:
```
uv sync
```
This creates a new Python virtual environment and a lock file. Commit the resulting `uv.lock` to your repository.

If you plan to use notebooks, initialize these extras as well:
```
uv run nbstripout --install
uv sync --group=jupyter
uv run deriva-ml-install-kernel
```
- `nbstripout` installs a Git hook to strip output cells automatically on commit.
- `deriva-ml-install-kernel` registers a Jupyter kernel for this environment.

You can verify available kernels with:
```
uv run jupyter kernelspec list
```

### 2) Optional dependency groups
Install extra groups on demand:
- Jupyter: `uv sync --group=jupyter`
- PyTorch: `uv sync --group=pytorch`
- TensorFlow: `uv sync --group=tensorflow`

If you plan to use these options regularly, add them to the `default-groups` list in `pyproject.toml` so they are always installed on `uv sync`.

## Updating Modules including DerivaML
You can use `uv` to update specific packages in your application. For example, to update DerivaML:
```
uv sync --upgrade-package deriva-ml
```
You can upgrade all packages as well—proceed with caution, as upgrading to the latest PyTorch or TensorFlow can require compatible drivers. One way around this is to pin specific versions of these libraries in `pyproject.toml` using `uv add`.

To upgrade the entire dependency set:
```
uv lock --upgrade
uv sync
```
After the upgrade, commit your updated `uv.lock` file.

### 3) Activating and deactivating the virtual environment
You can always prefix commands with `uv run`. If you prefer to activate the environment for a shell session:
- Bash/Zsh: `source .venv/bin/activate`
- Fish: `source .venv/bin/activate.fish`
- Csh/Tcsh: `source .venv/bin/activate.csh`

When finished, run `deactivate` to leave the environment.

### 4) Authenticating to access catalog data
Before accessing catalog data, log into Globus:
```
uv run deriva-globus-auth-utils login --host www.eye-ai.org
```

### 5) Running scripts and notebooks
- Run the main script with defaults:
```
uv run src/deriva_run.py
```
- Run the notebook non-interactively in the DerivaML environment (reproducible execution):
```
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb \
  --host www.eye-ai.org \
  --catalog 2 \
  --kernel <repository-name>
```
This executes all cells and uploads the executed notebook to the catalog.

## Managing releases and version tags
In addition to committing, it is advisable to tag the model at significant milestones and publish releases. The template includes a small script and a GitHub Action that together streamline creating release tags for a model. DerivaML uses semantic versioning.

Use the version bump script like this:
```
uv run bump-version major|minor|patch
```
The script automatically uses commit logs from pull requests to generate release notes.

## Getting the current version
DerivaML uses `setuptools-scm` to determine the current version of the model. This produces a dynamic version number that updates automatically when you create a new release using `bump-version` or commit on top of the latest tag.
```
uv run python -m setuptools_scm
```

## Experiment Management
DerivaML uses the Hydra configuration framework to manage configurations of scripts and notebooks and to conduct different kinds of experiments.

Rather than hard-coding values, use Hydra to specify values that can be changed at runtime. hydra‑zen is integrated to provide a simple Pythonic way to create and configure ML models.

- Hydra documentation: https://hydra.cc/docs/intro/
- hydra‑zen documentation: https://mit-ll-responsible-ai.github.io/hydra-zen/

## Configuration with Hydra & hydra‑zen
This template registers configuration choices with Hydra’s in-memory config store using hydra‑zen. 
Your script consumes these configs via a typed function interface, making it easy to switch datasets, assets, and model variants at runtime.

- Entrypoint: `src/deriva_run.py`
- Registered app config name: `deriva_model`
- Default selections (Hydra choices):
  - `deriva_ml: local`
  - `datasets: test1`
  - `assets: weights_1`
  - `model_config: default_model`

Conceptually, the script registers a top-level config stored under `deriva_model` and uses Hydra to wire defaults and overrides:
```
from hydra_zen import builds, store
from model_runner import run_model

# Register the top-level app config
_deriva_model = builds(
    run_model,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "local"},
        {"datasets": "test1"},
        {"assets": "weights_1"},
        {"model_config": "default_model"},
    ],
)
store(_deriva_model, name="deriva_model")
```

Where each group is defined:
- `deriva_ml`: `src/configs/deriva.py` (e.g., `local`, `eye-ai`)
- `datasets`: `src/configs/datasets.py` (e.g., `test1`, `test2`, `test3`) using `DatasetSpecConfig`
- `assets`: `src/configs/assets.py` (e.g., `weights_1`, `weights_2`) using `AssetRIDConfig`
- `model_config`: `src/configs/simple_model.py` (e.g., `default_model`, `epochs_20`, `epochs_100`)

Note on dataset types: this template stores datasets as lists of `DatasetSpecConfig` objects in `src/configs/datasets.py`.
The runner accepts these and constructs an `ExecutionConfiguration` (which normalizes the types internally). 
If you prefer, you can change the runner’s type hints to `list[DatasetSpecConfig]` to match the stored configs exactly.

Tip: see Hydra CLI help for your script:
```
uv run src/deriva_run.py --help
```

### Model configuration pattern (hydra‑zen)
This repository demonstrates the “build once, extend by instantiation” approach:
```
from hydra_zen import builds, store
from models.simple_model import simple_model

SimpleModelConfig = builds(
    simple_model,
    learning_rate=1e-3,
    epochs=10,
    populate_full_signature=True,
    zen_partial=True,
)

store(SimpleModelConfig, group="model_config", name="default_model")
store(SimpleModelConfig, epochs=20, group="model_config", name="epochs_20")
store(SimpleModelConfig, epochs=100, group="model_config", name="epochs_100")
```
- Only one `builds(...)` call; variants override fields on the built config.
- Hydra produces a callable; the script later invokes it.

### Running the script and overriding configs
- Use defaults:
```
uv run src/deriva_run.py
```
- Choose a different dataset or assets:
```
uv run src/deriva_run.py +datasets=test2 +assets=weights_2
```
- Choose a different model variant and/or override fields inline:
```
uv run src/deriva_run.py +model_config=epochs_100
uv run src/deriva_run.py +model_config.epochs=50
```
- Enable a dry run (downloads inputs, skips write-backs):
```
uv run src/deriva_run.py +dry_run=true
```

### Experiments (presets)
You can maintain curated experiment presets (named combinations of choices) in `src/configs/experiments.py`.
```
# Example experiment preset (already included in the template):
from hydra_zen import store

experiment_store = store(group="experiments")
experiment_store(
    {"datasets": "test2", "assets": "weights_2", "model_config": "epochs_100"},
    name="high_epochs_alt_data",
)
```
Run a single preset
```uv run src/deriva_run.py +experiments=high_epochs_alt_data```

The template includes example presets named `run1` and `run2` in `src/configs/experiments.py`.
You can also run multiple experiments in one invocation:
```
uv run src/deriva_run.py --multirun +experiment=run1,run2
```
## Using this template

From GitHub, create a new repository from this template.  

You need to have a "bridge" function that will connect the DerivaML,  In the template, this is the `simple_model` function.
```def simple_model(learning_rate: float, epochs: int,
                 ml_instance: DerivaML,
                 execution: Execution | None = None) -> None:
                 ```
 
- In the models directory, create a copy of the  the file `simple_model.py` to match your model name. If you don't
want to use the models directory, you can put your model code whereever you like and adjust the imports accordingly.
-
Replace the simple_model.py file in the models directory with your own model code.  
The initial arguments to your model should be whatever you want to vary in the underlying ML code.  You should 
keep the last ExecutionConfiguration argument, which is used to configure the model run and will automatically be
added by the framework when calling your model function.

Replace configs/simple_model.py with an version for your model that defines the model variants you want to run. 
You do not have to have a default varient, but this may be useful for testing.  


The model function should be defined in the model_runner.py file.
he arguments to the model function should match the arguments in the model config.
- Customize the project name and description
- Update the README to describe your model and its use case

## Loading CIFAR-10 Dataset

This template includes a script to load the CIFAR-10 dataset into a Deriva catalog. This is useful for testing ML workflows with a standard dataset.

### Usage

```bash
load-cifar10 --host <hostname> --catalog_id <catalog_id> [options]
```

### Required Arguments

- `--host`: The Deriva server hostname (e.g., `www.eye-ai.org`)
- `--catalog_id`: The catalog ID to load data into

### Optional Arguments

- `--num_images`: Number of images to load (default: 100)
- `--domain_schema`: Domain schema name (default: `cifar10`)
- `--working_dir`: Working directory for temporary files
- `--batch_size`: Batch size for uploads (default: 50)
- `--train_only`: Only load training images
- `--test_only`: Only load test images

### Example

```bash
# Load 500 CIFAR-10 images to a catalog
uv run load-cifar10 --host dev.eye-ai.org --catalog_id 5 --num_images 500

# Load only training images
uv run load-cifar10 --host dev.eye-ai.org --catalog_id 5 --num_images 200 --train_only
```

The script will:
1. Download CIFAR-10 data from torchvision
2. Create the necessary schema and tables if they don't exist
3. Upload images as assets with their labels
4. Create appropriate dataset entries

## Recommended Workflow and Coding Guidelines
We maintain operational and coding guidelines in a separate document:
- See: [coding-guidelines.md](./coding-guidelines.md)