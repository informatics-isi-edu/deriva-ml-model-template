# DerivaML Model Template

This repository provides a template for using DerivaML to develop a ML model.

Deriva-ML is a python library to simplify the process of creating and executing reproducible machine learning workflows
using a deriva catalog.

The code provenance aspects of DerivaML depend on a script or notebook being tracked in Git. 
This template assumes that GitHub is being used.

This template repository contains a basic project file, a simple Python script and equivalent notebook, along with a sample parameter file.

## Creating a new repository

This repository is set up as a template.  Its intended use is to create a new repository using the template and then customize it for your specific model.
To create a repository from the template, follow the instructions provided here [GitHub Template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

Templates for models set up as runnable python scripts, and Jupyter notebooks are provided.

# Project layout
```
.
├─ pyproject.toml                  # Project metadata, dependencies, uv config
├─ README.md                       # This guide
├─ src/
│  ├─ deriva_run.py                # Stand-alone script entrypoint for configuration (Hydra main)
|  ├─ model_runner.py              # Script entrypoint for running models in deriva-ml environment
│  ├─ models/
│  │  └─ simple_model.py           # Example model function
│  └─ configs/
│     ├─ deriva.py                 # DerivaML configs (catalog host, etc.)
│     ├─ datasets.py               # Dataset configs (list(s) of dataset specs)
│     ├─ assets.py                 # Asset list configs (e.g., weight files)
│     ├─ simple_model.py           # Model config definitions and variants
│     └─ experiments.py            # Placeholder for experiment presets
├─ notebooks/
│   └─ notebook_template.ipynb      # Example notebook
└─ .github/workflows/ci.yml        # GitHub Actions CI workflow
       
```
# GitHub Actions

This template uses GitHub Actions to automate the versioning of the model.  
GitHub Actions are configured in the `.github` directory, which you may not see by default
in your file browser. 


# Project Management

This template uses `uv` as a project management tool.  As a prerequisite, you should install the *uv* tool into your execution environment.
Once installed you should use the uv command line to manage your dependencies and virtual environment.

Instructions on how to install UV and use it as a project management tool can be found [here.](https://docs.astral.sh/uv/)


# Authenticating

You must be logged into Globus before you can access data in the catalog.
You can do this by using the following command:

```
uv run deriva-globus-auth-utils login --host www.eye-ai.org
```
# Initializing Your Repository

The baseline initialization of your repository is achieved by running the command:
```aiignore
uv sync
```

This will create a new python virtual environment, and an associated lock file.  You should
add the resulting uv.lock file to your Git repository.

If you are planning on running notebooks, you should initialize your repository with the command:
```aiignore

uv run nbstripout --install 
uv sync --group=jupityer
uv run deriva-ml-install-kernel
```

These commands install a Git pre-commit hook to strip output from your notebook prior to a commit
and create a jupyter kernal so you can run your notebooks from the command line.

You can now use uv to run your new jupiter kernel.  For example:
```aiignore
uv run jupyter kernelspec list
```

## Default Dependency Groups
If you plan on using any of the options repeatedly, you can add then to the default-groups list in the pyproject.toml file.

###  Activating the Virtual Environment.

If you plan on working in the same virtual environment for a period of time, it can be more confienent to activate the venv
rather than typing *uv run* repeated. You can accomplish this with the following shell command:
```aiignore
source .venv/bin/activate.csh
```
Once this has been entered, the venv in the repository will be used for all subsaquent commands and the `uv run` 
prefix isn't needed.

When you are done working in the repository, the command
```aiignore
deactivate
```

will remove any default venv selection
### Using pytorch

If you plan on using pytorch, you need to configura your venv with the command:
```aiignore
uv sync --group=tensorflow
```
You may need to adjust versions and indexes depending on your exact configuration of CUDA, Python and tensorflow.

### Using tensorflow

If you plan on using pytorch, you need to configura your venv with the command:
```aiignore
uv sync --group=pytorch
```
You may need to adjust versions and indexes depending on your exact configuration of CUDA, Python and tensorflow.

## Running a script.

You can run a python script in the appropriate venv using the uv command:
```aiignore
uv run src/deriva_run.py
```
The script can be configured using the hydra configuration and experiment management too.
Please see the section on Experiment Management below.

## Running a notebook.

Although you can run a notebook interactively in the regular Jupiter environment, it is recommended that once your 
notebook has been debugged, that you run it from start to finish in the deriva-ml environment.

This process is streamlined by the command:
```
uv run deriva-ml-run-notebook notebook-file --host HOSTHAME --catalog CATALOG_ID --kernel <repository-name> 
 ```
 command, which uses papermill to run all of the cells in the notebook in sequence and then uploaded the resulting notebook into the catalog.

The notebook can be configured using the hydra configuration and experiment management too.
Please see the section on Experiment Management below.


# Configuration with Hydra & hydra‑zen
This template uses hydra‑zen to register configuration choices into Hydra’s config store at import time. The script consumes those configs using a typed function interface.

- Entrypoint: `src/deriva_run.py`
  - Registered app config name: `app_config`
  - Defaults (Hydra choices):
    - `deriva_ml: local`
    - `datasets: test1`
    - `assets: weights_1`
    - `model_config: default_model`

The `main` signature determines the config groups and their types:
```
@store(
    name="app_config",
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"deriva_ml": "local"},
        {"datasets": "test1"},
        {"assets": "weights_1"},
        {"model_config": "default_model"},
    ],
)
def main(
    deriva_ml: DerivaMLConfig,
    datasets: list[DatasetSpec],
    model_config: Any,
    assets: list[RID] | None = None,
    dry_run: bool = False,
) -> None:
    ...
```

Available groups and where they are defined:
- `deriva_ml`: `src/configs/deriva.py`
  - Example choices: `local`, `eye-ai`
- `datasets`: `src/configs/datasets.py`
  - Example choices: `test1`, `test2`, `test3`
- `assets`: `src/configs/assets.py`
  - Example choices: `weights_1`, `weights_2`
- `model_config`: `src/configs/simple_model.py`
  - Example choices: `default_model`, `epochs_20`, `epochs_100`

Tip: run `uv run src/script_template.py --help` to see the Hydra help and choices.

### Model configuration pattern (hydra‑zen)
This repo demonstrates the "build once, extend by instantiation" approach:
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
- Hydra produces a callable; the script later calls `model_config(execution=e)`.

### Running the script and overriding configs
- Use defaults:
```
uv run src/deriva_run.py
```
- Choose a different dataset or assets:
```
uv run src/script_template.py +datasets=test2 +assets=weights_2
```
- Choose a different model variant and/or override fields inline:
```
uv run src/script_template.py +model_config=epochs_100
uv run src/script_template.py +model_config.epochs=50
```
- Enable dry run (download inputs, skip writebacks):
```
uv run src/script_template.py +dry_run=true
```

### Running the notebook non‑interactively
Once debugged, prefer running notebooks in the same managed environment for provenance and reproducibility:
```
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb \
  --host www.eye-ai.org \
  --catalog 2 \
  --kernel <repository-name>
```
This executes all cells and uploads the resulting notebook to the catalog.

### Environment management with `uv`
- Create/sync env: `uv sync`
- Activate env (shell):
  - Bash/Zsh: `source .venv/bin/activate`
  - Fish: `source .venv/bin/activate.fish`
  - Csh/Tcsh: `source .venv/bin/activate.csh`
- Deactivate: `deactivate`
- Optional groups:
  - Jupyter: `uv sync --group=jupyter`
  - PyTorch: `uv sync --group=pytorch`
  - TensorFlow: `uv sync --group=tensorflow`

You can define `default-groups` in `pyproject.toml` to always install selected extras.

### Experiments (presets)
You can add curated experiment presets (sets of config choices) in `src/configs/experiments.py` using Hydra’s composition. For example:
```
from hydra_zen import store

# Example: choose dataset, assets, and a model variant in one name
experiment_store = store(group="experiments")
experiment_store(
    {"datasets": "test2", "assets": "weights_2", "model_config": "epochs_100"},
    name="high_epochs_alt_data",
)
```
Then run with:
```
uv run src/script_template.py +experiments=high_epochs_alt_data
```


## Experiment Management

DerivaML uses the hydra configuration framework to manage configurations of script, and notebooks and also to 
conduct different types of experiments.

Rather than hard coding values into the script or notebook, you can use the hydra configuration framework to 
specify values that can be changed at runtime.  We have integrated hydra-zen, which provides a simple pythonic way to create and configure ML models.

Documentation on hydra can be found [here.](https://hydra.cc/docs/intro/). Documentation on hydra-zen can be found here. 

A sample configuration script is found in src/configuration.py

## Recommended Workflow

Every model should live in its own repository that follows this template. 
The pyproject.toml file is set up to use uv, which makes it straight forward to create an manage Pyton environments.
It is recommended that you commit the uv.lock file that is created on first setup into your repo and that you update
your environment only through the uv command line.
The basic configuration for the environment should include deriva-ml and the domain specific modules.
These are included as default dependencies in the template.

Best practice is to commit any changes to the model prior to running it.  
This will maximize the ability of DerivaML to track what code was used to produce the model result.
It is recommended practice to use git branch and pull requests even if you are working on your own.
This way you will have better records of the changes made over time as you evolve your model.

During debugging, a *dry_run* option is available in the `Execution.create_execution` method.  
This option will download all of the input files associated with an execution, but will not create any Execution records,
and will not upload any resulting files.  
Once you are confident that your model/notebook is correct, the best practice is remove the dry-run option, create a new version tag and then run that model to completion,.

### Configuring Executions
Executions should be managed using [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/). Sample configuration files for hydra are provided in the configs module.

## Managing releases and version tags

In addition to commiting, it is advisable to tag the model as significant milestones and version as a release.
The template includes a bash script and a GitHub action that in combination that streamlines the process of creating release tags for a model.
DerivaML uses semantic versioning.

The script takes a single argument whose values can be patch, minor or major. E.G.
```aiignore
uv run bump-version major|minor|patch
```
The bump-version code will automatically use the commit log from any pull requests to generate release notes. 

## Getting the current version

DerivaML uses setuptools-scm to determine the current version of the model.  
THis gives you a dynamic version number that is updated automatically when you create a new release, or have commits on the latest release.
You can determine the current version of the model from the command line by entering:
```aiignore
uv run python -m setuptools_scm
```

## Updating Modules including DerivaML

The uv tool can be used to update specific packages in your application.
To update a package, such as DerivaML us the command:
```
 uv sync --upgrade-package deriva-ml
```
You can upgrade all of the packages in your application, however, you should proceed with caution, as upgrading to the latest version of pytorch or tensorflow can cause problems if you don't have the correct version of the driver installed.
One way around this is to pin to specific versions of these libraries when you add them to you pyproject.toml file using the `uv add` command.

Once the upgrade is complete, you will want to recommit your uv.lock file.


## Best Practices

You *SHOULD* use an established doc string format for your code. This will make it easier for others to understand your code. DerivaML uses the google docstring format. For more information on docstring formats, please see [here.](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

You *SHOULD* Use type hints wherever possible.

Even if you are working by yourself, you *SHOULD* work in a Git branch and do a pull request.  Rebase your branch regularly to keep it up to date with the main branch.

You *MUST* always run your code from a hydra-zen configuration file.  You should commit your code before running.
If you are running an experiment, do a bump-version before running.

No change is too small to properly track in GithHub and DerivaML.

You *SHOULD* use bump-version to create version tags prior to running your model.  DerivaML uses semantic versioning. So use minor version increments for new features, patch increments for bug fixes, and major versison increments for breaking changes.

As a general rule, you *SHOULD NOT* commit any data files.  These should be stored in DerivaML.

You *MUST* not commit a notebook with output cells.  This will make it difficult to track changes to the notebook.
If you install the nbstripout as described above, this will be taken care of automatically.

If you are using notebooks, try to make sure they *SHOULD* focused on a single task. 
As a rule of thumb, notebooks should be used for analysis and visualization, not for training models.
Having a lot of difference cells that you pick and choose from is a invitation to make mistakes.  
You *MUST* make sure that you can run the notebook from start to finish without any intervention and when you are satusfied, you can use deriva-ml-run-notebook to run the notebook in the deriva-ml environment.
Using this command will automatically upload the resulting notebook with output cells filled to the catalog.

DerivaML provides functions for managing generic ML workflows.  It was designed to be extended via inheritance to provide domain specific functionality.
If you are working on a domain specific model, you should consider creating a new module that inherits from the DerivaML class.
In this situation, you would instantiate your domain specific version of DerivaML in your script or notebook, not DerivaML directly.