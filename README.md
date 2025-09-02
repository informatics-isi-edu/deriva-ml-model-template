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

Templates for models set up as runnable python scripts, and Jupyter notebooks are provided along with a sample parameter file.

## Project Management

This template uses `uv` as a project management tool.  As a prerequisite, you should install the *uv* tool into your execution environment.
Once installed you should use the uv command line to manage your dependencies and virtual environment.

Instructions on how to install UV can be found [here.](https://docs.astral.sh/uv/)
## Authenticating

You must be logged into Globus before you can access data in the catalog.
You can do this by using the following command:

```
deriva-globus-auth-utils login --host www.eye-ai.org
```
## Initializing Your Repository

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
uv run install_kernel
```

These commands install a Git pre-commit hook to strip output from your notebook prior to a commit
and create a jupyter kernal so you can run your notebooks from the command line.

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

You can now use uv to run your new jupiter kernel.  For example:
```aiignore
uv run jupyter kernelspec list
```
## Running a script.

You can run a python script in the appropriate venv using the uv command:
```aiignore
uv run src/script_template.py
```

## Running a notebook.

Although you can run a notebook interactively in the regular Jupiter environment, it is recommended that once your 
notebook has been debugged, that you run it from start to finish in the deriva-ml environment.

This process is streamlined by the command:
```
uv run deriva-ml-run-notebook notebook-file --host HOSTHAME --catalog CATALOG_ID [--file PARAMETER_FILE]
 ```
 command, which uses papermill to substitute values into a parameters
cell in your notebook, and then runs every cell in sequence and uploaded the resulting notebook into the catalog.


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
