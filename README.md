# DerivaML Model Template

This repository provides a template for using DerivaML to develop a ML model.

Deriva-ML is a python library to simplify the process of creating and executing reproducible machine learning workflows
using a deriva catalog.

The code provenance aspects of DerivaML depend on a script or notebook being tracked in Git. 
This template assumes that GitHub is being used.

## Creating a new repository

This repository is set up as a template.  Its intended use is to create a new repository using the template and then customize it for your specific model.
To create a repository from the template, follow the instructions provided here [GitHub Template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

Templates for models set up as runnable python scripts, and Jupyter notebooks are provided.

## Project Management

This template uses uv as a project management tool.  As a prerequisite, you should install the uv tool into your execution environment.
Once installed you should use the uv command line to manage your dependencies and virtual environment.


## Getting the current version

You can determine the current version of the model from the command line by entering:
```aiignore
uv run python -m setuptools_scm
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
uv run deriva-ml-run-notebook notbook-file
 ```
 command, which uses papermill to substitute values into a parameters
cell in your notebook, and then runs every cell in sequence and uploaded the resulting notebook into the catalog.

Note that run-notebook assumes that the nboutput tool has been installed so as to ensure that cell outputs are not
commited to the repository. To install nboutput, you should type the command:
```aiignore
nboutput --install
```
You only need to do this once after which the command is installed as a git pre-commit hook and run every time you 
commit your notebook to the repository.

## Recommended Workflow

Every model should live in its own repository that follows this template. 
To enhance reproducibility, each repository should have its own venv.
The pyproject.toml file is set up to us uv, which makes it straight forward to create an manage Pyton enviroments.
It is recommended that you commit the uv.lock file that is created on first setup into your repo and that you update
your environment only through the uv command line.
The basic configuration for the environment should include deriva-ml and the domain specific modules.
These are included as default dependencies in the template.

Best practice is to commit any changes to the model prior to running it.  
This will maximize the ability of DerivaML to track what code was used to produce the model result.
We should assume that deriva-ml are installed as modules into the environment.  

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
./bump-version major
```
