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

## Getting the current version

You can determine the current version of the model from the command line by entering:
```aiignore
python -m setuptools_scm
```
There is also a DerivaML method that will return the current version of the executable script.
## Managing releases and version tags

The template includes a bash script that streamlines the process of creating release tags for a model.
DerivaML uses semantic versioning.

The script takes a single argument whose values can be patch, minor or major. E.G.
```aiignore
./bump-version major
```

### Installing the GitHub CLI

The script bump-version.sh will create a new release tag in GitHub.  This script requires the 
GitHUB CLI be installed. 

See [https://cli.github.com](https://cli.github.com) for instructions on how to install and configure the CLI.
