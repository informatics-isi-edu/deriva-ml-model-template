# DerivaML Model Template

This repository provides a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## What's Included

- A basic project configuration using hydra-zen
- A simple Python script entrypoint with Hydra CLI
- An example model and configuration modules
- An equivalent Jupyter notebook template
- Sample parameter and environment setup
- GitHub Actions for automated versioning

## Quick Links

- [Quick Start Guide](getting-started/quick-start.md) - Get up and running in minutes
- [Creating a New Model](getting-started/creating-models.md) - Step-by-step guide for adding models
- [Creating a New Notebook](getting-started/creating-notebooks.md) - Step-by-step guide for adding notebooks
- [Configuration Guide](configuration/overview.md) - Understanding hydra-zen configuration
- [Coding Guidelines](reference/coding-guidelines.md) - Best practices and standards

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git and GitHub account
- Access to a Deriva catalog

## Project Layout

```
.
├─ pyproject.toml                  # Project metadata, dependencies
├─ src/
│  ├─ deriva_run.py                # Script entrypoint (Hydra main)
│  ├─ model_runner.py              # Helper for running models
│  ├─ models/
│  │  └─ simple_model.py           # Example model function
│  └─ configs/
│     ├─ deriva.py                 # DerivaML connection configs
│     ├─ datasets.py               # Dataset specifications
│     ├─ assets.py                 # Asset RID configurations
│     └─ experiments.py            # Experiment presets
├─ notebooks/
│  └─ notebook_template.ipynb      # Example notebook
└─ docs/                           # This documentation
```

## Related Resources

- [DerivaML Documentation](https://informatics-isi-edu.github.io/deriva-ml/)
- [DerivaML MCP Server](https://github.com/informatics-isi-edu/deriva-ml-mcp) - AI assistant integration
- [Hydra-zen Documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/)
