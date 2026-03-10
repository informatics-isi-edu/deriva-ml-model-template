# DerivaML Model Template

This repository provides a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## What's Included

- Python-first configuration using hydra-zen (no YAML)
- CLI entry points via `deriva-ml-run` and `deriva-ml-run-notebook`
- An example model (CIFAR-10 CNN) with 7 configuration variants
- Experiment presets and named multirun configurations
- A ROC analysis notebook
- GitHub Actions for automated versioning

## Quick Links

- [Quick Start Guide](getting-started/quick-start.md) - Get up and running in minutes
- [Creating a New Model](getting-started/creating-models.md) - Step-by-step guide for adding models
- [Creating a New Notebook](getting-started/creating-notebooks.md) - Step-by-step guide for adding notebooks
- [Configuration Guide](configuration/overview.md) - Understanding hydra-zen configuration

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git and GitHub account
- Access to a DerivaML catalog — use an existing server or run one locally with [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker)

## Project Layout

```
.
├── pyproject.toml                  # Project metadata and dependencies
├── Experiments.md                  # Registry of defined experiments
├── CLAUDE.md                       # Claude Code project instructions
├── src/
│   ├── configs/                    # Hydra-zen configurations (Python, no YAML)
│   │   ├── base.py                 # Base DerivaModelConfig
│   │   ├── deriva.py               # Catalog connection settings
│   │   ├── datasets.py             # Dataset specifications
│   │   ├── assets.py               # Asset RID configurations
│   │   ├── workflow.py             # Workflow definitions
│   │   ├── cifar10_cnn.py          # Model variant configs
│   │   ├── experiments.py          # Experiment presets
│   │   ├── multiruns.py            # Named multirun configurations
│   │   ├── roc_analysis.py         # ROC analysis notebook config
│   │   └── dev/                    # Alternate catalog configs
│   ├── models/                     # Model implementations
│   │   └── cifar10_cnn.py          # CIFAR-10 CNN model
│   └── scripts/                    # Data loading scripts
│       └── load_cifar10.py         # CIFAR-10 dataset loader
├── notebooks/
│   └── roc_analysis.ipynb          # ROC curve analysis notebook
└── docs/                           # Documentation (auto-published)
```

## Related Resources

- [DerivaML Documentation](https://informatics-isi-edu.github.io/deriva-ml/)
- [DerivaML MCP Server](https://github.com/informatics-isi-edu/deriva-mcp) - AI assistant integration
- [Hydra-zen Documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/)
