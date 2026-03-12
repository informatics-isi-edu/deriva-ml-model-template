# DerivaML Model Template

This repository provides a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## Documentation

**[View Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)**

Quick links:
- [Quick Start Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/quick-start/) - Get up and running in minutes
- [Environment Setup](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/environment-setup/) - Detailed setup instructions
- [Creating a New Model](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-models/) - Step-by-step guide for adding models
- [Creating a New Notebook](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-notebooks/) - Step-by-step guide for adding notebooks
- [Configuration Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/configuration/overview/) - Understanding hydra-zen configuration
- [Coding Guidelines](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/coding-guidelines/) - Best practices and standards

## What's Included

- Python-first configuration using hydra-zen (no YAML)
- CLI entry points via `deriva-ml-run` and `deriva-ml-run-notebook`
- An example model (CIFAR-10 CNN) with 7 configuration variants
- Experiment presets and named multirun configurations
- A ROC analysis notebook with hydra-zen configuration
- GitHub Actions for automated versioning and documentation

## Quick Start

> **Note:** Running models requires access to a DerivaML catalog. If you don't have access to an existing Deriva server, you can run one locally using [deriva-docker](https://github.com/informatics-isi-edu/deriva-docker).

### 1. Create Your Repository

Use this template to create a new repository: [Creating a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

### 2. Enable GitHub Pages

After creating your repository from this template, enable GitHub Pages for automatic documentation deployment:

1. Go to your repository **Settings > Pages**
2. Under "Build and deployment", set **Source** to **"GitHub Actions"**
3. Save

The documentation workflow will automatically deploy on each push to main.

### 3. Initialize Environment

```bash
# Create environment and install dependencies
uv sync

# For notebook support
uv sync --group=jupyter
uv run nbstripout --install
uv run deriva-ml-install-kernel
```

### 4. Set Up Claude Code (Optional)

If using [Claude Code](https://claude.ai/code), install the DerivaML MCP server and skills plugin for catalog tools and guided workflows.

**Install the MCP server** (provides catalog tools, dataset management, execution tracking):

```bash
# Using Docker (recommended)
docker pull ghcr.io/informatics-isi-edu/deriva-mcp:latest
```

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "deriva": {
      "type": "stdio",
      "command": "/bin/sh",
      "args": [
        "-c",
        "docker run -i --rm --add-host localhost:host-gateway -e HOME=$HOME -v $HOME/.deriva:$HOME/.deriva:ro -v $HOME/.bdbag:$HOME/.bdbag -v $HOME/.deriva-ml:$HOME/.deriva-ml ghcr.io/informatics-isi-edu/deriva-mcp:latest"
      ]
    }
  }
}
```

**Install the skills plugin** (provides workflow guidance and auto-invoked best practices):

```
/plugin marketplace add informatics-isi-edu/deriva-mcp
/plugin install deriva
```

**Update the skills plugin** when a new version is released:

```
/plugin install deriva
```

To check if all DerivaML components are up to date (skills, deriva-ml library, MCP server):

```
/deriva:check-versions
```

See the [deriva-mcp README](https://github.com/informatics-isi-edu/deriva-mcp) for full setup options including HTTP transport, localhost configuration, and native installs.

### 5. Authenticate

```bash
uv run deriva-globus-auth-utils login --host <hostname>
```

### 6. Run

```bash
# Run the example model with defaults
uv run deriva-ml-run

# Dry run (no catalog writes)
uv run deriva-ml-run dry_run=true

# Use an experiment preset
uv run deriva-ml-run +experiment=cifar10_quick

# Named multirun
uv run deriva-ml-run +multirun=quick_vs_extended

# Show available configs
uv run deriva-ml-run --info

# Run a notebook
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb

# Override host/catalog from command line
uv run deriva-ml-run --host localhost --catalog 45 +experiment=cifar10_quick
```

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

## Versioning

Create version tags before significant runs (DerivaML records the git commit for provenance):

```bash
uv run bump-version patch   # Bug fixes
uv run bump-version minor   # New features
uv run bump-version major   # Breaking changes
```

## CIFAR-10 Example

The template includes a complete CIFAR-10 CNN example. See the [CIFAR-10 documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/cifar10-example/) for details.

```bash
# Load data into catalog
uv run load-cifar10 --host <hostname> --catalog_id <id> --num_images 500

# Train the model
uv run deriva-ml-run +experiment=cifar10_quick

# Run a learning rate sweep
uv run deriva-ml-run +multirun=lr_sweep

# Analyze results with ROC curves
uv run deriva-ml-run-notebook notebooks/roc_analysis.ipynb
```

### Dataset Types

The CIFAR-10 example includes multiple dataset configurations:

| Type | Use Case |
|------|----------|
| `cifar10_small_labeled_split` | Quick experiments with evaluation (recommended) |
| `cifar10_labeled_split` | Full experiments with evaluation |
| `cifar10_small_split` | Quick training without evaluation |
| `cifar10_split` | Full training without evaluation |

**Important:** For ROC analysis or accuracy metrics, use the **labeled** datasets. The unlabeled datasets have test images without ground truth labels.

## Using Claude Code with DerivaML

With the [DerivaML MCP server and skills plugin](https://github.com/informatics-isi-edu/deriva-mcp) (see step 4), you can interact with catalogs through natural language and get guided workflows for common tasks. Skills auto-trigger based on context, or you can invoke them directly with `/deriva:<skill-name>` (e.g., `/deriva:create-dataset`, `/deriva:run-experiment`).

## Further Reading

- [Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)
- [DerivaML Library](https://informatics-isi-edu.github.io/deriva-ml/) - Core library documentation
- [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) - Configuration framework
