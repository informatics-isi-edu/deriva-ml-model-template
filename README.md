# DerivaML Model Template

This repository provides a template for creating ML models integrated with DerivaML, a Python library for reproducible ML workflows backed by a Deriva catalog. It captures code provenance, configuration, and outputs for reproducibility.

## Documentation

**📖 [View Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)**

Quick links:
- [Quick Start Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/quick-start/) - Get up and running in minutes
- [Environment Setup](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/environment-setup/) - Detailed setup instructions
- [Creating a New Model](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-models/) - Step-by-step guide for adding models
- [Creating a New Notebook](https://informatics-isi-edu.github.io/deriva-ml-model-template/getting-started/creating-notebooks/) - Step-by-step guide for adding notebooks
- [Configuration Guide](https://informatics-isi-edu.github.io/deriva-ml-model-template/configuration/overview/) - Understanding hydra-zen configuration
- [Coding Guidelines](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/coding-guidelines/) - Best practices and standards

## What's Included

- A basic project configuration using hydra-zen
- A simple Python script entrypoint with Hydra CLI
- An example model (CIFAR-10 CNN) with configuration variants
- A Jupyter notebook template with simplified initialization
- GitHub Actions for automated versioning and documentation

## Quick Start

### 1. Create Your Repository

Use this template to create a new repository: [Creating a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)

### 2. Enable GitHub Pages

After creating your repository from this template, enable GitHub Pages for automatic documentation deployment:

1. Go to your repository **Settings → Pages**
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

If using [Claude Code](https://claude.ai/code), install the DerivaML MCP server and skills plugin.

**Install the MCP server** (provides catalog tools, dataset management, execution tracking):

```bash
# Using Docker (recommended)
docker pull ghcr.io/informatics-isi-edu/deriva-mcp:latest
```

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "deriva-ml": {
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

**Install the skills plugin** (provides workflow guidance, auto-invoked best practices):

```
/plugin marketplace add informatics-isi-edu/deriva-mcp
/plugin install deriva
```

See the [deriva-mcp README](https://github.com/informatics-isi-edu/deriva-mcp) for full setup options including HTTP transport, localhost configuration, and native installs.

### 5. Authenticate

```bash
uv run deriva-globus-auth-utils login --host www.eye-ai.org
```

### 6. Run

```bash
# Run the example model
uv run src/deriva_run.py

# Dry run (no catalog writes)
uv run src/deriva_run.py dry_run=true

# Run a notebook
uv run deriva-ml-run-notebook notebooks/notebook_template.ipynb \
  --host www.eye-ai.org --catalog 2 --kernel <repo-name>
```

## Project Layout

```
.
├─ pyproject.toml                  # Project metadata and dependencies
├─ src/
│  ├─ deriva_run.py                # Script entrypoint (Hydra main)
│  ├─ model_runner.py              # Model execution helper
│  ├─ models/                      # Model implementations
│  │  ├─ simple_model.py
│  │  └─ cifar10_cnn.py
│  └─ configs/                     # Hydra-zen configurations
│     ├─ deriva.py                 # Connection settings
│     ├─ datasets.py               # Dataset specifications
│     ├─ assets.py                 # Asset configurations
│     └─ experiments.py            # Experiment presets
├─ notebooks/
│  └─ notebook_template.ipynb      # Example notebook
└─ docs/                           # Documentation (auto-published)
```

## Versioning

Create version tags before significant runs:

```bash
uv run bump-version patch   # Bug fixes
uv run bump-version minor   # New features
uv run bump-version major   # Breaking changes
```

## CIFAR-10 Example

The template includes a complete CIFAR-10 CNN example. See the [CIFAR-10 documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/reference/cifar10-example/) for details.

```bash
# Load data into catalog
uv run load-cifar10 --host dev.eye-ai.org --catalog_id 5 --num_images 500

# Train the model
uv run deriva-ml-run model_config=cifar10_default

# Run an experiment with evaluation
uv run deriva-ml-run +experiment=cifar10_quick
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

With the [DerivaML MCP server and skills plugin](https://github.com/informatics-isi-edu/deriva-mcp) (see step 4), you can interact with catalogs through natural language and get guided workflows for common tasks. Use `/deriva:<skill-name>` to invoke skills like `/deriva:create-dataset`, `/deriva:run-experiment`, etc.

## Further Reading

- [Full Documentation](https://informatics-isi-edu.github.io/deriva-ml-model-template/)
- [DerivaML Library](https://informatics-isi-edu.github.io/deriva-ml/) - Core library documentation
- [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) - Configuration framework
