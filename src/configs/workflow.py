"""Workflow configurations.

Configuration Group: ``workflow``
---------------------------------
A :class:`~deriva_ml.execution.Workflow` describes the computational pipeline
behind a run — its name, a human-readable description, and one or more
workflow types — and automatically captures Git provenance (repository URL,
commit checksum, version) when an execution starts. The runner selects one
with ``workflow=<name>``.

REQUIRED: a config named ``default_workflow`` must be registered in this group.
It is used when no ``workflow`` override is supplied.

``workflow_type`` may be a single string or a list of strings for multi-faceted
categorization (e.g. ``["Training", "Evaluation"]``).

Usage:
    # Use the default workflow
    uv run deriva-ml-run

    # Select a specific workflow
    uv run deriva-ml-run workflow=<name>
"""

from hydra_zen import store, builds

from deriva_ml.execution import Workflow

# ---------------------------------------------------------------------------
# Workflow configurations
# ---------------------------------------------------------------------------

# A generic default so the skeleton resolves and runs. Replace the name,
# type, and description with the specifics of your pipeline.
DefaultWorkflow = builds(
    Workflow,
    name="Model Run",
    workflow_type="Training",
    description="Placeholder workflow for the template — replace with your pipeline's details.",
    populate_full_signature=True,
)

# ---------------------------------------------------------------------------
# Register with the hydra-zen store.
# ---------------------------------------------------------------------------
workflow_store = store(group="workflow")

# REQUIRED: ``default_workflow`` is used when no workflow is specified.
workflow_store(DefaultWorkflow, name="default_workflow")

# -----------------------------------------------------------------------------
# Add additional workflows below. Example:
# -----------------------------------------------------------------------------
#
#   MyWorkflow = builds(
#       Workflow,
#       name="<Pipeline Name>",
#       workflow_type=["<Type A>", "<Type B>"],
#       description="<what the pipeline does>",
#       populate_full_signature=True,
#   )
#   workflow_store(MyWorkflow, name="<your_workflow>")
