"""
THis module is the main entry point for the model script.  Its job is to take the arguments provided by hydra and
create an execution configuration and run the model.

This script is intended to be used as a template for other model scripts.  You should modify it to reflect the actual
workflow to be created and call the domain specific version of DerivaML.
"""
from typing import Any

from deriva_ml import DerivaML, DerivaMLConfig, MLVocab, RID
from deriva_ml.dataset import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration
from hydra_zen.typing import Builds

def run_model(
    deriva_ml: DerivaMLConfig,
    execution_config: Builds[ExecutionConfiguration],
    model_config: Any,
    dry_run: bool = False,
) -> None:
    """
    Main entry point for this script.  Argument names must correspond to configuration groups in hydra configuration.
    Args:
        deriva_ml: A configuration for DerivaML with values corresponding to parameters DerivaML()
        datasets: A list of datasets to use in creating an ExecutionConfig
        model_config: Configuration for the ML model.  This is a callable that wraps the actual model code.
        assets:  A list of assets to be used in creating an ExecutionConfig. Typically will contain the files with the
            weights.
        dry_run: Optional dryrun parameter for Execution.  Other configuration arguments could be added here.

    Returns:

    """

    # Make a connection to the Deriva catalog.  You will need to change the class being used if you have a
    # derived catalog from DerivaML.  For example, in the case of an EyeAI catalog, you would use EyeAI instead of
    # DerivaML.
    ml_instance = DerivaML.instantiate(deriva_ml)

    # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
    # This call should be changed to reflect the actual types of workflow being run.
    ml_instance.add_term(
        MLVocab.workflow_type,
        "Template Model Script",
        description="Initial setup of Model Notebook",
    )
    workflow = ml_instance.create_workflow(
        name="demo-workflow",
        workflow_type="Template Model Script",
        description="A Model Template Workflow")

    # Create an execution instance.
    execution = ml_instance.create_execution(execution_config, workflow=workflow, dry_run=dry_run)

    with execution as e:
        # The model function has been partially configured, so we need to instantiate it with the execution object.
        # Note that model_config is a callable created by hydra-zen, not a dataclass.
        model_config(execution=e)
    print("Uploading outputs...")
    execution.upload_execution_outputs()