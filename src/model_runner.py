"""
THis module is the main entry point for the model script.  Its job is to take the arguments provided by hydra and
create a DerivaML execution environment and then and run the model.

This script is intended to be used as a template for other model scripts.  You should modify it to reflect the actual
workflow to be created and call the domain specific version of DerivaML.
"""
import logging
from typing import Any

from deriva_ml import DerivaML, DerivaMLConfig, MLVocab, RID
from deriva_ml.dataset import DatasetSpec
from deriva_ml.execution import ExecutionConfiguration, Workflow


def run_model(
    deriva_ml: DerivaMLConfig,
    datasets: list[DatasetSpec],
    assets: list[RID],
    description: str,
    workflow: Workflow,
    model_config: Any,
    dry_run: bool = False,
) -> None:
    """
    Main entry point for this script.  Argument names must correspond to configuration groups in hydra configuration.
    Args:
        deriva_ml: A configuration for DerivaML with values corresponding to parameters DerivaML()
        datasets: A list of datasets to use in creating an ExecutionConfig
        assets:  A list of assets to be used in creating an ExecutionConfig. Typically will contain the files with the
            weights.
        model_config: Configuration for the ML model.  This is a callable that wraps the actual model code.
        description: A description of the execution.
        workflow: A workflow to associate with the execution.
        dry_run: Optional dryrun parameter for Execution.  Other configuration arguments could be added here.

    Returns:

    """

    # Hydra wants to set up logging...lets get rid of it.
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Make a connection to the Deriva catalog.  You will need to change the class being used if you have a
    # derived catalog from DerivaML.  For example, in the case of an EyeAI catalog, you would use EyeAI instead of
    # DerivaML.
    ml_instance = DerivaML.instantiate(deriva_ml)

    # Create an execution instance.
    execution_config = ExecutionConfiguration(
        datasets=datasets, assets=assets, description=description
    )
    execution = ml_instance.create_execution(
        execution_config, workflow=workflow, dry_run=dry_run
    )

    with execution as e:
        # The model function has been partially configured, so we need to instantiate it with the execution object.
        # Note that model_config is a callable created by hydra-zen, not a dataclass.
        model_config(execution=e)
    _assets = execution.upload_execution_outputs()

