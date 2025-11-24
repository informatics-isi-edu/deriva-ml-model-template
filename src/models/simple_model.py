"""
This module defines a simple model function that can be used as a template for a model script.  This should be replaced
with the proper model functions.

Additional parameters can be added to the function signature as needed.
"""
from deriva_ml.execution import Execution


def simple_model(learning_rate: float, epochs: int, execution: Execution) -> None:
    """A  simple model function.  This should be replaced with the proper top level model for the script.

    This is a very simple example of calling a model architecture with parameters. In an actual implementation,
    this would use actual model code from tensorflow, pytorch or some other framework.

    Args:
        learning_rate: Sample model parameter
        epochs:  Sample model parameter
        execution: DerivaML execution object that will contain datasets and assets.

    Returns:

    """
    weights = execution.asset_paths
    datasets = execution.datasets
    print(
        f"Training with learning rate: {learning_rate} and epochs: {epochs} and dataset"
    )
    print(weights)
    print(datasets)
