"""
This module defines a simple model function that can be used as a template for a model script.
This should be replaced with the proper model functions.

In addition to the model function parameters, the function takes an optional execution parameter.
The calling function will instantiate this function with the execution object that will contain the datasets and assets
as well as information about the execution environment, such as the working directory.

Additional parameters can be added to the function signature as needed.

"""
from deriva_ml.execution import Execution
from deriva_ml import MLAsset, ExecAssetType, DerivaML

def simple_model(learning_rate: float, epochs: int,
                 ml_instance: DerivaML,
                 execution: Execution | None = None) -> None:
    """A  simple model function.  This should be replaced with the proper top level model for the script.

    This is a very simple example of calling a model architecture with parameters. In an actual implementation,
    this would use actual model code from tensorflow, pytorch or some other framework.

    Args:
        learning_rate: Sample model parameter
        epochs:  Sample model parameter
        ml_instance: DerivaML instance that will be used to access paths.
        execution: DerivaML execution object that will contain datasets and assets.

    Returns:

    """
    print(f"Executing on: {ml_instance.host_name} in catalog {ml_instance.catalog_id}")
    weights = execution.asset_paths
    datasets = execution.datasets
    print(
        f"Training with learning rate: {learning_rate} and epochs: {epochs} and dataset"
    )

    # Assets in the execution object are stored in a dictionary with table names as keys and lists of
    # AssetFilePath objects as values.
    for table, assets in weights.items():
        print(f"Table: {table} Assets")
        for asset in assets:
            print(f"  {asset}")

    print("Datasets")
    print(datasets)
    output_file = execution.asset_file_path(MLAsset.execution_asset, "output.txt", ExecAssetType.output_file)

    with output_file.open("w") as f:
        f.write("This is a sample output file.")


