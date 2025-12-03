"""
Define the possible execution configurations.

The datasets and assets that will be used for the execution are specified as part of the execution configuration and
will need to be defined in the datasets and assets configuration files.
"""
from hydra_zen import  store
from deriva_ml.execution import ExecutionConfiguration

execution_store = store(group="execution_config")
execution_store(ExecutionConfiguration,
                name="default_execution",
                description="My default execution configuration",
                hydra_defaults=["_self_" , {"/datasets": "test1"}, {"/assets": "weights_1"}]
                )
