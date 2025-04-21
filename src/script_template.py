"""
This file is a template for how to set up a stand-alone script to execute a model.
"""
import logging

from deriva_ml import DerivaML, ExecutionConfiguration, DatasetSpec, MLVocab, Execution
from deriva.core import BaseCLI
from typing import Optional

# These should be set to be the RIDs of input datasets and assets that are downloaded prior to execution.
datasets = []
models = []

class DerivaDemoCLI(BaseCLI):
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)

        self.parser.add_argument("--catalog", default=1, metavar="<1>", help="Catalog number. Default: 1")
        self.parser.add_argument("--test", action="store_true", help="Use demo catalog.")
        self.parser.add_argument("--dry-run", action="store_true", help="Perform execution in dry-run mode.")

        self.execution: Optional[Execution] = None
        self.deriva_ml: Optional[DerivaML] = None
        self.logger = logging.getLogger(__name__)

    def main(self):
        """Parse arguments and set up execution environment."""
        args = self.parse_cli()
        hostname = args.host
        catalog_id = args.catalog

        self.deriva_ml = DerivaML(hostname, catalog_id)  # This should be changed to the domain specific class.
        print(f'Executing script {self.deriva_ml.executable_path} version: {self.deriva_ml.get_version()}')

        # Create a workflow instance for this specific version of the script.  Return an existing workflow if one is found.
        self.deriva_ml.add_term(MLVocab.workflow_type, "Demo Notebook", description="Initial setup of Model Notebook")
        workflow = self.deriva_ml.create_workflow('demo-workflow', 'Demo Notebook')

        # Create an execution instance that will work with the latest version of the input datasets.
        config = ExecutionConfiguration(
            datasets=[DatasetSpec(rid=dataset, version=self.deriva_ml.dataset_version(dataset)) for dataset in
                      datasets],
            assets=models,
            workflow=workflow,
            parameters='parameters..json'
        )

        self.execution = self.deriva_ml.create_execution(config, dry_run=args.dry_run)
        with self.execution as e:
            self.do_stuff(e)
        self.execution.upload_execution_outputs()

    def do_stuff(self, execution: Execution):

        print(f" Execution with parameters: {execution.parameters}")
        print(f" Execution with input assets: {[a.as_posix() for a in execution.asset_paths]}")
        print(f"Execution datasets: {execution.datasets}")


if __name__ == "__main__":
    cli = DerivaDemoCLI(description="Deriva ML Execution Script Demo",
                        epilog="")
    cli.main()
