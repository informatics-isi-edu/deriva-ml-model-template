"""
This file is a template for how to set up a stand-alone script to execute a model.
"""
from deriva_ml import DerivaML, ExecutionConfiguration, DatasetSpec, MLVocab, Execution
from deriva.core import BaseCLI
from typing import Optional
datasets = []
models = []


class DerivaDemoCLI(BaseCLI):
    """Main class to part command line arguments and call model"""

    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)

        self.parser.add_argument("--catalog", default=1, metavar="<1>", help="Catalog number. Default: 1")
        self.execution: Optional[Execution] = None
        self.deriva_ml: Optional[DerivaML] = None

    def main(self):
        """Parse arguments and set up execution environment."""
        args = self.parser.parse_cli()
        hostname = args.hostname
        catalog_id = args.catalog

        self.deriva_ml = DerivaML(hostname, catalog_id)  # This should be changed to the domain specific class.

        self.deriva_ml.add_term(MLVocab.workflow_type, "Demo Notebook", description="Initial setup of Model Notebook")
        workflow_rid = self.deriva_ml.create_workflow('demo-workflow', 'Demo Notebook')

        config = ExecutionConfiguration(
            datasets=[DatasetSpec(rid=dataset, version=self.deriva_ml.dataset_version(dataset)) for dataset in datasets],
            assets=models,
            workflow=workflow_rid
        )
        self.execution = self.deriva_ml.create_execution(config)
        with self.execution as _e:
            self.do_stuff()
        self.execution.upload_execution_outputs()

    def do_stuff(self):
        """Put your model here"""
        pass


if __name__ == "__main__":
    cli = DerivaDemoCLI(description="Deriva ML Execution Script Demo",
                        epilog="")
    cli.main()
