from deriva_ml import DerivaML, ExecutionConfiguration, DatasetSpec, MLVocab
from deriva_ml.demo_catalog import create_demo_catalog
from deriva.core import BaseCLI, KeyValuePairArgs, format_credential, format_exception, urlparse

import sys
datasets = []
models = []
hostname = 'dev.eye-ai.org'
catalog_id = 'eye-ai'


def do_stuff():
    pass


def main(hostname, catalog_id):
    deriva_ml = DerivaML(hostname, catalog_id)
    deriva_ml.add_term(MLVocab.workflow_type, "Demo Notebook", description="Initial setup of Model Notebook")
    workflow_rid = deriva_ml.create_workflow('demo-workflow', 'Demo Notebook')
    config = ExecutionConfiguration(
        datasets=[DatasetSpec(rid=dataset, version=deriva_ml.dataset_version(dataset)) for dataset in datasets],
        assets=models,
        workflow=workflow_rid
    )
    execution = deriva_ml.create_execution(config)
    with execution as e:
        do_stuff()
    execution.upload_execution_outputs()

class DerivaDemoCLI(BaseCLI):
    def __init__(self, description, epilog, **kwargs):
        BaseCLI.__init__(self, description, epilog, **kwargs)
        self.parser.add_argument("--catalog", default=1, metavar="<1>", help="Catalog number. Default: 1")

    def main(self):
        try:
            args = self.parse_cli()
            main(hostname=args.hostname, catalog_id=args.catalog)
        except ValueError as e:
            sys.stderr.write(str(e))
            return 2
        if not args.quiet:
            sys.stderr.write("\n")
        return 0


if __name__ == "__main__":
    cli = DerivaDemoCLI(description="Deriva ML Execution Script Demo",
                        epilog="")
    cli.main()
