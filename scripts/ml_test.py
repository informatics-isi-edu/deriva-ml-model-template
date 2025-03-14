from deriva_ml import DerivaML, ExecutionConfiguration, DatasetSpec, RID
datasets = []
models = []
hostname = 'dev.eye-ai.org'
catalog_id = 'eye-ai'


def do_stuff():
    pass


def main():
    deriva_ml = DerivaML(hostname, catalog_id)
    my_url = DerivaML.github_url()
    ml_instance = DerivaML(hostname, catalog_id)
    ml_instance.lookup_workflow(my_url)
    config = ExecutionConfiguration(
        datasets=[DatasetSpec(rid=dataset,
                              version=ml_instance.dataset_version(dataset)) for dataset in datasets],
        assets=models,
        workflow=ml_instance.lookup_workflow(my_url)
    )
    execution = ml_instance.create_execution(config)
    with execution as e:
        do_stuff()
    execution.upload_execution_outputs()


if __name__ == "__main__":
    main()
