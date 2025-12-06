from hydra_zen import store, builds

from deriva_ml.execution import Workflow

# Register the base config as the default model.
WorkflowConf = builds(Workflow,
                  name="Model Template Workflow",
                  workflow_type="Template Model Script",
                  description="A Model Template Workflow",
                  populate_full_signature=True)

model_store = store(group="workflow")
model_store(WorkflowConf, name="default_workflow")

