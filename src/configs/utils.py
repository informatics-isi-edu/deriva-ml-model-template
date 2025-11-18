
def get_configuration(
        deriva_ml: DerivaMLConfig,
        datasets: DatasetConfigList,
        assets: list[RID],
        model_config: Any,
        dry_run: bool
):
    signature = inspect.signature(get_configuration)
    parameter_names = [param.name for param in signature.parameters.values()]
    vars = locals()
    return tuple([vars[name] for name in parameter_names])


def get_configuration(
     *args
):
    signature = inspect.signature(get_configuration)
    parameter_names = [param.name for param in signature.parameters.values()]
    vars = locals()
    return tuple([vars[name] for name in parameter_names])