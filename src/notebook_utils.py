"""Notebook display utilities.

Helper functions for displaying DerivaML objects in Jupyter notebooks.
These are candidates for inclusion in the deriva-ml library.
"""

from IPython.display import display, Markdown


def display_dataset_list(datasets, ml, title: str = "**Input Datasets:**") -> None:
    """Display datasets as a markdown list with nested children.

    Args:
        datasets: List of Dataset objects to display.
        ml: DerivaML instance (used for building URLs).
        title: Markdown title to display above the list.
    """
    if not datasets:
        return

    display(Markdown(title))

    lines = []
    for ds in datasets:
        url = f"https://{ml.host_name}/chaise/record/#{ml.catalog_id}/deriva-ml:Dataset/RID={ds.dataset_rid}"
        version = str(ds.current_version) if ds.current_version else "n/a"
        types = ", ".join(ds.dataset_types) if ds.dataset_types else ""
        desc = ds.description or ""

        lines.append(f"- [{ds.dataset_rid}]({url}) v{version} [{types}]: {desc}")

        # Add nested children (direct children only)
        children = ds.list_dataset_children(recurse=False)
        for child in children:
            child_url = f"https://{ml.host_name}/chaise/record/#{ml.catalog_id}/deriva-ml:Dataset/RID={child.dataset_rid}"
            child_version = str(child.current_version) if child.current_version else "n/a"
            child_types = ", ".join(child.dataset_types) if child.dataset_types else ""
            child_desc = child.description or ""

            lines.append(
                f"  - [{child.dataset_rid}]({child_url}) v{child_version} [{child_types}]: {child_desc}"
            )

    display(Markdown("\n".join(lines)))


def display_experiment_summary(exp, ml) -> None:
    """Display a formatted summary of an experiment.

    Args:
        exp: Experiment object to display.
        ml: DerivaML instance (used for building URLs).
    """
    # Header with execution link
    display(Markdown(f"---\n### {exp.name} ([{exp.execution_rid}]({exp.get_chaise_url()}))"))

    # Description
    if exp.description:
        display(Markdown(f"**Description:** {exp.description}"))

    # Config choices
    if exp.config_choices:
        choices_str = ", ".join(f"`{k}={v}`" for k, v in sorted(exp.config_choices.items()))
        display(Markdown(f"**Configuration Choices:** {choices_str}"))

    # Model configuration (filter internal fields)
    model_cfg = {k: v for k, v in exp.model_config.items() if not k.startswith("_")}
    if model_cfg:
        display(Markdown("**Model Configuration:**"))
        config_lines = [f"- **{k}**: {v}" for k, v in sorted(model_cfg.items())]
        display(Markdown("\n".join(config_lines)))

    # Input datasets
    if exp.input_datasets:
        display_dataset_list(exp.input_datasets, ml)

    # Input assets
    if exp.input_assets:
        display(Markdown("**Input Assets:**"))
        asset_lines = [
            f"- [{asset.asset_rid}]({asset.get_chaise_url()}): {asset.filename}"
            for asset in exp.input_assets
        ]
        display(Markdown("\n".join(asset_lines)))
