"""Analysis notebook configuration (scaffold).

This module shows how to register a Hydra-zen config for an *analysis
notebook* — a notebook that runs under DerivaML provenance and consumes asset
RIDs (e.g. prediction-probability files) rather than datasets.

By DerivaML convention, notebook ``X.ipynb`` uses the config registered as
``notebook_config("X", ...)``. ``run_notebook()`` derives the config name from
the calling notebook's filename, so the explicit string is only needed when the
notebook filename and config name diverge.

The template ships with this single generic ``analysis`` config as an example.
Rename it (and add more) to match the notebooks you build under ``notebooks/``.

Usage:
    In a notebook (``notebooks/analysis.ipynb``):

        from deriva_ml.execution import run_notebook

        ml, execution, config = run_notebook()
        # Config name auto-derived from the notebook filename.
        # - config.assets: the selected asset RID list
        # - config.show_detail: example custom parameter

    From the command line:

        deriva-ml-run-notebook notebooks/analysis.ipynb \\
            --host <hostname> --catalog <id> \\
            assets=<your_assets>

Configuration Groups consumed:
    - deriva_ml: connection settings (default_deriva, ...)
    - assets: asset RID lists (default_asset, no_assets, ...)
"""

from dataclasses import dataclass

from deriva_ml.execution import BaseConfig, notebook_config


@dataclass
class AnalysisConfig(BaseConfig):
    """Configuration for an analysis notebook.

    Add your own notebook parameters as fields here — they become Hydra
    overridable (``deriva-ml-run-notebook ... show_detail=false``).

    Attributes:
        show_detail: Example boolean parameter. Replace with parameters your
            notebook actually needs.
    """

    show_detail: bool = True


# Generic default analysis config. ``no_assets``/``no_datasets`` keep the
# example runnable on a fresh clone; point ``assets`` at a real asset group
# once you have one (see ``configs/assets.py``).
notebook_config(
    "analysis",
    config_class=AnalysisConfig,
    defaults={"assets": "no_assets", "datasets": "no_datasets"},
    description="Analysis of the selected asset group.",
)

# -----------------------------------------------------------------------------
# Register one config per analysis notebook you add. Example:
# -----------------------------------------------------------------------------
#
#   notebook_config(
#       "<your_notebook>",
#       config_class=AnalysisConfig,
#       defaults={"assets": "<your_assets>", "datasets": "no_datasets"},
#       description="<what this analysis does>",
#   )
