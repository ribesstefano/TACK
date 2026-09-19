"""MkDocs build hooks.

Copies the tutorial notebook from ``notebooks/`` into the docs tree so
mkdocs-jupyter can render it, without duplicating the notebook itself —
``notebooks/ensemble_predictor_tutorial.ipynb`` stays the single source of
truth. The copy under ``docs/tutorial/`` is regenerated on every build and
is gitignored.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = ROOT / "notebooks" / "ensemble_predictor_tutorial.ipynb"
DEST_NOTEBOOK = ROOT / "docs" / "tutorial" / "ensemble_predictor_tutorial.ipynb"


def on_pre_build(config, **kwargs):
    if not SOURCE_NOTEBOOK.exists():
        raise FileNotFoundError(
            f"Tutorial notebook not found at {SOURCE_NOTEBOOK}; "
            "docs/hooks.py expects it to exist before building the site."
        )
    DEST_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_NOTEBOOK, DEST_NOTEBOOK)
