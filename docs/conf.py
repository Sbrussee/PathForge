import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "PathBench"
copyright = "2025, Siemen Brussee"
author = "Siemen Brussee"
release = "2.0"
version = "2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "PathBench 2.0"
html_logo = None

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "lazyslide",
    "wsidata",
    "timm",
    "geopandas",
    "anndata",
    "torchmil",
    "torchmetrics",
    "torchsurv",
    "pycox",
    "torch_geometric",
    "huggingface_hub",
    "typer",
    "reportlab",
    "shapely",
    "optuna",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_use_rtype = True

autosummary_generate = True

# Keep docs builds self-contained in test and CI environments where outbound
# inventory fetches may be slow or unavailable.
intersphinx_mapping = {}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
