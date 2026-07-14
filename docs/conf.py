import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

sys.path.insert(0, os.path.abspath("../src"))

project = "PathForge"
copyright = "2026, Siemen Brussee"
author = "Siemen Brussee"
try:
    release = package_version("pathforge")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "PathForge"
html_logo = None

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Keep every callable's complete typed signature visible and repeat type
# information alongside parameter descriptions. This makes inherited and
# otherwise sparsely documented callables usable from the generated reference.
autodoc_class_signature = "separated"
autodoc_preserve_defaults = True
autodoc_typehints = "both"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"

autodoc_mock_imports = [
    "lazyslide",
    "wsidata",
    "timm",
    "geopandas",
    "anndata",
    "spatialdata",
    "dask",
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
napoleon_use_ivar = True

# Keep docs builds self-contained in test and CI environments where outbound
# inventory fetches may be slow or unavailable.
intersphinx_mapping = {}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]
