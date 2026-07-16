"""Sphinx configuration."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.abspath(".."))  # Source code dir relative to this file


project = "SDM-EUREC4A"
author = "Nils Niebaum"
copyright = "2023, Nils Niebaum"
# Add your repository URL
github_url = f"https://github.com/nilsnevertree/sdm-eurec4a"
extensions = [
    # "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",  # Create neat summary tables
    # "sphinx_click",
    "myst_parser",
    # "sphinx.ext.coverage",
    # "sphinx_automodapi.automodapi",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
# html_theme = "furo"
html_theme = "sphinx_book_theme"
autosummary_generate = True  # Turn on sphinx.ext.autosummary

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
