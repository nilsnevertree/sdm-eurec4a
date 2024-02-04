"""Sphinx configuration."""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Source code dir relative to this file


project = "Using Super-Droplet Model and EUREC4A data to simulate rain evaporation"
author = "Nils Niebaum"
copyright = "2023, Nils Niebaum"
# Add your repository URL
github_url = (
    f"https://github.com/nilsnevertree/kalman-reconstruction-partially-observed-systems"
)
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ["source/templates"]