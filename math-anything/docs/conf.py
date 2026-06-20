"""Sphinx configuration for math-anything."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "math-anything"
copyright = "2026, Math Anything Contributors"
author = "Math Anything Contributors"
release = "3.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
