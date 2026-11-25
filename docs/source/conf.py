# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RFantibody'
copyright = '2025, Institute for Protein Design, University of Washington'
author = 'Nathaniel Bennett, Joseph Watson, Institute for Protein Design, Rosetta Commons'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_nb",  # also loads myst_parser and handles .md/.ipynb
              "sphinx_copybutton",
              ]

templates_path = ['_templates']
exclude_patterns = ["readme.md", "readmelink.md", "readme_link.rst"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static', '../../models/rfd3/.assets']

html_theme_options = {
    "sidebar_hide_name":False,
    #"announcement": "<em>THIS DOCUMENTATION IS CURRENTLY UNDER CONSTRUCTION</em>",
    "light_css_variables": {
        "color-brand-primary": "#F68A33", # Rosetta Teal
        "color-brand-content": "#37939B", # Rosetta Orange
        #"color-admonition-background": "#CCE8E8", # Rosetta light orange
        "font-stack": "Open Sans, sans-serif",
        "font-stack--headings": "Open Sans, sans-serif",
        "color-background-hover": "#DCE8E8ff",
        "color-announcement-background" : "#F68A33dd",
        "color-announcement-text": "#070707",
        "color-brand-visited": "#37939B",
        },
    "dark_css_variables": {
        "color-brand-primary": "#37939B", # Rosetta teal
        "color-brand-content": "#F68A33", # Rosetta orange
        #"color-admonition-background": "#20565B", # Rosetta light orange
        "font-stack": "Open Sans, sans-serif",
        "font-stack--headings": "Open Sans, sans-serif",
        "color-brand-visited": "#37939B",
        }
    }

# .md and .ipynb suffixes are registered by myst_nb itself
source_suffix = {
    ".rst": "restructuredtext",
    }

# -- Notebook options (myst-nb) ----------------------------------------------
# Never execute notebooks at build time: the example notebook needs a GPU,
# model weights, and Google Colab. Readers run it via the "Open in Colab" badge.
nb_execution_mode = "off"

# Generate GitHub-style anchors for headings (levels 1-5) so in-page
# links like [Setup](#setup) resolve, e.g. the tutorial table of contents.
myst_heading_anchors = 5

html_js_files = [
    ('https://scripts.simpleanalyticscdn.com/latest.js', {'async': 'async', 'defer': 'defer'}),
]
suppress_warnings = ["myst.xref_missing"]
