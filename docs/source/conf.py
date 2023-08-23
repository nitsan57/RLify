# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../../'))


project = 'RLify'
copyright = '2023, Nitsan Levy'
author = 'Nitsan Levy'
release = '0.0.1'

# Import the theme 
import sphinx_rtd_theme

# Configure theme
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
autodoc_mock_imports = ['external_modules', 'gym'] 

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'private-members': True,
    'special-members': True,
    # 'inherited-members': True,
    # 'show-inheritance': True,
    # 'ignore-module-all': True
}


autodoc_recursive = True
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme', 'sphinx.ext.napoleon', 'sphinx_autodoc_typehints', "myst_parser"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autodoc_typehints = "description"
# add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']


