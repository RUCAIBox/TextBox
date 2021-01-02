from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['matplotlib>=3.1.3', 'torch>=1.2.0', 'numpy>=1.17.2', 'nltk>=3.4.5',
                    'pyyaml>=5.3.1', 'fast_bleu>=0.0.86', 'rouge>=1.0.0','transformers>=4.0.1']

setup_requires = []

extras_require = {}

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='textbox',
    version=
    '0.0.1',  # please remember to edit textbox/__init__.py in response, once updating the version
    description='A package for building text generation systems',
    url='https://github.com/RUCAIBox/TextBox',
    author='TextBoxTeam',
    author_email='ContactTextBoxTeam',
    packages=[
        package for package in find_packages()
        if package.startswith('textbox')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False)
