from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['matplotlib>=3.1.3', 'torch>=1.2.0', 'numpy>=1.17.2',
                    'pyyaml>=5.3.1', 'fast_bleu>=0.0.86', 'rouge>=1.0.0']

setup_requires = []

extras_require = {}

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='recbole',
    version=
    '0.1.0',  # please remember to edit recbole/__init__.py in response, once updating the version
    description='A package for building recommender systems',
    url='https://github.com/RUCAIBox/RecBole',
    author='RecBoleTeam',
    author_email='ContactRecBoleTeam',
    packages=[
        package for package in find_packages()
        if package.startswith('recbole')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False)
