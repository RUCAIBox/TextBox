from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    'matplotlib>=3.1.3', 'torch>=1.6.0', 'numpy>=1.17.2', 'nltk>=3.4.5', 'pyyaml>=5.3.1', 'fast_bleu>=0.0.89',
    'py-rouge>=1.1', 'transformers>=4.5.1', 'tqdm>=4.42.1', 'sentencepiece>=0.1.95'
]

setup_requires = []

extras_require = {}

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

classifiers = [
    'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent',
    'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: 3.8'
]

with open("PYPI.md", "r") as f:
    long_description = f.read()

setup(
    name='textbox',
    version='0.2.1',
    description='A package for building text generation systems',
    url='https://github.com/RUCAIBox/TextBox',
    author='TextBoxTeam',
    author_email='rucaibox@163.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.2",
    packages=[package for package in find_packages() if package.startswith('textbox')],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False
)
