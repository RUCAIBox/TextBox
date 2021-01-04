from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['matplotlib>=3.1.3', 'torch>=1.6.0', 'numpy>=1.17.2', 'nltk>=3.4.5',
                    'pyyaml>=5.3.1', 'fast_bleu>=0.0.86', 'rouge>=1.0.0','transformers>=4.0.1']

setup_requires = []

extras_require = {}

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

classifiers = ['License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']

long_description = '# TextBox\n'\
                   'TextBox is developed based on Python and PyTorch for reproducing and developing text generation algorithms in a unified, comprehensive and efficient framework for research purpose. Our library includes 16 text generation algorithms, covering two major tasks:\n'\
                   '\n+ Unconditional (input-free) Generation\n'\
                   '+ Sequence to Sequence (Seq2Seq) Generation\n'\
                   '\nWe provide the support for 6 benchmark text generation datasets. A user can apply our library to process the original data copy, or simply download the processed datasets by our team.\n'

setup(
    name='textbox',
    version='0.0.3',
    description='A package for building text generation systems',
    url='https://github.com/RUCAIBox/TextBox',
    author='TextBoxTeam',
    author_email='rucaibox@163.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.2",
    packages=[
        package for package in find_packages()
        if package.startswith('textbox')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False)
