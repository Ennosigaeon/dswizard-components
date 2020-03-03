# -*- coding: utf-8 -*-
"""
    Setup file for scaffold.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(name='sklearn-components',
          version='0.1',
          description='Contains sklearn algorithms',
          license='MIT',
          packages=['sklearn-components'],
          zip_safe=False)

    # setup(use_pyscaffold=True)
