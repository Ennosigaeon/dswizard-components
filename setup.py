# -*- coding: utf-8 -*-
import pathlib

from pkg_resources import VersionConflict, require

try:
    require('setuptools>=38.3')
except VersionConflict:
    import sys

    print('Error: version of setuptools is too old (<38.3)!')
    sys.exit(1)

from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# The text of the README file
README = (pathlib.Path(__file__).parent / 'README.md').read_text()

if __name__ == '__main__':
    setup(
        name='dswizard-components',
        version='0.2.3',
        description='Contains all base algorithms used by dswizard',
        long_description=README,
        long_description_content_type='text/markdown',
        author='Marc Zoeller',
        author_email='m.zoeller@usu.de',
        url='https://github.com/Ennosigaeon/dswizard-components',
        license='MIT',
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9'
        ],
        packages=find_namespace_packages(include=['dswizard.*']),
        python_requires='>=3.7',
        include_package_data=True,
        install_requires=requirements,
        keywords=['automl', 'machine learning', 'pipeline synthesis']
    )
