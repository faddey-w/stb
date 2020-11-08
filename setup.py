#!/usr/bin/env python3
from setuptools import setup, find_packages
from Cython.Build import cythonize


setup(
    name="strateobots",
    version="0.2.0",
    packages=find_packages(include=["strateobots*"]),
    package_data={"": ["strateobots/visualizer_app/frontend"]},
    include_package_data=True,
    install_requires=[
        "tornado",
        "tensorflow==1.15",
        "torch==1.6",
        "cython==0.29.21",
    ],
    ext_modules=cythonize(
        "strateobots/ai/evolution/evo_core.pyx",
        compiler_directives={"language_level": 3},
    ),
)
