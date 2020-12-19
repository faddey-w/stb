#!/usr/bin/env python3
from setuptools import setup, find_packages
from Cython.Build import cythonize

cython_exts = cythonize(
    "stb/ai/evolution/evo_core.pyx",
    compiler_directives={"language_level": 3},
)
for ext in cython_exts:
    ext.optional = True

setup(
    name="stb",
    version="0.2.0",
    packages=find_packages(include=["stb*"]),
    package_data={"": ["stb/visualizer_app/frontend"]},
    include_package_data=True,
    install_requires=[
        "tornado",
        "cython==0.29.21",
    ],
    extras_require={
        "ml": [
            "tensorflow==1.15",
            "torch==1.6",
        ],
    },
    ext_modules=cython_exts,
)
