#!/usr/bin/env python3
import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

cython_exts = []
if not os.getenv("STB_NO_EVOCORE"):
    cython_exts.extend(
        cythonize(
            "stb/ai/evolution/evo_core.pyx",
            compiler_directives={"language_level": 3},
        )
    )

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
