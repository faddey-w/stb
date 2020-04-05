#!/usr/bin/env python3
from distutils.core import setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize(
        "strateobots/ai/evolution/evo_core.pyx",
        compiler_directives={"language_level": 3},
    ),
    script_args=["build_ext", "--inplace"],
)
