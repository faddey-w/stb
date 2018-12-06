import os
import argparse

from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension


def _parse_cli_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args(argv)
    return args.root, args.code_files


def main(argv=None):
    # root_dir, code_files = _parse_cli_args(argv)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_files = [
        root_dir + '/strateobots/engine.py',
        root_dir + '/strateobots/ai/treesearch.py',
        root_dir + '/strateobots/ai/simple_duel.py',
    ]

    ext_modules = []
    for f in code_files:
        if not f.endswith('.py'):
            continue
        modpath = os.path.relpath(f, root_dir)
        imppath = modpath.replace('.py', '').replace('/', '.')
        ext_modules.append(Extension(imppath, [modpath]))
    setup(
        name='Corlina',
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules,
        script_args=['build_ext', '--inplace']
    )


if __name__ == '__main__':
    main()
