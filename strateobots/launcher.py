import argparse
import configparser
import subprocess
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('program', choices=PROGRAMS.keys())
    parser.add_argument('--config', default='config.ini')

    # for program="dqn"
    parser.add_argument('--new', action='store_true')

    # for program="board"
    parser.add_argument('--last', action='store_true')

    opts = parser.parse_args()

    with open(opts.config) as f:
        config = configparser.ConfigParser()
        config.read_file(f)

    PROGRAMS[opts.program](config, opts)


def run_dqn(config, opts):
    """
    :type config: configparser.ConfigParser
    """
    save_model = config.getboolean('dqn', 'save_model', fallback=False)
    save_logs = config.getboolean('dqn', 'save_logs', fallback=True)
    save_dir = config.get('dqn', 'save_directory', fallback=None)
    max_games = config.getint('dqn', 'max_games', fallback=None)

    if opts.new:
        save_dir = None

    from strateobots.ai.dqn_duel.train import entrypoint
    entrypoint(save_model, save_logs, save_dir, max_games)


def run_tensorboard(config, opts):
    """
    :type config: configparser.ConfigParser
    """
    section = config.get('tensorboard', 'from')
    last = config.getboolean('tensorboard', 'last', fallback=opts.last)
    if last:
        logs_root = config.get(section, 'logs_root')
        runs = [x for x in os.listdir(logs_root) if not x.startswith('_')]
        last_run = max(runs)
        save_dir = os.path.join(logs_root, last_run)
    else:
        save_dir = config.get(section, 'save_directory')

    save_dir = os.path.abspath(save_dir)
    subprocess.call('tensorboard --logdir ' + save_dir, shell=True,
                    stdout=sys.stdout, stderr=sys.stderr)


PROGRAMS = {
    'dqn': run_dqn,
    'board': run_tensorboard,
}
