import argparse
import configparser
import subprocess
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('program', choices=PROGRAMS.keys())
    parser.add_argument('--config', default='config.ini')
    opts = parser.parse_args()

    with open(opts.config) as f:
        config = configparser.ConfigParser()
        config.read_file(f)

    PROGRAMS[opts.program](config)


def run_dqn(config):
    """
    :type config: configparser.ConfigParser
    """
    save_model = config.getboolean('dqn', 'save_model', fallback=False)
    save_logs = config.getboolean('dqn', 'save_logs', fallback=False)
    save_dir = config.get('dqn', 'save_directory', fallback=None)
    max_games = config.getint('dqn', 'max_games', fallback=None)

    from strateobots.ai.dqn_duel.train import entrypoint
    entrypoint(save_model, save_logs, save_dir, max_games)


def run_tensorboard(config):
    """
    :type config: configparser.ConfigParser
    """
    section = config.get('tensorboard', 'from')
    save_dir = config.get(section, 'save_directory')

    save_dir = os.path.abspath(save_dir)
    subprocess.call('tensorboard --logdir ' + save_dir, shell=True,
                    stdout=sys.stdout, stderr=sys.stderr)


PROGRAMS = {
    'dqn': run_dqn,
    'board': run_tensorboard,
}

