#!/usr/bin/env PYTHONPATH=. python

import argparse
import os
import ssh_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('server', nargs='?')
    ap.add_argument('--config', '-C', default='config.ini')
    opts = ap.parse_args()

    ssh_conn, home_dir = ssh_lib.get_ssh_connection(opts.config, opts.server)
    command = ssh_conn.get_connect_commandline()
    print(command)
    os.system(command)


if __name__ == '__main__':
    main()

