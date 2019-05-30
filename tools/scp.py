#!/usr/bin/env PYTHONPATH=. python

import os
import argparse
import logging
import ssh_lib


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument('direction', choices=["get", "put"])
    ap.add_argument('file1')
    ap.add_argument('file2', nargs="?")
    opts = ap.parse_args()
    if opts.file2 is None:
        opts.file2 = opts.file1

    ssh_conn, home_dir = ssh_lib.get_ssh_connection("config.ini")
    with ssh_conn:
        if opts.direction == "get":
            ssh_conn.download_file(os.path.join(home_dir, opts.file1), opts.file2)
        else:
            with open(opts.file1, "rb") as f:
                ssh_conn.send_file(os.path.join(home_dir, opts.file2), f)


if __name__ == '__main__':
    main()

