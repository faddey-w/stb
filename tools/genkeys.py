#!/usr/bin/env python
import argparse
from strateobots import cryptoutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("private_keyfile")
    p.add_argument("public_keyfile", nargs="?")
    args = p.parse_args()

    private_keyfile = args.private_keyfile
    if args.public_keyfile:
        public_keyfile = args.public_keyfile
    else:
        public_keyfile = args.public_keyfile + ".pub"

    private, public = cryptoutil.generate_keypair()

    with open(private_keyfile, "w") as f:
        f.write(private)
    with open(public_keyfile, "w") as f:
        f.write(public_keyfile)


if __name__ == "__main__":
    main()
