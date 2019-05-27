import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("storage_path")
    opts = parser.parse_args()

    for name in os.listdir(opts.storage_path):
        dir_path = os.path.join(opts.storage_path, name)
        if os.path.isdir(dir_path):
            for subname in os.listdir(dir_path):
                if subname.startswith("_extra_"):
                    pth = os.path.join(dir_path, subname)
                    if os.path.isfile(pth):
                        os.remove(pth)


if __name__ == "__main__":
    main()
