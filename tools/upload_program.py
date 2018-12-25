import argparse
import requests
import configparser
import os
from urllib import parse as urlparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', default='config.ini')
    parser.add_argument('file')
    parser.add_argument('name', nargs='?')

    opts = parser.parse_args()

    with open(opts.config) as f:
        config = configparser.ConfigParser()
        config.read_file(f)
    server_url = config.get('upload_program', 'server_url')

    file_path = opts.file
    with open(file_path, 'rb') as f:
        code_txt = f.read()

    if opts.name is None:
        name = os.path.split(file_path)[1]
        if name.endswith('.py'):
            name = name[:-3]
    else:
        name = opts.name
    name = name.replace(' ', '_')

    api_url = urlparse.urljoin(server_url, '/api/v1/programs/' + name)

    resp = requests.post(api_url, code_txt)
    print(resp.reason)


if __name__ == '__main__':
    main()
