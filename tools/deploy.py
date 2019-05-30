#!/usr/bin/env PYTHONPATH=. python3

import os
import argparse
import subprocess
import logging
import datetime
import ssh_lib as ssh


log = logging.getLogger()


def get_local_git_commit():
    gh = subprocess.check_output("git rev-parse HEAD", shell=True).decode().strip()
    log.info("Local git hash: %s", gh)
    return gh


def get_remote_git_commit(client: ssh.SSH, remote_dir: str):
    gh = client.cmd(f"cd {remote_dir}; git rev-parse HEAD")
    gh = gh.decode().strip()
    log.info("Remote git hash: %s", gh)
    return gh


def generate_patch():
    log.info("generating patch...")
    return subprocess.check_output("git diff HEAD", shell=True)


def send_and_deploy_patch(client: ssh.SSH, patch_contents: bytes, remote_dir: str):
    rem_temp_file = client.cmd("mktemp").decode().strip()
    # rem_temp_file = os.path.join(remote_dir, "tmp.patch")
    log.info("sending patch contents...")
    client.send_file(rem_temp_file, patch_contents)
    client.cmd(f"cd {remote_dir}; git reset --hard")
    client.cmd(f"cd {remote_dir}; git apply --index {rem_temp_file}")
    client.cmd(f"rm {rem_temp_file}")


def send_configs(client: ssh.SSH, remote_dir: str):
    # log.info("sending remote config...")
    # with open("remote-config.ini", "rb") as f:
    #     client.send_file(os.path.join(remote_dir, "config.ini"), f)
    pass


def freshen_remote(client: ssh.SSH, remote_dir: str):
    log.info("Freshen remote's code...")
    client.cmd(f"cd {remote_dir}; git reset --hard")
    client.cmd(f"cd {remote_dir}; git pull")


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("server", nargs="?")
    opts = ap.parse_args()

    ssh_conn, remote_path = ssh.get_ssh_connection("config.ini", opts.server)

    with ssh_conn:

        local_commit = get_local_git_commit()
        remote_commit = get_remote_git_commit(ssh_conn, remote_path)

        if local_commit != remote_commit:
            freshen_remote(ssh_conn, remote_path)
            remote_commit = get_remote_git_commit(ssh_conn, remote_path)

        if local_commit != remote_commit:
            raise Exception("Git commits do not match")
        patch = generate_patch()
        send_and_deploy_patch(ssh_conn, patch, remote_path)
        send_configs(ssh_conn, remote_path)
    log.info("Done at %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
