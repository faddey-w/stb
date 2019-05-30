import os
import io
import paramiko
import logging
import warnings


log = logging.getLogger()


class SSH:
    def __init__(self, ssh_address, ssh_user, private_key_path=None):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if ssh_address.count(":") == 1:
            hostname, port = ssh_address.split(":")
            port = int(port)
        else:
            hostname, port = ssh_address, paramiko.config.SSH_PORT
        if private_key_path is None:
            private_key_path = os.path.expanduser("~/.ssh/id_rsa")
        pkey = paramiko.RSAKey.from_private_key_file(private_key_path)
        self._address = ssh_address
        self._connect_params = dict(
            hostname=hostname, port=port, username=ssh_user, timeout=30, pkey=pkey
        )
        self._private_key_path = private_key_path
        self._client = client
        self._sftp = None

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        log.info("connecting to %s...", self._address)
        with warnings.catch_warnings():
            # annoying deprecation warning from Crypto lib
            warnings.simplefilter("ignore")
            self._client.connect(**self._connect_params)

    def stop(self):
        if self._sftp is not None:
            self._sftp.close()
        self._client.close()

    def cmd(self, command: str) -> bytes:
        log.info("remote: %s", command)
        chan = self._client.get_transport().open_session()  # type: paramiko.Channel
        chan.exec_command(command)
        output = chan.makefile("rb").read()
        errors = chan.makefile_stderr("r").read()
        rc = chan.recv_exit_status()
        if rc != 0:
            raise SshCommandError(f"{command} -> exited with {rc}: {errors}")
        return output

    def _get_stfp(self) -> paramiko.SFTP:
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def send_file(self, remote_path: str, contents):
        log.info("sending file to %s", remote_path)
        if isinstance(contents, (str, bytes)):
            if isinstance(contents, str):
                contents = contents.encode()
            contents_io = io.BytesIO()
            contents_io.write(contents)
            contents_io.seek(0)
            contents = contents_io
        self._get_stfp().putfo(contents, remote_path)

    def download_file(self, remote_path: str, local_file=None) -> bytes:
        log.info("downloading file from %s", remote_path)
        if local_file is None:
            local_file = io.BytesIO()
            return_value = True
            need_close = False
        elif isinstance(local_file, str):
            local_file = open(local_file, "wb")
            return_value = False
            need_close = True
        else:
            return_value = False
            need_close = False
        self._get_stfp().getfo(remote_path, local_file)
        if need_close:
            local_file.close()
        if return_value:
            return local_file.getvalue()

    def stat_file(self, remote_path: str):
        try:
            return self._get_stfp().stat(remote_path)
        except FileNotFoundError:
            return None

    def get_connect_commandline(self):
        user = self._connect_params["username"]
        return f"ssh -i {self._private_key_path} {user}@{self._address}"


class SshCommandError(Exception):
    pass


import configparser
import aws_resources


def get_ssh_connection(ini_file, server_name=None):

    cp = configparser.ConfigParser()
    cp.read([ini_file])
    if server_name is None:
        server_name = cp.get("server", "default")
    if server_name.startswith("aws."):
        api = aws_resources.EC2Instances(config_ini=ini_file)
        instance_name = server_name[4:]
        instance = api.get_by(name=instance_name)
        private_key_path = cp.get("server.aws", "private_key_path")
        username = cp.get("server.aws", "user_" + instance_name)
        home_dir = cp.get("server.aws", "home_dir").format(user=username)
        return api.get_ssh(instance, private_key_path, username), home_dir
    section = f"server.{server_name}"
    ssh_user = cp.get(section, "ssh_user")
    ssh_address = cp.get(section, "ssh_address")
    home_dir = cp.get(section, "home_dir")
    client = SSH(ssh_address, ssh_user)
    return client, home_dir

