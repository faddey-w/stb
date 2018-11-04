import os
import json
import shutil
from strateobots.util import cached_with_timeout_m


class ReplayDataStorage:

    def __init__(self, storage_directory):
        self.storage_directory = storage_directory
        os.makedirs(storage_directory, exist_ok=True)

    def save_replay(self, key, metadata, replay_data):
        _, metadata_path, replay_path = self._prepare_paths(key, True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        with open(replay_path, 'w') as f:
            json.dump(replay_data, f, separators=(',', ':'))

    def list_keys(self):
        keys = os.listdir(self.storage_directory)
        return [
            k for k in keys
            if os.path.isdir(os.path.join(self.storage_directory, k))
        ]

    def load_metadata(self, key):
        _, pth, _ = self._prepare_paths(key, False)
        try:
            with open(pth, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise SimulationNotFound(key)

    def load_replay_data(self, key):
        _, _, pth = self._prepare_paths(key, False)
        try:
            with open(pth, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise SimulationNotFound(key)

    def remove_replay(self, key):
        dir_path, _, _ = self._prepare_paths(key, False)
        try:
            shutil.rmtree(dir_path)
        except os.error:
            raise SimulationNotFound(key)

    def _prepare_paths(self, key, do_create):
        dir_path = os.path.join(self.storage_directory, key)
        if do_create:
            os.makedirs(dir_path, exist_ok=True)
        metadata_path = os.path.join(dir_path, 'metadata.json')
        replay_path = os.path.join(dir_path, 'replay.json')
        return dir_path, metadata_path, replay_path


class CachedReplayDataStorage(ReplayDataStorage):

    @cached_with_timeout_m(5)
    def load_metadata(self, key):
        return super().load_metadata(key)

    @cached_with_timeout_m(30)
    def load_replay_data(self, key):
        return super().load_replay_data(key)


class SimulationNotFound(Exception):

    def __init__(self, key):
        self.key = key
        super(SimulationNotFound, self).__init__(key)
