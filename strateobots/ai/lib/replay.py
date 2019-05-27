import numpy as np
import random
import time
import logging
from strateobots.replay import ReplayDataStorage
from strateobots.ai.lib.model_function import encode_vector_for_model
from strateobots.ai.lib import data


log = logging.getLogger(__name__)


class ReplayMemory:
    def __init__(
        self,
        storage_directory,
        model,
        props_function,
        cache_key=None,
        controls=data.ALL_CONTROLS,
        rotate_storage=False,
        max_games_keep=100,
    ):
        self._rds = ReplayDataStorage(storage_directory)
        self.model = model
        self.max_games_keep = max_games_keep
        self.props_function = props_function
        self._data = []
        self._prepared_epoch = None
        self._cache_key = cache_key
        self.rotate_storage = rotate_storage
        self.controls = controls

    def reload(self, force_generate_cache=False, load_predicate=None):
        self._data = []
        log.info("Loading data from storage")
        remaining = self.max_games_keep
        keys = self._rds.list_keys()
        while remaining > 0 and keys:
            key = keys.pop(-1)
            metadata = self._rds.load_metadata(key)
            team1, team2 = metadata["team1"], metadata["team2"]
            for t1, t2 in [(team1, team2), (team2, team1)]:
                if load_predicate:
                    ok = load_predicate(metadata, t1, t2)
                    if not ok:
                        continue
                rd = _ReplayData(key, metadata, t1, t2)
                self._ensure_numpy_loaded(rd, force_generate_cache)
                self._data.append(rd)
                remaining -= 1

    def add_replay(self, metadata, replay_data, load_predicate=None):
        key = time.strftime("%Y%m%d_%H%M%S")
        self._rds.save_replay(key, metadata, replay_data)

        team1, team2 = replay_data[0]["bots"].keys()

        results = []
        for t1, t2 in [(team1, team2), (team2, team1)]:
            if load_predicate:
                ok = load_predicate(metadata, t1, t2)
                if not ok:
                    continue
            rd = _ReplayData(key, metadata, t1, t2)
            self._load_numpy_data(rd, replay_data)
            self._save_cache(rd)
            self._data.append(rd)
            results.append(rd)
        while len(self._data) > self.max_games_keep:
            if self.rotate_storage:
                self._rds.remove_replay(self._data[0].key)
            del self._data[0]
        return results

    def total_items(self):
        return sum(rd.ticks.size - 1 for rd in self._data)

    def prepare_epoch(
        self, batch_size, n_batches, shuffle=False, selector=None, replay_predicate=None
    ):
        if replay_predicate is not None:
            assert selector is None

            def selector(games):
                return [rd for rd in games if replay_predicate(rd)]

        epoch = self._get_data_flat(n_batches * batch_size, shuffle, selector)
        self._prepared_epoch = batch_size, n_batches, epoch

    def get_prepared_epoch_batch(self, batch_index):
        batch_size, _, epoch = self._prepared_epoch
        start = batch_size * batch_index
        end = batch_size * (batch_index + 1)
        props, actions, st_before, st_after = epoch
        actions = {ctl: actions[ctl][start:end] for ctl in self.controls}
        return props[start:end], actions, st_before[start:end], st_after[start:end]

    def _get_data_flat(self, size, shuffle, selector):
        all_games = self._data
        if selector:
            all_games = selector(all_games)
        if size is None:
            games = all_games
            size = sum(rd.ticks.size for rd in games)
            requested_size = size
        else:
            games = all_games[::-1]
            if shuffle:
                random.shuffle(games)

            least_size = size
            i = 0
            while least_size > 0:
                try:
                    # there is minus one because we consider pairs
                    # previous state + current state
                    # as single entry of training
                    least_size -= games[i].ticks.size - 1
                except IndexError:
                    raise NotEnoughData
                i += 1
            games = games[:i]
            requested_size = size
            size += -least_size  # least_size is negative now

        props_buffer = np.empty([size, getattr(self.props_function, "dimension", 1)])
        actions_buffer = {ctl: np.empty([size]) for ctl in self.controls}
        states_buffer = np.empty([size, self.model.state_dimension])
        next_states_buffer = np.empty([size, self.model.state_dimension])

        i = 0
        for rd in games:
            # there is minus one because we consider pairs
            # previous state + current state
            # as single entry of training
            n = rd.ticks.size - 1

            props = self.props_function(rd)
            if not isinstance(props, np.ndarray):
                props = np.array(props)
            assert props.shape[0] == n
            if np.ndim(props) == 1:
                props = np.reshape(props, (props.size, 1))
            props_buffer[i : i + n] = props
            for ctl in self.controls:
                actions_buffer[ctl][i : i + n] = rd.actions_data_dict[ctl][:-1]
            states_buffer[i : i + n] = rd.state_data[:-1]
            next_states_buffer[i : i + n] = rd.state_data[1:]
            i += n

        if shuffle:
            indices = np.arange(size)
            np.random.shuffle(indices)
            props_buffer = props_buffer[indices]
            for ctl in self.controls:
                actions_buffer[ctl] = actions_buffer[ctl][indices]
            states_buffer = states_buffer[indices]
            next_states_buffer = next_states_buffer[indices]
        if requested_size != size:
            props_buffer = props_buffer[:requested_size]
            for ctl in self.controls:
                actions_buffer[ctl] = actions_buffer[ctl][:requested_size]
            states_buffer = states_buffer[:requested_size]
            next_states_buffer = next_states_buffer[:requested_size]

        return props_buffer, actions_buffer, states_buffer, next_states_buffer

    def _ensure_numpy_loaded(self, rd, force_generate_cache):
        if not force_generate_cache and self._cache_key is not None:
            cache_ok = self._try_load_from_cache(rd)
            need_load_data = not cache_ok
        else:
            need_load_data = False

        if need_load_data:
            self._load_numpy_data(rd)
            self._save_cache(rd)

    def _load_numpy_data(self, rd, replay_data=None):
        if replay_data is None:
            replay_data = self._rds.load_replay_data(rd.key)
        rd.json_data = replay_data
        last_tick = next(
            i
            for i in range(len(rd.json_data) - 1, -1, -1)
            if rd.json_data[i]["controls"][rd.team] is not None
        )
        n_ticks = last_tick + 1

        state_array = np.empty([n_ticks, self.model.state_dimension], dtype=np.float32)
        encoder = self.model.data_encoder()
        action_arrays = {
            ctl: np.empty([n_ticks], dtype=np.float32) for ctl in self.controls
        }
        for i in range(n_ticks):
            item = rd.json_data[i]
            state_vector = encode_vector_for_model(
                encoder, item, rd.team, rd.opponent_team
            )
            state_array[i] = state_vector
        for ctl in self.controls:
            act_arr = action_arrays[ctl]
            if data.is_categorical(ctl):
                get_value = getattr(data, "ctl_" + ctl).categories.index
            else:
                get_value = lambda x: x
            for i in range(n_ticks):
                act_arr[i] = get_value(rd.json_data[i]["controls"][rd.team][0][ctl])

        rd.state_data = state_array
        rd.actions_data_dict = action_arrays
        rd.ticks = np.arange(n_ticks)

    def _try_load_from_cache(self, rd):
        state_cache_path, actions_cache_paths, n_ticks_path = self._get_cache_paths(rd)
        try:
            state = np.load(state_cache_path, allow_pickle=False)
            actions = {
                ctl: np.load(actions_cache_paths[ctl], allow_pickle=False)
                for ctl in self.controls
            }
            with open(n_ticks_path) as ntf:
                n_ticks = int(ntf.read())
        except:
            return False
        else:
            rd.state_data = state
            rd.actions_data_dict = actions
            rd.ticks = np.arange(n_ticks)
            return True

    def _save_cache(self, rd):
        state_cache_path, actions_cache_paths, _ = self._get_cache_paths(rd)
        np.save(state_cache_path, rd.state_data, allow_pickle=False)
        for ctl in self.controls:
            np.save(
                actions_cache_paths[ctl], rd.actions_data_dict[ctl], allow_pickle=False
            )

    def _get_cache_paths(self, rd):
        state_cache_path = self._rds.get_path_for_extra_data(
            rd.key, "cache_{}_{}_state.npy".format(self._cache_key, rd.team)
        )
        actions_cache_paths = {
            ctl: self._rds.get_path_for_extra_data(
                rd.key,
                "cache_{}_{}_action_{}.npy".format(self._cache_key, rd.team, ctl),
            )
            for ctl in self.controls
        }
        n_ticks_path = self._rds.get_path_for_extra_data(
            rd.key, "cache_{}_{}_nticks".format(self._cache_key, rd.key)
        )
        return state_cache_path, actions_cache_paths, n_ticks_path


class _ReplayData:
    def __init__(self, key, metadata, team, opponent_team):
        self.key = key
        self.metadata = metadata
        self.json_data = None
        self.team = team
        self.opponent_team = opponent_team
        self.state_data = None
        self.actions_data_dict = None
        self.ticks = None


class NotEnoughData(Exception):
    pass
