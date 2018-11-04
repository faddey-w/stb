import numpy as np
import os
import random
import time
import json
from strateobots.replay import ReplayDataStorage
from strateobots.ai.lib.integration import encode_vector_for_model
from strateobots.ai.lib import data


class ReplayMemory:

    def __init__(self, storage_directory, model, load_winner_data=False):
        self._rds = ReplayDataStorage(storage_directory)
        self.model = model
        self._load_winner_data = load_winner_data
        self._data = []
        self._prepared_epoch = None

        for key in self._rds.list_keys():
            rd = _ReplayData(key)
            self._load_numpy(rd)
            self._data.append(rd)

    def add_replay(self, metadata, replay_data):
        key = time.strftime('%Y%m%d_%H%M%S')
        self._rds.save_replay(key, metadata, replay_data)
        rd = _ReplayData(key)
        self._load_numpy(rd)
        self._data.append(rd)

    def prepare_epoch(self, size=None, shuffle=False):
        self._prepared_epoch = self.get_loser_data_flat(size, shuffle)

    def get_prepared_epoch_batch(self, batch_size, batch_index):
        start = batch_size * batch_index
        end = batch_size * (batch_index + 1)
        ticks, actions, st_before, st_after = self._prepared_epoch
        actions = {
            ctl: actions[start:end]
            for ctl in data.ALL_CONTROLS
        }
        return ticks[start:end], actions, st_before[start:end], st_after[start:end]

    def get_loser_data_flat(self, size=None, shuffle=False):
        if size is None:
            size = sum(rd.ticks.size for rd in self._data)
            games = self._data
            requested_size = size
        else:
            games = self._data[::-1]
            if shuffle:
                random.shuffle(games)

            least_size = size
            i = 0
            while least_size > 0:
                least_size -= games[i].ticks.size
                i += 1
            games = games[:i+1]
            requested_size = size
            size -= least_size  # least_size is negative

        ticks_buffer = np.empty([size])
        actions_buffer = {ctl: np.empty([size]) for ctl in data.ALL_CONTROLS}
        states_buffer = np.empty([size, self.model.state_dimension])
        next_states_buffer = np.empty([size, self.model.state_dimension])

        i = 0
        for rd in games:
            n = rd.ticks.size
            ticks_buffer[i:i+n] = rd.ticks
            for ctl in data.ALL_CONTROLS:
                actions_buffer[ctl][i:i+n] = rd.loser_action_idx[ctl]
            states_buffer[i:i+n] = rd.loser_numpy_data[:-1]
            next_states_buffer[i:i+n] = rd.loser_numpy_data[1:]
            i += n

        if shuffle:
            indices = np.arange(size)
            np.random.shuffle(indices)
            ticks_buffer = ticks_buffer[indices]
            actions_buffer = actions_buffer[indices]
            states_buffer = states_buffer[indices]
            next_states_buffer = next_states_buffer[indices]
        if requested_size != size:
            ticks_buffer = ticks_buffer[:requested_size]
            actions_buffer = actions_buffer[:requested_size]
            states_buffer = states_buffer[:requested_size]
            next_states_buffer = next_states_buffer[:requested_size]

        return ticks_buffer, actions_buffer, states_buffer, next_states_buffer

    def _load_numpy(self, rd):
        rd.json_data = self._rds.load_replay_data(rd.key)
        team1, team2 = rd.json_data[-1]['bots'].keys()
        if not rd.json_data[-1]['bots'][team1]:
            winner_team = team2
            loser_team = team1
        elif not rd.json_data[-1]['bots'][team2]:
            winner_team = team1
            loser_team = team2
        else:
            winner_team = None
            loser_team = team1

        last_tick = next(i for i in range(len(rd.json_data)-1, -1)
                         if rd.json_data['controls'][team1] is not None)

        rd.loser_numpy_data, rd.loser_action_idx = \
            self._load_numpy_data_for_team(rd, loser_team, winner_team, last_tick+1)
        if self._load_winner_data and winner_team is not None:
            rd.winner_numpy_data, rd.winner_action_idx = \
                self._load_numpy_data_for_team(rd, winner_team, loser_team, last_tick+1)
        rd.ticks = np.arange(last_tick+1)

    def _load_numpy_data_for_team(self, rd, team, opponent_team, end_t):
        state_array = np.empty([end_t+1, self.model.state_dimension], dtype=np.float32)
        action_arrays = {
            ctl: np.empty([end_t], dtype=np.float32)
            for ctl in data.ALL_CONTROLS
        }
        prev_state = None
        for i in range(end_t+1):
            item = rd.json_data[i]
            state_vector, prev_state = encode_vector_for_model(
                self.model, item, prev_state,
                team, opponent_team
            )
            state_array[i] = state_vector
        for ctl in data.ALL_CONTROLS:
            act_arr = action_arrays[ctl]
            get_idx = getattr(data, 'ctl_'+ctl).categories.index
            for i in range(end_t):
                act_arr[i] = get_idx(rd.json_data[i]['controls'][team][0][ctl])

        return state_array, action_arrays


class _ReplayData:

    def __init__(self, key):
        self.key = key
        self.json_data = None
        self.winner_numpy_data = None
        self.loser_numpy_data = None
        self.winner_action_idx = None
        self.loser_action_idx = None
        self.ticks = None

