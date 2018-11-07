import numpy as np
import random
import time
import logging
from strateobots.replay import ReplayDataStorage
from strateobots.ai.lib.integration import encode_vector_for_model
from strateobots.ai.lib import data


log = logging.getLogger(__name__)


class ReplayMemory:

    def __init__(self, storage_directory, model, props_function, load_winner_data=False, max_games_keep=100):
        self._rds = ReplayDataStorage(storage_directory)
        self.model = model
        self._load_winner_data = load_winner_data
        self.max_games_keep = max_games_keep
        self.props_function = props_function
        self._data = []
        self._prepared_epoch = None

        log.info('Loading data from storage')
        for key in self._rds.list_keys()[-self.max_games_keep:]:
            rd = _ReplayData(key)
            self._load_numpy(rd)
            self._data.append(rd)

    def add_replay(self, metadata, replay_data):
        key = time.strftime('%Y%m%d_%H%M%S')
        self._rds.save_replay(key, metadata, replay_data)
        rd = _ReplayData(key)
        self._load_numpy(rd)
        self._data.append(rd)
        if len(self._data) > self.max_games_keep:
            # self._rds.remove_replay(self._data[0].key)
            del self._data[0]

    def prepare_epoch(self, win_batch_size, lost_batch_size, n_batches, shuffle=False):
        win_epoch = self.get_winner_data_flat(n_batches * win_batch_size, shuffle)
        lost_epoch = self.get_loser_data_flat(n_batches * lost_batch_size, shuffle)
        self._prepared_epoch = win_batch_size, lost_batch_size, n_batches, win_epoch, lost_epoch

    def get_prepared_epoch_batch(self, batch_index):
        win_batch_size, lost_batch_size, _, win_epoch, lost_epoch = self._prepared_epoch
        props_w, act_w, st_bef_w, st_aft_w = self._get_batch(win_batch_size, batch_index, win_epoch)
        props_l, act_l, st_bef_l, st_aft_l = self._get_batch(lost_batch_size, batch_index, lost_epoch)

        actions = {
            ctl: np.concatenate([act_w[ctl], act_l[ctl]])
            for ctl in act_w.keys()
        }
        return (
            np.concatenate([props_w, props_l]),
            actions,
            np.concatenate([st_bef_w, st_bef_l]),
            np.concatenate([st_aft_w, st_aft_l]),
        )

    def _get_batch(self, batch_size, batch_index, epoch_data):
        start = batch_size * batch_index
        end = batch_size * (batch_index + 1)
        props, actions, st_before, st_after = epoch_data
        actions = {
            ctl: actions[ctl][start:end]
            for ctl in data.ALL_CONTROLS
        }
        return props[start:end], actions, st_before[start:end], st_after[start:end]

    def get_loser_data_flat(self, size=None, shuffle=False):
        return self._get_data_flat('loser', size, shuffle)

    def get_winner_data_flat(self, size=None, shuffle=False):
        return self._get_data_flat('winner', size, shuffle)

    def _get_data_flat(self, side, size, shuffle):
        all_games = [
            rd for rd in self._data
            if getattr(rd, side+'_state_data') is not None
        ]
        if size is None:
            size = sum(rd.ticks.size for rd in all_games)
            games = all_games
            requested_size = size
        else:
            games = all_games[::-1]
            if shuffle:
                random.shuffle(games)

            least_size = size
            i = 0
            while least_size > 0:
                try:
                    least_size -= games[i].ticks.size-1
                except IndexError:
                    raise NotEnoughData
                i += 1
            games = games[:i]
            requested_size = size
            size -= least_size  # least_size is negative

        props_buffer = np.empty([size, getattr(self.props_function, 'dimension', 1)])
        actions_buffer = {ctl: np.empty([size]) for ctl in data.ALL_CONTROLS}
        states_buffer = np.empty([size, self.model.state_dimension])
        next_states_buffer = np.empty([size, self.model.state_dimension])

        i = 0
        for rd in games:
            n = rd.ticks.size-1
            team = getattr(rd, side + '_team')

            action_idx = getattr(rd, side + '_action_idx')
            state_data = getattr(rd, side + '_state_data')

            props = self.props_function(rd.json_data, team, side == 'winner')[:-1]
            if np.ndim(props) == 1:
                props = np.reshape(props, (props.size, 1))
            props_buffer[i:i+n] = props
            for ctl in data.ALL_CONTROLS:
                actions_buffer[ctl][i:i+n] = action_idx[ctl][:-1]
            states_buffer[i:i+n] = state_data[:-1]
            next_states_buffer[i:i+n] = state_data[1:]
            i += n

        if shuffle:
            indices = np.arange(size)
            np.random.shuffle(indices)
            props_buffer = props_buffer[indices]
            for ctl in data.ALL_CONTROLS:
                actions_buffer[ctl] = actions_buffer[ctl][indices]
            states_buffer = states_buffer[indices]
            next_states_buffer = next_states_buffer[indices]
        if requested_size != size:
            props_buffer = props_buffer[:requested_size]
            for ctl in data.ALL_CONTROLS:
                actions_buffer[ctl] = actions_buffer[ctl][:requested_size]
            states_buffer = states_buffer[:requested_size]
            next_states_buffer = next_states_buffer[:requested_size]

        return props_buffer, actions_buffer, states_buffer, next_states_buffer

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
        rd.winner_team = winner_team
        rd.loser_team = loser_team

        last_tick = next(i for i in range(len(rd.json_data)-1, -1, -1)
                         if rd.json_data[i]['controls'][team1] is not None)

        rd.loser_state_data, rd.loser_action_idx = \
            self._load_numpy_data_for_team(rd, loser_team, winner_team, last_tick+1)
        if self._load_winner_data and winner_team is not None:
            rd.winner_state_data, rd.winner_action_idx = \
                self._load_numpy_data_for_team(rd, winner_team, loser_team, last_tick+1)
        rd.ticks = np.arange(last_tick+1)

    def _load_numpy_data_for_team(self, rd, team, opponent_team, end_t):
        state_array = np.empty([end_t, self.model.state_dimension], dtype=np.float32)
        action_arrays = {
            ctl: np.empty([end_t], dtype=np.float32)
            for ctl in data.ALL_CONTROLS
        }
        prev_state = None
        for i in range(end_t):
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
        self.winner_team = None
        self.loser_team = None
        self.winner_state_data = None
        self.loser_state_data = None
        self.winner_action_idx = None
        self.loser_action_idx = None
        self.ticks = None


class NotEnoughData(Exception):
    pass
