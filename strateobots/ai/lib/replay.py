import numpy as np
import random
import time
import logging
from strateobots.replay import ReplayDataStorage
from strateobots.ai.lib.model_function import encode_vector_for_model
from strateobots.ai.lib import data


log = logging.getLogger(__name__)


class ReplayMemory:

    def __init__(self, storage_directory, model, props_function,
                 load_winner_data=False, load_loser_data=True,
                 cache_key=None, force_generate_cache=False,
                 load_predicate=None,
                 rotate_storage=False,
                 max_games_keep=100):
        self._rds = ReplayDataStorage(storage_directory)
        self.model = model
        self._load_winner_data = load_winner_data
        self._load_loser_data = load_loser_data
        self.max_games_keep = max_games_keep
        self.props_function = props_function
        self._data = []
        self._prepared_epoch = None
        self._cache_key = cache_key
        self.rotate_storage = rotate_storage

        log.info('Loading data from storage')
        remaining = self.max_games_keep
        keys = self._rds.list_keys()
        while remaining > 0 and keys:
            key = keys.pop(-1)
            if load_predicate:
                ok = load_predicate(self._rds.load_metadata(key))
                if not ok:
                    continue
            rd = _ReplayData(key)
            self._load_numpy(rd, force_generate_cache)
            self._data.append(rd)
            remaining -= 1

    def add_replay(self, metadata, replay_data):
        key = time.strftime('%Y%m%d_%H%M%S')
        self._rds.save_replay(key, metadata, replay_data)
        rd = _ReplayData(key)
        self._load_numpy(rd, True)
        self._data.append(rd)
        if len(self._data) > self.max_games_keep:
            if self.rotate_storage:
                self._rds.remove_replay(self._data[0].key)
            del self._data[0]

    def total_items(self):
        return sum(rd.ticks.size-1 for rd in self._data)

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
            for ctl in (*data.ALL_CONTROLS, 'target_orientation')
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
        actions_buffer = {ctl: np.empty([size]) for ctl in (*data.ALL_CONTROLS, 'target_orientation')}
        states_buffer = np.empty([size, self.model.state_dimension])
        next_states_buffer = np.empty([size, self.model.state_dimension])

        i = 0
        for rd in games:
            n = rd.ticks.size-1
            team = getattr(rd, side + '_team')

            action_idx = getattr(rd, side + '_action_idx')
            state_data = getattr(rd, side + '_state_data')

            props = self.props_function(rd.json_data, n, team, side == 'winner')[:n]
            if np.ndim(props) == 1:
                props = np.reshape(props, (props.size, 1))
            props_buffer[i:i+n] = props
            for ctl in (*data.ALL_CONTROLS, 'target_orientation'):
                actions_buffer[ctl][i:i+n] = action_idx[ctl][:-1]
            states_buffer[i:i+n] = state_data[:-1]
            next_states_buffer[i:i+n] = state_data[1:]
            i += n

        if shuffle:
            indices = np.arange(size)
            np.random.shuffle(indices)
            props_buffer = props_buffer[indices]
            for ctl in (*data.ALL_CONTROLS, 'target_orientation'):
                actions_buffer[ctl] = actions_buffer[ctl][indices]
            states_buffer = states_buffer[indices]
            next_states_buffer = next_states_buffer[indices]
        if requested_size != size:
            props_buffer = props_buffer[:requested_size]
            for ctl in (*data.ALL_CONTROLS, 'target_orientation'):
                actions_buffer[ctl] = actions_buffer[ctl][:requested_size]
            states_buffer = states_buffer[:requested_size]
            next_states_buffer = next_states_buffer[:requested_size]

        return props_buffer, actions_buffer, states_buffer, next_states_buffer

    def _load_numpy(self, rd, force_generate_cache):
        need_loser_data = self._load_loser_data
        need_winner_data = self._load_winner_data
        if not force_generate_cache and self._cache_key is not None:
            if need_loser_data:
                need_loser_data, rd.loser_state_data, rd.loser_action_idx = \
                    self._try_load_from_cache(rd, 'loser')
            if need_winner_data:
                need_winner_data, rd.winner_state_data, rd.winner_action_idx = \
                    self._try_load_from_cache(rd, 'winner')

        n_ticks_path = self._rds.get_path_for_extra_data(
            rd.key,
            'cache_{}_nticks'.format(self._cache_key)
        )
        try:
            with open(n_ticks_path) as ntf:
                n_ticks = int(ntf.read())
            need_n_ticks = False
        except:
            need_n_ticks = True
            n_ticks = None

        if need_winner_data or need_loser_data or need_n_ticks:
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

            if need_loser_data:
                rd.loser_state_data, rd.loser_action_idx = \
                    self._load_numpy_data_for_team(rd, loser_team, winner_team, last_tick+1)
            if need_winner_data and winner_team is not None:
                rd.winner_state_data, rd.winner_action_idx = \
                    self._load_numpy_data_for_team(rd, winner_team, loser_team, last_tick+1)

            n_ticks = last_tick + 1
            if need_n_ticks:
                with open(n_ticks_path, 'w') as ntf:
                    ntf.write(str(n_ticks))

        rd.ticks = np.arange(n_ticks)

        if need_winner_data:
            self._save_cache(rd, 'winner')
        if need_loser_data:
            self._save_cache(rd, 'loser')

    def _load_numpy_data_for_team(self, rd, team, opponent_team, end_t):
        state_array = np.empty([end_t, self.model.state_dimension], dtype=np.float32)
        encoder = self.model.data_encoder()
        action_arrays = {
            ctl: np.empty([end_t], dtype=np.float32)
            for ctl in (*data.ALL_CONTROLS, 'target_orientation')
        }
        for i in range(end_t):
            item = rd.json_data[i]
            state_vector = encode_vector_for_model(
                encoder, item, team, opponent_team)
            state_array[i] = state_vector
        for ctl in data.ALL_CONTROLS:
            act_arr = action_arrays[ctl]
            get_idx = getattr(data, 'ctl_'+ctl).categories.index
            for i in range(end_t):
                act_arr[i] = get_idx(rd.json_data[i]['controls'][team][0][ctl])

        action_arrays['target_orientation'] = np.array([
            rd.json_data[i]['controls'][team][0]['target_orientation']
            for i in range(end_t)
        ], dtype=np.float32)

        return state_array, action_arrays

    def _try_load_from_cache(self, rd, winner_or_loser):
        state_cache_path = self._rds.get_path_for_extra_data(
            rd.key,
            'cache_{}_{}_state.npy'.format(self._cache_key, winner_or_loser)
        )
        idx_cache_path = self._rds.get_path_for_extra_data(
            rd.key,
            'cache_{}_{}_actionidx.npy'.format(self._cache_key, winner_or_loser)
        )
        state = idx = None
        try:
            with open(state_cache_path, 'rb') as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(0, 0)
                    state = np.load(f, allow_pickle=False)

            if state is not None:
                idx = np.load(idx_cache_path, allow_pickle=False)
                idx_dict = {}
                for i, ctl in enumerate((*data.ALL_CONTROLS, 'target_orientation')):
                    idx_dict[ctl] = idx[i]
                idx = idx_dict

            need_load = False
        except:
            need_load = True
        return need_load, state, idx

    def _save_cache(self, rd, winner_or_loser):
        state_cache_path = self._rds.get_path_for_extra_data(
            rd.key,
            'cache_{}_{}_state.npy'.format(self._cache_key, winner_or_loser)
        )
        idx_cache_path = self._rds.get_path_for_extra_data(
            rd.key,
            'cache_{}_{}_actionidx.npy'.format(self._cache_key, winner_or_loser)
        )

        with open(state_cache_path, 'wb') as sf, open(idx_cache_path, 'wb') as af:
            state = rd.get_state_data(winner_or_loser)
            idx_dict = rd.get_action_idx(winner_or_loser)
            if state is not None:
                idx = np.stack([
                    idx_dict[ctl]
                    for ctl in (*data.ALL_CONTROLS, 'target_orientation')
                ])
                np.save(sf, state, allow_pickle=False)
                np.save(af, idx, allow_pickle=False)


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

    def get_state_data(self, winner_or_loser):
        return getattr(self, '{}_state_data'.format(winner_or_loser))

    def get_action_idx(self, winner_or_loser):
        return getattr(self, '{}_action_idx'.format(winner_or_loser))


class NotEnoughData(Exception):
    pass
