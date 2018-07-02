import numpy as np
import os
import random


class ReplayMemory:

    def __init__(self, capacity, state_size, action_size):
        self._capacity = capacity
        self._states = np.empty((capacity, state_size, 2), np.float32)
        self._actions = np.empty((capacity, action_size), np.float32)
        self._used = 0
        self._last_insert = -1
        self._state_size = state_size
        self._action_size = action_size

    def save(self, save_dir):
        """
        Saves replay memory to files within specified directory
        :param save_dir: directory path where to save data
        """
        np.save(os.path.join(save_dir, 'state.npy'), self._states[:self._used])
        np.save(os.path.join(save_dir, 'action.npy'), self._actions[:self._used])

    def load(self, save_dir):
        """
        Loads replay memory from files within specified directory
        :param save_dir: directory path from where to load data
        """
        states = np.load(os.path.join(save_dir, 'state.npy'))
        actions = np.load(os.path.join(save_dir, 'action.npy'))
        load_size = states.shape[0]
        max_idx = min(self._used + load_size, self._capacity)
        min_idx = max(0, max_idx - load_size)
        insert_size = min(load_size, max_idx-min_idx)
        self._states[min_idx:min_idx+insert_size] = states[:insert_size]
        self._actions[min_idx:min_idx+insert_size] = actions[:insert_size]
        self._used = min_idx+insert_size
        self._last_insert = self._used-1

    @property
    def used_size(self):
        """
        :return: number of stored entries
        """
        return self._used

    def put_entry(self, state_before, action, state_after):
        """
        Adds a new entry into the memory
        :param state_before: 1-D vector of state before action
        :param action: 1-D vector that describes taken action
        :param state_after: 1-D vector of state after action
        :return: entry_id
        """
        if self._used < self._capacity:
            eid = self._used
            self._used += 1
        else:
            eid = (self._last_insert + 1) % self._capacity
        self._last_insert = eid
        self._states[eid, :, 0] = state_before
        self._actions[eid, :] = action
        self._states[eid, :, 1] = state_after
        return eid

    def get_entry(self, entry_id):
        """
        Gets memory entry at specified id.
        :param entry_id: result of previous put_entry() call
        :return: (state_before, action, state_after)
        """
        return self._states[entry_id, :, 0], self._actions[entry_id, :], self._states[entry_id, :, 1]

    def get_last_entries(self, n):
        """
        Returns N last entries
        :param n: number of entries to take
        :return: 2-D vectors of samples of (state_before, action, state_after)
        """
        if n > self._used:
            raise ValueError('not enough entries')
        if n <= self._last_insert + 1:
            smpl = slice(self._last_insert+1-n, self._last_insert+1)
        else:
            smpl = (
                *range(self._used-n+self._last_insert+1, self._used),
                *range(self._last_insert+1),
            )
        return self._states[smpl, :, 0], self._actions[smpl, :], self._states[smpl, :, 1]

    def get_random_sample(self, sample_size):
        """
        Generates a random sample of entries
        :param sample_size: number of entries to take
        :return: 2-D vectors of samples of (state_before, action, state_after)
        """
        smpl = tuple(random.sample(range(self._used), sample_size))
        return self._states[smpl, :, 0], self._actions[smpl, :], self._states[smpl, :, 1]

    def get_random_slice(self, slice_size):
        # TODO doctring
        start = random.randint(0, self._used-slice_size)
        slc = slice(start, start+slice_size)
        return self._states[slc, :, 0], self._actions[slc, :], self._states[slc, :, 1]

