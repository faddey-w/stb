import numpy as np
import os
import random
import json


class ReplayMemory:

    def __init__(self, capacity, *vector_sizes):
        self._capacity = capacity
        self._storages = [
            np.empty((capacity, vsz), np.float32)
            for vsz in vector_sizes
        ]
        self._sizes = vector_sizes
        self._used = 0
        self._last_insert = -1

    def save(self, save_dir):
        """
        Saves replay memory to files within specified directory
        :param save_dir: directory path where to save data
        """
        os.makedirs(save_dir, exist_ok=True)
        for i, strg in enumerate(self._storages):
            np.save(os.path.join(save_dir, '{}.npy'.format(i)), strg[:self._used])

    def load(self, save_dir):
        """
        Loads replay memory from files within specified directory
        :param save_dir: directory path from where to load data
        """
        data = [
            np.load(os.path.join(save_dir, '{}.npy'.format(i)))
            for i in range(len(self._sizes))
        ]
        load_size = data[0].shape[0]
        max_idx = min(self._used + load_size, self._capacity)
        min_idx = max(0, max_idx - load_size)
        insert_size = min(load_size, max_idx-min_idx)
        for i, strg in enumerate(self._storages):
            strg[min_idx:min_idx+insert_size] = data[i][:insert_size]
        self._used = min_idx+insert_size
        self._last_insert = self._used-1

    def update(self, memory):
        """
        Adds entries from given instance to this one.
        Equivalent to iterating entries from given buffer and putting them here.
        :type memory: ReplayMemory
        """
        if memory._used == 0:
            return
        if memory._used > self._capacity:
            data = memory.get_last_entries(self._capacity)
            for i, strg in enumerate(self._storages):
                strg[...] = data[i]
        else:
            data = [
                strg[:memory._used]
                for strg in memory._storages
            ]
            if memory._last_insert < memory._used-1:
                data = [
                    np.concatenate([
                        strg[memory._last_insert + 1:memory._used],
                        strg[:memory._last_insert + 1],
                    ], 0)
                    for strg in data
                ]
            if memory._used < self._capacity - self._used:
                assert self._last_insert == self._used - 1
                for i, arr in enumerate(data):
                    self._storages[i][self._used:self._used+memory._used] = arr
                self._used += memory._used
                self._last_insert = self._used - 1
            elif memory._used < self._capacity - self._last_insert - 1:
                for i, arr in enumerate(data):
                    self._storages[i][self._last_insert + 1:self._last_insert + 1 + memory._used] = arr
                self._last_insert += memory._used
            else:
                rem = self._capacity - (self._last_insert + 1)
                for i, arr in enumerate(data):
                    self._storages[i][self._last_insert+1:] = arr[:rem]
                    self._storages[i][:memory._used-rem] = arr[rem:]
                self._used = self._capacity
                self._last_insert = memory._used - rem - 1

    def trunc(self, to_size, offset=0):
        assert offset+to_size <= self._used
        if offset != 0:
            for i, strg in enumerate(self._storages):
                strg[:to_size] = strg[offset:offset+to_size]
        self._used = to_size

    @property
    def used_size(self):
        """
        :return: number of stored entries
        """
        return self._used

    def put_entry(self, *vectors):
        """
        Adds a new entry into the memory
        :param vectors: 1-D vectors
        :return: entry_id
        """
        if self._used < self._capacity:
            eid = self._used
            self._used += 1
        else:
            eid = (self._last_insert + 1) % self._capacity
        self._last_insert = eid
        for i, vec in enumerate(vectors):
            self._storages[i][eid, :] = vec
        return eid

    def get_entry(self, entry_id):
        """
        Gets memory entry at specified id.
        :param entry_id: result of previous put_entry() call
        :return: (state_before, action, state_after)
        """
        return [strg[entry_id, :] for strg in self._storages]

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
        return [strg[smpl, :] for strg in self._storages]

    def get_random_sample(self, sample_size):
        """
        Generates a random sample of entries
        :param sample_size: number of entries to take
        :return: 2-D vectors of samples of (state_before, action, state_after)
        """
        smpl = tuple(random.sample(range(self._used), sample_size))
        return [strg[smpl, :] for strg in self._storages]

    def get_random_slice(self, slice_size):
        # TODO doctring
        start = random.randint(0, self._used-slice_size)
        slc = slice(start, start+slice_size)
        return [strg[slc, :] for strg in self._storages]


class BalancedMemory:

    def __init__(self, keyfunc, cap_per_class, vector_sizes):
        self._keyfunc = keyfunc
        self._cap_per_class = cap_per_class
        self._vector_sizes = vector_sizes
        self._classes = {}

    def save(self, save_dir, serialize_key=str):
        keymap = {key: str(i) for i, key in enumerate(self._classes.keys())}
        for key, mem in self._classes.items():
            subdir = keymap[key]
            mem.save(os.path.join(save_dir, subdir))
        with open(os.path.join(save_dir, 'keymap.json'), 'w') as f:
            json.dump(
                {serialize_key(key): subdir
                 for key, subdir in keymap.items()},
                f
            )

    def load(self, save_dir, load_key=str):
        with open(os.path.join(save_dir, 'keymap.json')) as f:
            keymap = json.load(f)
        for key, subdir in keymap.items():
            key = load_key(key)
            mem = self._get_mem(key)
            mem.load(os.path.join(save_dir, subdir))
            self._classes[key] = mem

    def update(self, memory):
        arrays = memory.get_last_entries(memory.used_size)
        for vectors in zip(*arrays):
            self.put_entry(*vectors)

    def trunc(self, to_size, offset=0):
        to_sizes = self._distribute(to_size)
        offsets = self._distribute(offset)

        for key, mem in self._classes.items():
            to_size = to_sizes[key]
            offset = offsets[key]
            offset = max(0, min(offset, mem.used_size - to_size))
            mem.trunc(to_size, offset)

    @property
    def used_size(self):
        return sum(mem.used_size for mem in self._classes.values())

    def put_entry(self, *vectors):
        key = self._keyfunc(*vectors)
        mem = self._get_mem(key)
        eid = mem.put_entry(*vectors)
        return key, eid

    def get_entry(self, entry_id):
        key, entry_id = entry_id
        return self._classes[key].get_entru(entry_id)

    def get_last_entries(self, n):
        ns = self._distribute(n, n)
        return self._combine_samples(
            self._classes[key].get_last_entries(n)
            for key, n in ns.items()
        )

    def get_random_sample(self, sample_size):
        szs = self._distribute(sample_size, sample_size)
        return self._combine_samples(
            self._classes[key].get_random_sample(sz)
            for key, sz in szs.items()
        )

    def get_random_slice(self, slice_size):
        szs = self._distribute(slice_size, slice_size)
        return self._combine_samples(
            self._classes[key].get_random_slice(sz)
            for key, sz in szs.items()
        )

    def _distribute(self, parameter, max_bins=None):
        if not self._classes:
            raise ValueError("BalancedMemory is empty")
        itemlist = list(self._classes.items())
        if max_bins is not None and max_bins < len(itemlist):
            itemlist = random.sample(itemlist, max_bins)
        sizes = {key: mem.used_size for key, mem in itemlist}
        keys_asc = sorted(sizes.keys(), key=sizes.get)
        result = {}
        while keys_asc:
            average = parameter / len(keys_asc)
            key = keys_asc.pop(0)
            sz = sizes[key]
            take = min(sz, int(average))
            result[key] = take
            parameter -= take
        return result

    def _get_mem(self, key):
        try:
            return self._classes[key]
        except KeyError:
            mem = ReplayMemory(self._cap_per_class, *self._vector_sizes)
            self._classes[key] = mem
            return mem

    def _combine_samples(self, samplesets):
        return [
            np.concatenate(column)
            for column in zip(*samplesets)
        ]
