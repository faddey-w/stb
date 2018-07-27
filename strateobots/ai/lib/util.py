import tensorflow as tf
from .data import state2vec


class Average:

    def __init__(self):
        self.value = 0
        self.n = 0

    def reset(self):
        self.value = 0
        self.n = 0

    def add(self, x):
        self.value += x
        self.n += 1

    def get(self):
        return self.value / max(1, self.n)


def normalize_state(state):
    normalizer = tf.one_hot([
        state2vec[0, 'x'],
        state2vec[0, 'y'],
        state2vec[1, 'x'],
        state2vec[1, 'y'],
        state2vec[2, 'x'],
        state2vec[2, 'y'],
        state2vec[3, 'x'],
        state2vec[3, 'y'],
    ], depth=state2vec.vector_length, on_value=1.0 / 1000, off_value=1.0)
    normalizer = tf.reduce_min(normalizer, 0)
    state *= normalizer

    normalizer = tf.one_hot([
        state2vec[0, 'vx'],
        state2vec[0, 'vy'],
        state2vec[1, 'vx'],
        state2vec[1, 'vy'],
    ], depth=state2vec.vector_length, on_value=1.0 / 10, off_value=1.0)
    normalizer = tf.reduce_min(normalizer, 0)
    state *= normalizer
    return state
