import numpy as np
import tensorflow as tf

from strateobots.ai.lib import layers
from strateobots.ai.lib.util import add_batch_shape


class SelectOneAction:

    def __init__(self, qfunc_model, state):
        n_actions = qfunc_model.n_actions
        all_actions = [
            [1.0 if i == j else 0.0 for j in range(n_actions)]
            for i in range(n_actions)
        ]

        batch_shape = layers.shape_to_list(state.shape[:-1])
        batched_all_actions = np.reshape(
            all_actions,
            [n_actions] + [1] * len(batch_shape) + [n_actions]
        ) + np.zeros([n_actions, *batch_shape, n_actions])

        self.all_actions = tf.constant(all_actions, dtype=tf.float32)
        self.batched_all_actions = tf.constant(batched_all_actions, dtype=tf.float32)
        self.state = add_batch_shape(state, [1])

        self.qfunc = qfunc_model.apply(self.state, self.batched_all_actions)
        self.max_idx = tf.argmax(self.qfunc.get_quality(), 0)
        self.max_q = tf.reduce_max(self.qfunc.get_quality(), 0)

        self.action = tf.gather_nd(
            self.all_actions,
            tf.expand_dims(self.max_idx, -1),
        )
