import logging

import tensorflow as tf

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import state2vec
from .. import core

log = logging.getLogger(__name__)


class SimpleLogExpModel:

    def __init__(self, n_actions, cfg):
        self.var_list = []
        self.n_actions = n_actions

        self.nodes = []
        in_dim = 2 * state2vec.vector_length + n_actions + 1
        for i, out_dim in enumerate(cfg):
            node = layers.Linear('Lin{}'.format(i), in_dim, out_dim)
            self.nodes.append(node)
            in_dim = out_dim

        self.regress = layers.Linear('Regress', in_dim, 1)

        self.var_list.extend(
            var
            for node in [*self.nodes, self.regress]
            for var in node.var_list
        )

    def apply(self, state, action):
        return QualityFunction(self, state, action)


class QualityFunction:
    def __init__(self, model, state, action):
        """
        :type model: SimpleLogExpModel
        :param state: [..., state_vector_len]
        :param action: [..., action_vector_len]
        """
        # import pdb; pdb.set_trace()
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
        state += tf.zeros_like(action[..., :1])
        self.model = model  # type: SimpleLogExpModel
        self.state = state  # type: tf.Tensor
        self.action = action
        self.state_log = tf.log(0.1 + tf.abs(state))

        x0 = select_features(state, state2vec, (0, 'x'))
        y0 = select_features(state, state2vec, (0, 'y'))
        x1 = select_features(state, state2vec, (1, 'x'))
        y1 = select_features(state, state2vec, (1, 'y'))
        to_enemy = tf.atan2(y1-y0, x1-x0)

        self.inputs = tf.concat([state, action, to_enemy, self.state_log], -1)

        self.layers = []
        vector = self.inputs
        for layer in model.nodes:
            node = layer.apply(vector, tf.nn.relu)
            vector = node.out
            self.layers.append(node)
        vector = tf.exp(-tf.nn.relu(vector))

        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(vector)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.q_v = model.regress.apply(vector, tf.identity)
            self.quality = tf.squeeze(self.q_v.out, -1)

    def get_quality(self):
        return self.quality


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)


class QualityFunctionModelset(core.QualityFunctionModelset):

    node_cls = SimpleLogExpModel
    name = 'QFuncSimpleLogExp'
