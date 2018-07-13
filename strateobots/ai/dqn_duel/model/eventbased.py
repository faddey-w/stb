import logging

import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import action2vec, state2vec

log = logging.getLogger(__name__)


class QualityFunctionModel:

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, linear_cfg, logical_cfg, values_cfg):
        assert linear_cfg[-1][1] == logical_cfg[-1] == values_cfg[-1][1]
        self.name = 'QFuncEB'
        self.var_list = []

        with tf.variable_scope(self.name):
            n_lin0 = state2vec.vector_length + action2vec.vector_length + 2
            # n_lin0 = state2vec.vector_length + action2vec.vector_length + 1

            self.linear = []
            in_dim = n_lin0
            for i, (hidden_dim, out_dim) in enumerate(linear_cfg):
                node = layers.ResidualV2('Lin{}'.format(i), in_dim,
                                         hidden_dim, out_dim)
                self.linear.append(node)
                in_dim = out_dim

            self.logical = []
            in_dim = n_lin0
            for i, out_dim in enumerate(logical_cfg):
                node = layers.Linear('Log{}'.format(i), in_dim, out_dim)
                self.logical.append(node)
                in_dim = out_dim

            self.values = []
            in_dim = n_lin0
            for i, (hidden_dim, out_dim) in enumerate(values_cfg):
                node = layers.ResidualV2('Val{}'.format(i), in_dim,
                                         hidden_dim, out_dim)
                self.values.append(node)
                in_dim = out_dim

            self.event_weight = layers.Linear('EvtW', values_cfg[-1][1], 1)

        self.var_list.extend(
            var
            for node_list in [
                self.linear,
                self.logical,
                self.values,
                [self.event_weight],
            ]
            for node in node_list
            for var in node.var_list
        )

    def apply(self, state, action):
        return QualityFunction(self, state, action)


class QualityFunction:
    def __init__(self, model, state, action):
        """
        :type model: QualityFunctionModel
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
        # import pdb; pdb.set_trace()
        state *= normalizer
        state += tf.zeros_like(action[..., :1])
        self.model = model  # type: QualityFunctionModel
        self.state = state  # type: tf.Tensor
        self.action = action

        x0 = select_features(state, state2vec, (0, 'x'))
        y0 = select_features(state, state2vec, (0, 'y'))
        x1 = select_features(state, state2vec, (1, 'x'))
        y1 = select_features(state, state2vec, (1, 'y'))
        to_enemy = tf.atan2(y1 - y0, x1-x0)
        load = select_features(state, state2vec, (0, 'load'))
        shot_ready = tf.cast(load > 0.99, tf.float32)

        self.vector0 = tf.concat([
            state, action,
            to_enemy,
            shot_ready,
        ], -1)

        self.linear = []
        vector = self.vector0
        for layer in model.linear:
            node = layer.apply(vector, tf.abs)
            vector = node.out
            self.linear.append(node)
        linear_out = vector

        self.values = []
        vector = self.vector0
        for layer in model.values:
            node = layer.apply(vector, tf.nn.relu)
            vector = node.out
            self.values.append(node)
        values_out = vector

        self.logical = []
        vector = self.vector0
        for layer in model.logical:
            node = layer.apply(vector, tf.sigmoid)
            vector = node.out
            self.logical.append(node)
        logical_out = vector

        self.events = logical_out * values_out * tf.exp(-tf.nn.relu(linear_out))
        # self.events = values_out * tf.exp(-tf.nn.relu(linear_out))

        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.events)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.q_v = model.event_weight.apply(self.events, tf.identity)
            self.quality = tf.squeeze(self.q_v.out, -1)

    def get_quality(self):
        return self.quality

    def call(self, state, action, session):
        return session.run(self.quality, feed_dict={
            self.state: state,
            self.action: action,
        })
Model = QualityFunctionModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
