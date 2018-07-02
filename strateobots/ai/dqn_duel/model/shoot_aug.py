import logging

import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers, stable
from strateobots.ai.lib.data import action2vec, state2vec

log = logging.getLogger(__name__)


class QualityFunctionModel:

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, fc_cfg, exp_layers, pool_layers, join_point):
        assert len(exp_layers) <= len(fc_cfg)
        assert join_point <= len(fc_cfg)
        assert fc_cfg[-1] == 7+1  # action2vec.vector_length - 4 + 1

        self.exp_layers_cfg = tuple(exp_layers)
        self.pool_layers_cfg = tuple(pool_layers)
        self.join_point = join_point
        self.fc_cfg = tuple(fc_cfg)
        self.name = 'QFuncAug'
        self.var_list = []

        self.layers = []
        self.alt_layers = {}

        with tf.variable_scope(self.name):

            d_in = 3
            for i, d_out in enumerate(self.fc_cfg):
                if i == join_point:
                    d_in += 2
                lr = layers.Linear.Model('FC{}'.format(i), d_in, d_out)
                self.layers.append(lr)
                if i in pool_layers:
                    lr = layers.Linear.Model('Alt{}'.format(i), d_in, d_out)
                    self.alt_layers[i] = lr
                d_in = d_out

            for lr in self.layers:
                self.var_list.extend(lr.var_list)

            for lr in self.alt_layers.values():
                self.var_list.extend(lr.var_list)

    def apply(self, state, action):
        return QualityFunction(self, state, action)


class QualityFunction:
    def __init__(self, model, state, action):
        """
        :param model: QualityFunctionModel
        :param state: [..., state_vector_len]
        :param action: [..., action_vector_len]
        """
        self.model = model  # type: QualityFunctionModel
        self.state = state  # type: tf.Tensor
        self.action = select_features(
            action, action2vec,
            'rotate_left',
            'rotate_no',
            'rotate_right',
            'tower_rotate_left',
            'tower_rotate_no',
            'tower_rotate_right',
            'fire'
        )  # type: tf.Tensor

        x0 = select_features(state, state2vec, (0, 'x'))
        y0 = select_features(state, state2vec, (0, 'y'))
        x1 = select_features(state, state2vec, (1, 'x'))
        y1 = select_features(state, state2vec, (1, 'y'))
        to_enemy = tf.atan2(y1-y0, x1-x0)
        orientations = select_features(
            state, state2vec,
            (0, 'orientation'),
            (0, 'tower_orientation'),
        )

        self.lin0 = select_features(  # 2
            state, state2vec,
            (1, 'hp_ratio'),
            (1, 'load'),
        )
        self.angles0 = tf.concat([to_enemy, orientations], -1)  # 3
        self.angles0 = pi + ((self.angles0 - pi) % (2*pi))  # normed to -pi..pi

        def make_activation(dim, activation=tf.nn.relu, angle=False):
            def function(vec):
                if angle:
                    vec = (vec % (2*pi)) - pi
                half = dim // 2
                vec1 = activation(vec[..., :half])
                vec2 = tf.identity(vec[..., half:])
                return tf.concat([vec1, vec2], -1)

            return function

        self.levels = []
        self.alt_levels = {}
        vector = self.angles0
        for i, lm in enumerate(self.model.layers):
            if i == self.model.join_point:
                vector = tf.concat([vector, self.lin0], -1)
            lr = layers.Linear(lm.name, vector, lm, tf.nn.relu)
            self.levels.append(lr)
            if i in self.model.alt_layers:
                alt_lm = self.model.alt_layers[i]
                alt_lr = layers.Linear(alt_lm.name, vector, alt_lm, tf.nn.relu)
                self.alt_levels[i] = alt_lr
                vector = tf.maximum(lr.out, alt_lr.out)
            else:
                vector = lr.out

        self.features = vector
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.features)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        action_mask = tf.concat([self.action, tf.ones_like(self.action[..., :1])], -1)
        masked_features = action_mask * self.features
        with tf.control_dependencies([finite_assert]):
            self.quality = tf.reduce_mean(masked_features, axis=-1)

    def get_quality(self):
        return self.quality

    def call(self, state, action, session):
        return session.run(self.quality, feed_dict={
            self.state: state,
            self.action: action,
        })


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
