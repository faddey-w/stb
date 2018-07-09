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

    def __init__(self, lin_h, log_h, lin_o, n_evt):
        self.name = 'QFuncAug'
        self.var_list = []

        with tf.variable_scope(self.name):
            n_lin0 = 9
            n_log0 = 3

            self.lin_h = layers.Linear('Lin_H', n_lin0, lin_h)
            self.lin_o = layers.Linear('Lin', n_lin0 + lin_h, 2*(lin_o + n_log0))
            # self.lin_o = layers.Linear('Lin', n_lin0 + lin_h, lin_o)

            self.log_h = layers.Linear('Log_H', n_lin0, log_h)
            self.log_o = layers.Linear('Log', n_lin0 + log_h, lin_o)

            self.evt_hp = layers.Linear('Evt_hp', 2*(lin_o + n_log0), n_evt)
            self.evt_const = layers.Linear('Evt_const', 2*(lin_o + n_log0), n_evt)
            # self.evt_hp = layers.Linear('Evt_hp', lin_o, n_evt)
            # self.evt_const = layers.Linear('Evt_const', lin_o, n_evt)

            self.event_weight = layers.Linear('EvtW', 2*n_evt, 1)

        self.var_list.extend([
            *self.lin_h.var_list,
            *self.lin_o.var_list,
            *self.log_h.var_list,
            *self.log_o.var_list,
            *self.evt_hp.var_list,
            *self.evt_const.var_list,
            *self.event_weight.var_list,
        ])

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
        state += tf.zeros_like(action[..., :1])
        self.model = model  # type: QualityFunctionModel
        self.state = state  # type: tf.Tensor
        self.action = action  # select_features(
        #     action, action2vec,
        #     'rotate_left',
        #     'rotate_no',
        #     'rotate_right',
        #     'tower_rotate_left',
        #     'tower_rotate_no',
        #     'tower_rotate_right',
        #     'fire'
        # )  # type: tf.Tensor

        x0 = select_features(state, state2vec, (0, 'x'))
        y0 = select_features(state, state2vec, (0, 'y'))
        x1 = select_features(state, state2vec, (1, 'x'))
        y1 = select_features(state, state2vec, (1, 'y'))
        to_enemy = tf.atan2(y1-y0, x1-x0)
        orientations = select_features(  # 3
            state, state2vec,
            (0, 'orientation'),
            (0, 'tower_orientation'),
            (2, 'orientation'),
        )
        rotations = select_features(  # 2
            action, action2vec, 'rotate_left', 'tower_rotate_left'
        ) - select_features(
            action, action2vec, 'rotate_right', 'tower_rotate_right'
        )
        shot_ready = select_features(state, state2vec, (0, 'load')) > 0.999

        lin0 = select_features(  # 3
            state, state2vec,
            (1, 'hp_ratio'),
            (0, 'load'),
            (2, 'remaining_range'),
        )
        hp = select_features(state, state2vec, (1, 'hp_ratio'))
        angles0 = tf.concat([to_enemy, orientations, rotations], -1)  # 6
        angles0 = ((angles0 + pi) % (2*pi)) - pi  # normed to -pi..pi

        self.linear0 = tf.concat([lin0, angles0], -1)  # 9
        self.logical0 = tf.concat([  # 3
            tf.to_float(shot_ready),
            select_features(state, state2vec, (2, 'present')),
            select_features(action, action2vec, 'fire'),
        ], -1)

        self.lin_h = model.lin_h.apply(self.linear0, tf.nn.relu)
        self.lin_o = model.lin_o.apply(tf.concat([
            self.linear0, self.lin_h.out
        ], -1), tf.identity)

        self.log_h = model.log_h.apply(self.linear0, tf.nn.relu)
        self.log_o = model.log_o.apply(tf.concat([
            self.linear0, self.log_h.out
        ], -1), tf.sigmoid)
        # import pdb; pdb.set_trace()

        flags = tf.concat([
            self.log_o.out,
            self.logical0,
            1-self.log_o.out,
            1-self.logical0,
        ], -1)
        self.features = self.lin_o.out * flags
        # self.features = self.lin_o.out

        self.evt_hp = model.evt_hp.apply(self.features, tf.nn.relu)
        self.evt_const = model.evt_const.apply(self.features, tf.nn.relu)

        self.components = tf.concat([
            hp * tf.exp(-self.evt_hp.out),
            tf.exp(-self.evt_const.out),
        ], -1)

        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.components)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.q_v = model.event_weight.apply(self.components, tf.identity)
        self.quality = tf.squeeze(self.q_v.out, -1)

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
