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

    def __init__(self, coord_cfg, angle_cfg, fc_cfg, exp_layers):
        assert len(coord_cfg) == len(angle_cfg)
        assert len(exp_layers) <= len(fc_cfg)
        assert fc_cfg[-1] == 7+1  # action2vec.vector_length - 4 + 1

        self.vec2d_cfg = tuple(zip(coord_cfg, angle_cfg))
        self.exp_layers_cfg = tuple(exp_layers)
        self.fc_cfg = tuple(fc_cfg)
        self.name = 'QFunc'
        self.var_list = []

        self.vec2d_layers = []
        self.fc_layers = []

        with tf.variable_scope(self.name):
            coord_in, angle_in = 2, 2
            for i, (coord_out, angle_out) in enumerate(self.vec2d_cfg):
                lx = layers.Linear('L{}x'.format(i), coord_in, coord_out)
                ly = layers.Linear('L{}y'.format(i), coord_in, coord_out, lx.weight)
                la = layers.Linear('L{}a'.format(i), coord_in + angle_in, angle_out)
                self.vec2d_layers.append((lx, ly, la))
                coord_in, angle_in = coord_out, angle_out

            fc_in = 3 + coord_in + 2 * angle_in
            for i, fc_out in enumerate(self.fc_cfg):
                fc = layers.Linear('FC{}'.format(i), fc_in, fc_out)
                self.fc_layers.append(fc)
                fc_in = fc_out

            for lx, ly, la in self.vec2d_layers:
                self.var_list.extend([*lx.var_list, *ly.var_list, *la.var_list])

            for fc in self.fc_layers:
                self.var_list.extend(fc.var_list)

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

        self.lin0 = select_features(
            state, state2vec,
            (0, 'hp_ratio'),
            # (0, 'load'),
            (1, 'hp_ratio'),
            (1, 'load'),
        )
        self.x0 = select_features(
            state, state2vec,
            (0, 'x'),
            (1, 'x'),
        )
        self.y0 = select_features(
            state, state2vec,
            (0, 'y'),
            (1, 'y'),
        )
        self.angles0 = select_features(
            state, state2vec,
            (0, 'orientation'),
            (0, 'tower_orientation'),
            # (1, 'rotation'),
            # (1, 'tower_rotation'),
        )
        self.a0 = tf.concat([self.angles0, stable.atan2(self.y0, self.x0)], -1)

        def make_activation(dim, activation=tf.nn.relu, angle=False):
            def function(vec):
                if angle:
                    vec = (vec % (2*pi)) - pi
                half = dim // 2
                vec1 = activation(vec[..., :half])
                vec2 = tf.identity(vec[..., half:])
                return tf.concat([vec1, vec2], -1)

            return function

        self.vec2d_levels = []
        vectors = (self.x0, self.y0, self.a0)
        for mx, my, ma in self.model.vec2d_layers:
            x, y, a = vectors
            lx = mx.apply(x, make_activation(mx.out_dim))
            ly = my.apply(y, make_activation(my.out_dim))
            la = ma.apply(a, make_activation(ma.out_dim, angle=True))
            a_cos = tf.cos(la.out)
            a_sin = tf.sin(la.out)
            new_x = lx.out * a_cos - ly.out * a_sin
            new_y = lx.out * a_sin + ly.out * a_cos
            add_a = stable.atan2(ly.out, lx.out)
            new_a = tf.concat([la.out, add_a], -1)
            self.vec2d_levels.append((lx, ly, la, (new_x, new_y)))
            vectors = (new_x, new_y, new_a)

        fc_vec = tf.concat([vectors[0], vectors[2], self.lin0], -1)
        self.fc_layers = []
        for i, fc in enumerate(self.model.fc_layers):
            if i in model.exp_layers_cfg:
                act = make_activation(fc.out_dim, lambda t: 0.95 ** tf.nn.relu(t))
            else:
                act = make_activation(fc.out_dim)
            fc_lr = fc.apply(fc_vec, act)
            self.fc_layers.append(fc_lr)
            fc_vec = fc_lr.out

        self.features = fc_vec
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
Model = QualityFunctionModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
