import logging

import tensorflow as tf

from strateobots.ai.lib import layers, stable
from strateobots.ai.lib.data import action2vec, bot2vec
from strateobots.ai.dqn_duel import core

log = logging.getLogger(__name__)


class QualityFunction(core.QualityFunction):
    class Model:
        QualityFunction = None

        def __init__(self, levels):
            self.levels_cfg = tuple(levels)
            self.name = 'QFunc'
            self.construct_params = dict(levels=self.levels_cfg)
            self.var_list = []
            self.layers = []
            with tf.variable_scope(self.name):
                # l0x = layers.Linear.Model('L0x', 4, self.levels_cfg[0])
                # l0y = layers.Linear.Model('L0y', 4, self.levels_cfg[0], l0x.weight)
                # l0a = layers.Linear.Model('L0a', 29, self.levels_cfg[0])
                l0x = layers.Linear.Model('L0x', 6, self.levels_cfg[0])
                l0y = layers.Linear.Model('L0y', 6, self.levels_cfg[0], l0x.weight)
                l0a = layers.Linear.Model('L0a', 8, self.levels_cfg[0])
                self.layers.append((l0x, l0y, l0a))
                d_in = levels[0]
                for i, d_out in enumerate(self.levels_cfg[1:], 1):
                    lx = layers.Linear.Model('L{}x'.format(i), d_in, d_out)
                    ly = layers.Linear.Model('L{}y'.format(i), d_in, d_out, lx.weight)
                    la = layers.Linear.Model('L{}a'.format(i), 2 * d_in, d_out)
                    self.layers.append((lx, ly, la))
                    d_in = d_out

                for lx, ly, la in self.layers:
                    self.var_list.extend(
                        [*lx.var_list, *ly.var_list, *la.var_list])

                self.qw = tf.get_variable('QW', [3 * d_in, action2vec.vector_length - 4])
                self.var_list.append(self.qw)

    def __init__(self, model, state, action):
        """
        :param model: QualityFunction.Model
        :param state: [..., aug_state_vector_len]
        :param action: [..., action_vector_len]
        """
        self.model = model  # type: QualityFunction.Model
        self.state = state  # type: tf.Tensor
        self.action = action[..., 3:-1]  # type: tf.Tensor

        bvl = bot2vec.vector_length
        # self.angles0 = tf.concat([state[..., bvl-2:bvl], state[..., -2:]], -1)  # 4
        # self.cos0 = tf.concat([state[..., 4:bvl-2], state[..., bvl+4:-2], action], -1)  # 25
        # self.x0 = tf.concat([state[..., :4:2], state[..., bvl:bvl+4:2]], -1)  # 4
        # self.y0 = tf.concat([state[..., 1:4:2], state[..., bvl+1:bvl+4:2]], -1)  # 4
        # self.a0 = tf.concat([self.angles0, tf.acos(self.cos0)], -1)  # 29
        self.angles0 = tf.concat([state[..., bvl - 2:bvl], state[..., -2:]], -1)  # 4
        self.cos0 = tf.concat([state[..., 7:9], state[..., bvl + 7:bvl + 9]], -1)  # 4
        self.sin0 = tf.sqrt(1 - tf.square(self.cos0))
        self.x0 = tf.concat([state[..., :1], state[..., bvl:bvl + 1], self.cos0], -1)  # 6
        self.y0 = tf.concat([state[..., 1:2], state[..., bvl + 1:bvl + 2], self.sin0], -1)  # 6
        self.a0 = tf.concat([self.angles0, tf.acos(self.cos0)], -1)  # 8

        def make_activation(dim):
            def activation(vec):
                half = dim // 2
                vec1 = tf.nn.relu(vec[..., :half])
                vec2 = tf.identity(vec[..., half:])
                return tf.concat([vec1, vec2], -1)

            return activation

        self.levels = []
        vectors = (self.x0, self.y0, self.a0)
        for i, (mx, my, ma) in enumerate(self.model.layers):
            x, y, a = vectors
            lx = layers.Linear(mx.name, x, mx, make_activation(mx.out_dim))
            ly = layers.Linear(my.name, y, my, make_activation(my.out_dim))
            la = layers.Linear(ma.name, a, ma, make_activation(ma.out_dim))
            a_cos = tf.cos(la.out)
            a_sin = tf.sin(la.out)
            new_x = lx.out * a_cos - ly.out * a_sin
            new_y = lx.out * a_sin + ly.out * a_cos
            add_a = stable.atan2(ly.out, lx.out)
            new_a = tf.concat([la.out, add_a], -1)
            self.levels.append((lx, ly, la, (new_x, new_y)))
            vectors = (new_x, new_y, new_a)

        final_vector = tf.concat([vectors[0], vectors[2]], -1)
        self.features = layers.batch_matmul(final_vector, self.model.qw, )
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.features)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        masked_features = self.action * self.features
        with tf.control_dependencies([finite_assert]):
            self.quality = tf.reduce_mean(masked_features, axis=-1)

    def get_quality(self):
        return self.quality

    def call(self, state, action, session):
        return session.run(self.quality, feed_dict={
            self.state: state,
            self.action: action,
        })


Model = QualityFunction.Model
Model.QualityFunction = QualityFunction
