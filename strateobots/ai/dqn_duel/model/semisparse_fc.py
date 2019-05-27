import logging

import tensorflow as tf

from strateobots.ai.lib import layers, util
from strateobots.ai.lib.data import action2vec, state2vec

log = logging.getLogger(__name__)


class QualityFunctionModel:
    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, cfg, n_parts):
        self.name = "QFuncSemiSparse"
        self.var_list = []
        self.n_parts = n_parts

        with tf.variable_scope(self.name):
            self.node_lists = []
            in_dim = state2vec.vector_length + action2vec.vector_length + 1
            for i, out_dim in enumerate(cfg):
                if i != 0:
                    in_dim *= 2
                self.node_lists.append(
                    [
                        layers.Linear("Lin{}_{}".format(i, j), in_dim, out_dim)
                        for j in range(n_parts)
                    ]
                )
                in_dim = out_dim

            self.regress = layers.Linear("Regress", n_parts * cfg[-1], 1)

        self.var_list.extend(
            var
            for node_list in [*self.node_lists, [self.regress]]
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
        state = util.normalize_state(state)
        state += tf.zeros_like(action[..., :1])
        self.model = model  # type: QualityFunctionModel
        self.state = state  # type: tf.Tensor
        self.action = action

        x0 = select_features(state, state2vec, (0, "x"))
        y0 = select_features(state, state2vec, (0, "y"))
        x1 = select_features(state, state2vec, (1, "x"))
        y1 = select_features(state, state2vec, (1, "y"))
        to_enemy = tf.atan2(y1 - y0, x1 - x0)

        self.inputs = tf.concat([state, action, to_enemy], -1)

        self.layers = []
        vector_array = [self.inputs]
        for node_list in model.node_lists:
            out_array = []
            for i, node in enumerate(node_list):
                if len(vector_array) == 1:
                    vec = vector_array[0]
                else:
                    v1 = vector_array[i - 1]
                    v2 = vector_array[i]
                    vec = tf.concat([v1, v2], -1)
                out_array.append(node.apply(vec, tf.sigmoid))
            vector_array = [out.out for out in out_array]
            self.layers.append(out_array)
        vector = tf.concat(vector_array, -1)

        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(vector)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.q_v = model.regress.apply(vector, tf.identity)
            self.quality = tf.squeeze(self.q_v.out, -1)

    def get_quality(self):
        return self.quality

    def call(self, state, action, session):
        return session.run(
            self.quality, feed_dict={self.state: state, self.action: action}
        )


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx : idx + 1])
    return tf.concat(feature_tensors, -1)
