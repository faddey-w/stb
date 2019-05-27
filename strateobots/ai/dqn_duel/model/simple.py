import logging

import tensorflow as tf

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import action2vec, state2vec

log = logging.getLogger(__name__)


class QualityFunctionModel:
    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, cfg):
        self.name = "QFuncSimple"
        self.var_list = []

        with tf.variable_scope(self.name):
            self.nodes = []
            in_dim = state2vec.vector_length + action2vec.vector_length + 1
            for i, out_dim in enumerate(cfg):
                node = layers.Linear("Lin{}".format(i), in_dim, out_dim)
                self.nodes.append(node)
                in_dim = out_dim

            self.regress = layers.Linear("Regress", in_dim, 1)

        self.var_list.extend(
            var for node in [*self.nodes, self.regress] for var in node.var_list
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
        normalizer = tf.one_hot(
            [
                state2vec[0, "x"],
                state2vec[0, "y"],
                state2vec[1, "x"],
                state2vec[1, "y"],
                state2vec[2, "x"],
                state2vec[2, "y"],
                state2vec[3, "x"],
                state2vec[3, "y"],
            ],
            depth=state2vec.vector_length,
            on_value=1.0 / 1000,
            off_value=1.0,
        )
        normalizer = tf.reduce_min(normalizer, 0)
        # import pdb; pdb.set_trace()
        state *= normalizer
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
        vector = self.inputs
        for layer in model.nodes:
            node = layer.apply(vector, tf.sigmoid)
            vector = node.out
            self.layers.append(node)

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
