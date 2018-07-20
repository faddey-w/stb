import tensorflow as tf

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import state2vec
from strateobots.ai.ain_duel.model.base import BaseActionInferenceModel


X_FEATURES = (
    (0, 'x'),
    (0, 'vx'),
    (1, 'x'),
    (1, 'vx'),
    (2, 'x'),
    (3, 'x'),
)
Y_FEATURES = (
    (0, 'y'),
    (0, 'vy'),
    (1, 'y'),
    (1, 'vy'),
    (2, 'y'),
    (3, 'y'),
)


class ActionInferenceModel(BaseActionInferenceModel):

    def _create_layers(self, res_cfg, sigm_cfg):

        self.nodes = []
        in_dim = state2vec.vector_length + len(X_FEATURES)
        for i, (hidden_dim, out_dim) in enumerate(res_cfg):
            node = layers.ResidualV2('R{}'.format(i), in_dim, hidden_dim, out_dim)
            self.nodes.append(node)
            in_dim = out_dim
        for i, out_dim in enumerate(sigm_cfg):
            node = layers.Linear('S{}'.format(i), in_dim, out_dim)
            self.nodes.append(node)
            in_dim = out_dim

        return in_dim, self.nodes

    def create_inference(self, inference, state):
        inference.layers = []

        angles = tf.atan2(
            select_features(state, state2vec, *Y_FEATURES),
            select_features(state, state2vec, *X_FEATURES),
        )

        vector = tf.concat([state, angles], -1)
        for layer in self.nodes:
            func = tf.nn.relu if isinstance(layer, layers.ResidualV2) else tf.sigmoid
            node = layer.apply(vector, func)
            vector = node.out
            inference.layers.append(node)
        return vector

Model = ActionInferenceModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
