import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import state2vec
from strateobots.ai.ain_duel.model.base import BaseActionInferenceModel


class ActionInferenceModel(BaseActionInferenceModel):

    def _create_layers(self):
        self.a1 = layers.Linear('A1', 3, 1)
        return 2, [self.a1]
        # return 2, []

    def create_inference(self, inference, state):
        ex = select_features(state, (1, 'x'))
        ey = select_features(state, (1, 'y'))
        bx = select_features(state, (0, 'x'))
        by = select_features(state, (0, 'y'))

        bo = select_features(state, (0, 'orientation'))
        to = select_features(state, (0, 'tower_orientation'))
        ea = tf.atan2(ey-by, ex-bx)
        angles = tf.concat([ea, bo, to], -1)
        angles = self.a1.apply(angles, tf.cos).out
        # angles = -pi + angles % (2 * pi)
        # angles = tf.sin(ea - (bo + to))

        ehp = select_features(state, (1, 'hp_ratio'))

        features = tf.concat([angles, -ehp], -1)

        return features

Model = ActionInferenceModel


def select_features(tensor, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = state2vec[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
