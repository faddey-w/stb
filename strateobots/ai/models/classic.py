import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import state2vec
from strateobots.ai.rwr.model.base import BaseActionInferenceModel

ANGLE_FEATURES = (
    (0, 'orientation'),
    (0, 'tower_orientation'),
    (1, 'orientation'),
    (1, 'tower_orientation'),
    (2, 'orientation'),
    (3, 'orientation'),
)
RANGE_FEATURES = (
    (0, 'x'),
    (0, 'y'),
    (1, 'x'),
    (1, 'y'),
    (2, 'remaining_range'),
    (3, 'remaining_range'),
)
OTHER_FEATURES = (
    (0, 'hp_ratio'),
    (0, 'load'),
    (0, 'shield_ratio'),
    (1, 'hp_ratio'),
    (1, 'load'),
    (1, 'shield_ratio'),
    (2, 'present'),
    (3, 'present'),
)


class ActionInferenceModel(BaseActionInferenceModel):

    def _create_layers(self, layer_sizes, angle_sections):

        self.layers = []
        in_dim = (
            + (angle_sections + 1)  # vector from bot to enemy
            + (angle_sections + 1)  # vector from enemy to our bullet
            + (angle_sections + 1)  # vector from bot to enemy's bullet
            + 2 * (angle_sections + 1)  # 2 bot velocity vectors
            + angle_sections * len(ANGLE_FEATURES)
            + len(RANGE_FEATURES)
            + len(OTHER_FEATURES)
        )
        for i, out_dim in enumerate(layer_sizes):
            node = layers.Linear('Lin{}'.format(i), in_dim, out_dim)
            self.layers.append(node)
            in_dim = out_dim

        self.angle_sections = angle_sections

        return in_dim, self.layers

    def create_inference(self, inference, state):
        nodes = []
        f = selector(state)

        bmx = f(0, 'x')
        bmy = f(0, 'y')
        emx = f(1, 'x')
        emy = f(1, 'y')
        bbx = f(2, 'x')
        bby = f(2, 'y')
        ebx = f(3, 'x')
        eby = f(3, 'y')
        bmo = f(0, 'orientation')
        emo = f(1, 'orientation')

        b2e_mx = emx-bmx
        b2e_my = emy-bmy
        e2b_bx = bbx-emx
        e2b_by = bby-emy
        b2e_bx = ebx-bmx
        b2e_by = eby-bmy

        b2e_mr = tf_vec_length(b2e_mx, b2e_my)
        e2b_br = tf_vec_length(e2b_bx, e2b_by)
        b2e_br = tf_vec_length(b2e_bx, b2e_by)

        b2e_ma = tf_normed_angle(b2e_mx, b2e_my, bmo)
        e2b_ba = tf_normed_angle(e2b_bx, e2b_by, emo)
        b2e_ba = tf_normed_angle(b2e_bx, b2e_by, bmo)

        bvx = f(0, 'vx')
        bvy = f(0, 'vy')
        evx = f(1, 'vx')
        evy = f(1, 'vy')

        ranges = [
            b2e_mr,
            e2b_br,
            b2e_br,
            tf_vec_length(bvx, bvy),
            tf_vec_length(evx, evy),
            *map(f, RANGE_FEATURES),
        ]
        ranges_tensor = tf.concat(ranges, -1) / 500.0

        angles = [
            b2e_ma,
            e2b_ba,
            b2e_ba,
            tf_normed_angle(bvx, bvy, bmo),
            tf_normed_angle(evx, evy, emo),
            *map(f, ANGLE_FEATURES),
        ]
        angle_tensor_parts = []
        for angle in angles:
            a_interval = 2 * pi / self.angle_sections
            a_normed = angle / a_interval
            for i in range(self.angle_sections):
                a_before = a_normed - i + 1
                a_after = - a_normed + i + 1
                a_tensor_part = tf.maximum(0.0, tf.minimum(a_before, a_after))
                angle_tensor_parts.append(a_tensor_part)
        angles_tensor = tf.concat(angle_tensor_parts, -1)

        inference.input_tensor = tf.concat([
            ranges_tensor,
            angles_tensor,
            *map(f, OTHER_FEATURES),
        ], -1)

        vector = inference.input_tensor
        for layer in self.layers:
            node = layer.apply(vector, tf.sigmoid)
            vector = node.out
            nodes.append(node)
        inference.nodes = nodes
        return vector

Model = ActionInferenceModel


def selector(tensor):
    def select(*feature_name):
        if len(feature_name) == 1 and isinstance(feature_name[0], tuple):
            feature_name = feature_name[0]
        idx = state2vec[feature_name]
        return tensor[..., idx:idx+1]
    return select


def tf_vec_length(x, y):
    return tf.sqrt(tf.square(x) + tf.square(y))


def tf_normed_angle(x, y, rel_a=None):
    a = tf.atan2(y, x)
    if rel_a is not None:
        a -= rel_a
    return a % (2 * pi)