import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import state2vec
from strateobots.ai.ain_duel.model.base import BaseActionInferenceModelV2

ANGLE_FEATURES = (
    (0, "orientation"),
    (0, "tower_orientation"),
    (1, "orientation"),
    (1, "tower_orientation"),
    (2, "orientation"),
    (3, "orientation"),
)
RANGE_FEATURES = (
    (0, "x"),
    (0, "y"),
    (1, "x"),
    (1, "y"),
    (2, "remaining_range"),
    (3, "remaining_range"),
)
OTHER_FEATURES = (
    (0, "hp_ratio"),
    (0, "load"),
    (0, "shield_ratio"),
    (1, "hp_ratio"),
    (1, "load"),
    (1, "shield_ratio"),
    (2, "present"),
    (3, "present"),
)


class ActionInferenceModel(BaseActionInferenceModelV2):
    def _create_common_net(self):
        return _DataTransform

    def _create_rotate_net(self, layer_sizes, n_angles):
        return ClassicV2Net("Rotate", layer_sizes, n_angles)

    def _create_tower_rotate_net(self, layer_sizes, n_angles):
        return ClassicV2Net("TowerRotate", layer_sizes, n_angles)

    def _create_fire_net(self, layer_sizes, n_angles):
        return ClassicV2Net("Fire", layer_sizes, n_angles)

    def _create_shield_net(self, layer_sizes, n_angles):
        return ClassicV2Net("Shield", layer_sizes, n_angles)

    def _create_move_net(self, layer_sizes, n_angles):
        return ClassicV2Net("Move", layer_sizes, n_angles)


Model = ActionInferenceModel


def selector(tensor):
    def select(*feature_name):
        if len(feature_name) == 1 and isinstance(feature_name[0], tuple):
            feature_name = feature_name[0]
        idx = state2vec[feature_name]
        return tensor[..., idx : idx + 1]

    return select


def tf_vec_length(x, y):
    return tf.sqrt(tf.square(x) + tf.square(y))


def tf_normed_angle(x, y, rel_a=None):
    a = tf.atan2(y, x)
    if rel_a is not None:
        a -= rel_a
    return a % (2 * pi)


class LinearFFChain:
    def __init__(self, name, in_size, layer_sizes):
        self.name = name
        self.layers = []
        with tf.variable_scope(name):
            for i, out_size in enumerate(layer_sizes, 1):
                fc = layers.Linear(str(i), in_size, out_size)
                self.layers.append(fc)
                in_size = out_size
        self.out_size = in_size

    @property
    def var_list(self):
        return [v for layer in self.layers for v in layer.var_list]

    def apply(self, vector, activation):
        nodes = []
        for layer in self.layers:
            node = layer.apply(vector, activation)
            nodes.append(node)
            vector = node.out
        return nodes, vector


class _DataTransform:
    var_list = []

    def __init__(self, state):

        f = selector(state)

        bmx = f(0, "x")
        bmy = f(0, "y")
        emx = f(1, "x")
        emy = f(1, "y")
        bbx = f(2, "x")
        bby = f(2, "y")
        ebx = f(3, "x")
        eby = f(3, "y")
        bmo = f(0, "orientation")
        emo = f(1, "orientation")

        b2e_mx = emx - bmx
        b2e_my = emy - bmy
        e2b_bx = bbx - emx
        e2b_by = bby - emy
        b2e_bx = ebx - bmx
        b2e_by = eby - bmy

        b2e_mr = tf_vec_length(b2e_mx, b2e_my)
        e2b_br = tf_vec_length(e2b_bx, e2b_by)
        b2e_br = tf_vec_length(b2e_bx, b2e_by)

        b2e_ma = tf_normed_angle(b2e_mx, b2e_my, bmo)
        e2b_ba = tf_normed_angle(e2b_bx, e2b_by, emo)
        b2e_ba = tf_normed_angle(b2e_bx, b2e_by, bmo)

        bvx = f(0, "vx")
        bvy = f(0, "vy")
        evx = f(1, "vx")
        evy = f(1, "vy")

        ranges = [
            b2e_mr,
            e2b_br,
            b2e_br,
            tf_vec_length(bvx, bvy),
            tf_vec_length(evx, evy),
            *map(f, RANGE_FEATURES),
        ]
        self.ranges_tensor = tf.concat(ranges, -1) / 500.0

        self.angles_in = tf.concat(
            [
                b2e_ma,
                e2b_ba,
                b2e_ba,
                tf_normed_angle(bvx, bvy, bmo),
                tf_normed_angle(evx, evy, emo),
                *map(f, ANGLE_FEATURES),
            ],
            -1,
        )
        self.other_features = list(map(f, OTHER_FEATURES))

    @classmethod
    def apply(cls, state):
        return cls(state)


class Node:
    pass


class ClassicV2Net:
    def __init__(self, name, layer_sizes, n_angles):
        self.name = name
        angles = (
            +1  # vector from bot to enemy
            + 1  # vector from enemy to our bullet
            + 1  # vector from bot to enemy's bullet
            + 2  # bot velocity vectors
            + len(ANGLE_FEATURES)
        )
        in_dim = (
            +1  # vector from bot to enemy
            + 1  # vector from enemy to our bullet
            + 1  # vector from bot to enemy's bullet
            + 2  # bot velocity vectors
            + n_angles
            + len(RANGE_FEATURES)
            + len(OTHER_FEATURES)
        )
        with tf.variable_scope(name):
            self.angle_func = layers.Linear("Angles", angles, n_angles)
            self.ff_chain = LinearFFChain("FC", in_dim, layer_sizes)
        self.var_list = self.angle_func.var_list + self.ff_chain.var_list
        self.n_features = self.ff_chain.out_size

    def apply(self, data):
        angles_node = self.angle_func.apply(data.angles_in, tf.identity)

        inference = Node()
        vector = tf.concat(
            [data.ranges_tensor, tf.cos(angles_node.out), *data.other_features], -1
        )
        inference.input_tensor = vector

        inference.fc, inference.features = self.ff_chain.apply(vector, tf.sigmoid)
        return inference
