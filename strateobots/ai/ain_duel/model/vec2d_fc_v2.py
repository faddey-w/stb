import tensorflow as tf
from math import pi
from collections import namedtuple

from strateobots.ai.lib import layers
from strateobots.ai.lib.data import state2vec
from strateobots.ai.ain_duel.model.base import normalize_state, finite_assert, combine_predictions

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
A_FEATURES = (
    (0, 'orientation'),
    (0, 'tower_orientation'),
    (1, 'orientation'),
    (1, 'tower_orientation'),
    (2, 'orientation'),
    (3, 'orientation'),
)
R_FEATURES = (
    (2, 'remaining_range'),
    (3, 'remaining_range'),
)
OTHER_FEATURES = tuple(
    fn for fn in state2vec.field_names
    if fn not in X_FEATURES
    if fn not in Y_FEATURES
    if fn not in A_FEATURES
    if fn not in R_FEATURES
)
assert len(X_FEATURES) == len(Y_FEATURES)
EPS = 0.0001


class ActionInferenceModel:

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self,
                 move_vec, move_fc,
                 rotate_vec, rotate_fc,
                 tower_rotate_vec, tower_rotate_fc,
                 fire_vec, fire_fc,
                 shield_vec, shield_fc):
        self.name = 'AIN'

        with tf.variable_scope(self.name):
            self.move_net = Vec2dV2Network('move', move_vec, move_fc)
            self.rotate_net = Vec2dV2Network('rotate', rotate_vec, rotate_fc)
            self.tower_rotate_net = Vec2dV2Network('tower_rotate', tower_rotate_vec, tower_rotate_fc)
            self.fire_net = Vec2dV2Network('fire', fire_vec, fire_fc)
            self.shield_net = Vec2dV2Network('shield', shield_vec, shield_fc)

            self.move_last = layers.Linear('move_last', self.move_net.ff_chain.out_size, 3)
            self.rotate_last = layers.Linear('rotate_last', self.rotate_net.ff_chain.out_size, 3)
            self.tower_rotate_last = layers.Linear('tower_rotate_last', self.tower_rotate_net.ff_chain.out_size, 3)
            self.fire_last = layers.Linear('fire_last', self.fire_net.ff_chain.out_size, 1)
            self.shield_last = layers.Linear('shield_last', self.shield_net.ff_chain.out_size, 1)

        self.var_list = [
            *self.move_net.var_list,
            *self.rotate_net.var_list,
            *self.tower_rotate_net.var_list,
            *self.fire_net.var_list,
            *self.shield_net.var_list,

            *self.move_last.var_list,
            *self.rotate_last.var_list,
            *self.tower_rotate_last.var_list,
            *self.fire_last.var_list,
            *self.shield_last.var_list,
        ]

    def apply(self, state):
        state = normalize_state(state)
        classify = lambda x: tf.nn.softmax(x, -1)
        inference = Inference()
        inference.model = self
        inference.state = state
        inference.action_prediction = combine_predictions(
            move=self.move_last.apply(self.move_net.apply(state)[-1], classify).out,
            rotate=self.rotate_last.apply(self.rotate_net.apply(state)[-1], classify).out,
            tower_rotate=self.tower_rotate_last.apply(self.tower_rotate_net.apply(state)[-1], classify).out,
            fire=self.fire_last.apply(self.fire_net.apply(state)[-1], tf.sigmoid).out,
            shield=self.shield_last.apply(self.shield_net.apply(state)[-1], tf.sigmoid).out,
        )
        return inference


class Inference:
    pass


class Vec2dV2Network:

    def __init__(self, name, vec_cfg, fc_cfg):
        self.fc_cfg = tuple(fc_cfg)
        self.vec_cfg = tuple(map(tuple, vec_cfg))

        vec_in = len(X_FEATURES) + len(A_FEATURES) + len(R_FEATURES)
        with tf.variable_scope(name):
            self.vec2dv2 = Vec2dV2Chain('Vec2dV2', vec_in, vec_cfg)

            fc_in = 2 * self.vec2dv2.n_out + len(OTHER_FEATURES)
            self.ff_chain = LinearFFChain('FF', fc_in, fc_cfg)

    @property
    def var_list(self):
        return [*self.vec2dv2.var_list, *self.ff_chain.var_list]

    def apply(self, state):

        r0, a0, f0 = make_input_tensors(state)
        vec2dv2_nodes, out_r, out_a = self.vec2dv2.apply(r0, a0)

        vector = tf.concat([out_r, tf.cos(out_a), f0], -1)

        ff_nodes, features = self.ff_chain.apply(vector, tf.sigmoid)

        with finite_assert(features, self.var_list):
            features = tf.identity(features)

        return vec2dv2_nodes, ff_nodes, features

Model = ActionInferenceModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)


class Vec2dV2Layer:
    def __init__(self, name, vec_in, n_lin, n_kat):
        self.name = name
        with tf.variable_scope(name):
            self.lin_x = layers.Linear('LinX', vec_in, n_lin)
            self.lin_y = layers.Linear('LinY', vec_in, n_lin)
            self.rot = layers.ResidualV3('Rot', n_lin)

            self.kat_r = layers.Linear('KatR', vec_in, n_kat)
            self.kat_b = layers.Linear('KatB', vec_in, n_kat)

    @property
    def var_list(self):
        return [*self.lin_x.var_list,
                *self.lin_y.var_list,
                *self.rot.var_list,
                *self.kat_r.var_list,
                *self.kat_b.var_list]

    _Apply = namedtuple('_Apply', 'xln yln aln rkn bkn out_r out_a')

    def apply(self, r, a):
        x = r * tf.cos(a)
        y = r * tf.sin(a)

        xln = self.lin_x.apply(x, tf.identity)
        yln = self.lin_y.apply(y, tf.identity)

        rlt = tf.sqrt(tf.square(xln.out) + tf.square(yln.out) + EPS ** 2)
        alt = tf.asin(yln.out / (rlt + EPS))
        aln = self.rot.apply(alt, tf.nn.relu)

        rkn = self.kat_r.apply(r, tf.identity)
        bkn = self.kat_b.apply(r, lambda x: tf.nn.relu(x) + EPS)
        rkt = tf.clip_by_value(rkn.out, EPS, bkn.out)
        akt = tf.asin(rkt / (bkn.out + EPS))

        r = tf.concat([rlt, bkn.out], -1)
        a = tf.concat([aln.out, akt], -1)

        return self._Apply(xln, yln, aln, rkn, bkn, r, a)


class Vec2dV2Chain:
    def __init__(self, name, n_in, cfg):
        self.name = name
        self.layers = []
        with tf.variable_scope(name):
            for i, (n_lin, n_kat) in enumerate(cfg, 1):
                layer = Vec2dV2Layer(str(i), n_in, n_lin, n_kat)
                n_in = n_lin + n_kat
                self.layers.append(layer)
        self.n_out = n_in

    @property
    def var_list(self):
        return [v for layer in self.layers for v in layer.var_list]

    def apply(self, r, a):
        nodes = []
        for layer in self.layers:
            node = layer.apply(r, a)
            r, a = node.out_r, node.out_a
            nodes.append(node)
        return nodes, r, a


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


def make_input_tensors(state):
    x0 = select_features(state, state2vec, *X_FEATURES)
    y0 = select_features(state, state2vec, *Y_FEATURES)
    r0 = tf.concat([
        tf.sqrt(tf.square(x0) + tf.square(y0)),
        select_features(state, state2vec, *R_FEATURES),
        tf.ones_like(select_features(state, state2vec, *A_FEATURES))
    ], -1)
    a0 = tf.concat([
        tf.atan2(y0, x0),
        tf.zeros_like(select_features(state, state2vec, *R_FEATURES)),
        select_features(state, state2vec, *A_FEATURES)
    ], -1)
    f0 = select_features(state, state2vec, *OTHER_FEATURES)
    return r0, a0, f0
