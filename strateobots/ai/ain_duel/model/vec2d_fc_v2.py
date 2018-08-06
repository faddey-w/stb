import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers, stable
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


class ActionInferenceModel(BaseActionInferenceModel):

    def _create_layers(self, vec_cfg, fc_cfg):
        self.fc_cfg = tuple(fc_cfg)
        self.vec_cfg = tuple(map(tuple, vec_cfg))

        vec_in = len(X_FEATURES) + len(A_FEATURES) + len(R_FEATURES)
        self.vec_layers = []
        vec_layers_flat = []
        for i, (n_lin, n_kat) in enumerate(self.vec_cfg):
            lin_x = layers.Linear('VecLinX_{}'.format(i), vec_in, n_lin)
            lin_y = layers.Linear('VecLinY_{}'.format(i), vec_in, n_lin)
            rot = layers.ResidualV3('VecRot_{}'.format(i), n_lin)

            kat_r = layers.Linear('KatR_{}'.format(i), vec_in, n_kat)
            kat_b = layers.Linear('KatB_{}'.format(i), vec_in, n_kat)

            layer_tuple = lin_x, lin_y, rot, kat_r, kat_b
            self.vec_layers.append(layer_tuple)
            vec_layers_flat.extend(layer_tuple)

            vec_in = n_lin + n_kat

        fc_in = 2 * vec_in + len(OTHER_FEATURES)
        self.fc_layers = []
        for i, fc_out in enumerate(self.fc_cfg):
            # if fc_in == fc_out:
            #     fc = layers.ResidualV3('FC{}'.format(i), fc_out)
            # else:
            #     fc = layers.Residual('FC{}'.format(i), fc_in, fc_out)
            # fc = layers.Residual('FC{}'.format(i), fc_in, fc_out)
            fc = layers.Linear('FC{}'.format(i), fc_in, fc_out)
            self.fc_layers.append(fc)
            fc_in = fc_out
        return fc_in, [*vec_layers_flat, *self.fc_layers]

    def create_inference(self, inference, state):

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

        r, a = r0, a0
        inference.vec_nodes = []
        for i, (lin_x, lin_y, rot, kat_r, kat_b) in enumerate(self.vec_layers):
            # a = (a+pi)%(2*pi) - pi
            x = r * tf.cos(a)
            y = r * tf.sin(a)

            xln = lin_x.apply(x, tf.identity)
            yln = lin_y.apply(y, tf.identity)

            rlt = tf.sqrt(tf.square(xln.out) + tf.square(yln.out) + EPS**2)
            alt = tf.asin(yln.out / (rlt + EPS))
            aln = rot.apply(alt, tf.nn.relu)

            rkn = kat_r.apply(r, tf.identity)
            bkn = kat_b.apply(r, lambda x: tf.nn.relu(x) + EPS)
            rkt = tf.clip_by_value(rkn.out, EPS, bkn.out)
            akt = tf.asin(rkt / (bkn.out+EPS))

            # sensitive_tensors = rlt, bkn.out, aln.out, akt
            # informative_tensors = rlt, bkn.out, aln.out, akt, rkt, bkn.out
            # finite_assert = tf.Assert(
            #     tf.reduce_all(tf.is_finite(tf.concat(sensitive_tensors, -1))),
            #     [tf.reduce_all(tf.is_finite(t)) for t in informative_tensors],
            #     name="FiniteAssert{}".format(i)
            # )
            #
            # with tf.control_dependencies([finite_assert]):
            #     r = tf.concat([rlt, bkn.out], -1)
            #     a = tf.concat([aln.out, akt], -1)
            r = tf.concat([rlt, bkn.out], -1)
            a = tf.concat([aln.out, akt], -1)

            inference.vec_nodes.append((xln, yln, aln, rkn, bkn))

        vector = tf.concat([
            r,
            tf.cos(a),
            select_features(state, state2vec, *OTHER_FEATURES),
        ], -1)
        inference.fc_layers = []
        for i, fc in enumerate(self.fc_layers):
            # fc_lr = fc.apply(vector, tf.nn.relu)
            fc_lr = fc.apply(vector, lambda x: 2 * tf.sigmoid(x) - 1)
            inference.fc_layers.append(fc_lr)
            vector = fc_lr.out

        return vector

Model = ActionInferenceModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
