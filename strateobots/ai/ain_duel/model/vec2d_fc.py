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
OTHER_FEATURES = tuple(
    fn for fn in state2vec.field_names
    if fn not in X_FEATURES
    if fn not in Y_FEATURES
)


class ActionInferenceModel(BaseActionInferenceModel):

    def _create_layers(self, coord_cfg, angle_cfg, fc_cfg, n_residual=0):
        assert len(coord_cfg) == len(angle_cfg)
        assert len(fc_cfg) >= n_residual

        self.vec2d_cfg = tuple(zip(coord_cfg, angle_cfg))
        self.fc_cfg = tuple(fc_cfg)
        self.n_residual = n_residual

        self.vec2d_layers = []
        self.fc_layers = []

        vec2d_layers_flat = []

        coord_in = len(X_FEATURES) + len(OTHER_FEATURES)
        angle_in = state2vec.vector_length
        for i, (coord_out, angle_out) in enumerate(self.vec2d_cfg):
            lx = layers.Linear('L{}x'.format(i), coord_in, coord_out)
            ly = layers.Linear('L{}y'.format(i), coord_in, coord_out, lx.weight)
            la = layers.ResidualV2('L{}a'.format(i), coord_in + angle_in, angle_out // 2, angle_out)
            self.vec2d_layers.append((lx, ly, la))
            vec2d_layers_flat.extend((lx, ly, la))
            coord_in, angle_in = coord_out, angle_out

        fc_in = 3 * coord_in + angle_in
        for i, fc_out in enumerate(self.fc_cfg):
            fc = layers.Linear('FC{}'.format(i), fc_in, fc_out)
            self.fc_layers.append(fc)
            fc_in = fc_out
        return fc_in, vec2d_layers_flat + self.fc_layers

    def create_inference(self, inference, state):

        inference.x0 = select_features(
            state, state2vec,
            *X_FEATURES,
            *OTHER_FEATURES,
        )
        inference.y0 = tf.concat([
            select_features(state, state2vec, *Y_FEATURES),
            tf.ones_like(state[..., :len(OTHER_FEATURES)])
        ], -1)
        inference.a0 = tf.concat([
            state,
            # stable.atan2(inference.y0, inference.x0)
            tf.atan2(inference.y0, inference.x0)
        ], -1)

        inference.vec2d_levels = []
        vectors = (inference.x0, inference.y0, inference.a0)
        for mx, my, ma in self.vec2d_layers:
            x, y, a = vectors
            # lx = mx.apply(x, make_activation(mx.out_dim))
            # ly = my.apply(y, make_activation(my.out_dim))
            # la = ma.apply(a, make_activation(ma.out_dim, angle=True))
            lx = mx.apply(x, tf.identity)
            ly = my.apply(y, tf.identity)
            la = ma.apply(a, tf.nn.relu)
            angles = la.out[..., :lx.out.get_shape().as_list()[-1]]
            a_cos = tf.cos(angles)
            a_sin = tf.sin(angles)
            new_x = lx.out * a_cos - ly.out * a_sin
            new_y = lx.out * a_sin + ly.out * a_cos
            # add_a = stable.atan2(ly.out, lx.out)
            add_a = tf.atan2(ly.out, lx.out)
            new_a = tf.concat([la.out, add_a], -1)
            inference.vec2d_levels.append((lx, ly, la, (new_x, new_y)))
            vectors = (new_x, new_y, new_a)

        fc_vec = tf.concat(vectors, -1)
        inference.fc_layers = []
        for i, fc in enumerate(self.fc_layers):
            # fc_lr = fc.apply(fc_vec, make_activation(fc.out_dim, tf.sigmoid))
            fc_lr = fc.apply(fc_vec, tf.sigmoid)
            inference.fc_layers.append(fc_lr)
            fc_vec = fc_lr.out

        return fc_vec

Model = ActionInferenceModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)
