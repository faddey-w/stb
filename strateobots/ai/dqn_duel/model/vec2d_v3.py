import tensorflow as tf
from math import pi

from strateobots.ai.lib import layers, stable, util
from strateobots.ai.lib.data import action2vec, state2vec

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


A_ACTIONS = (
    'rotate_left',
    'rotate_no',
    'rotate_right',
    'tower_rotate_left',
    'tower_rotate_no',
    'tower_rotate_right',
)
R_ACTIONS = (
    'move_ahead',
    'move_no',
    'move_back',
)
OTHER_ACTIONS = tuple(
    fn for fn in action2vec.field_names
    if fn not in A_ACTIONS
    if fn not in R_ACTIONS
)


EPS = 0.0001


class QualityFunctionModel:

    def __new__(cls, **kwargs):
        self = super().__new__(cls)
        self.construct_params = kwargs
        return self

    def __init__(self, vec2d_cfg, fc_n_parts, fc_cfg):

        self.vec2d_cfg = tuple(vec2d_cfg)
        self.fc_cfg = tuple(fc_cfg)
        self.fc_n_parts = fc_n_parts
        self.name = 'QFuncVec2dV3'
        self.var_list = []

        with tf.variable_scope(self.name):
            vec_in = len(X_FEATURES + A_FEATURES + R_FEATURES + A_ACTIONS + R_ACTIONS)
            self.vec_layers = []
            vec_layers_flat = []
            for i, (n_lin, n_kat) in enumerate(self.vec2d_cfg):
                lin_x = layers.ResidualV2('VecLinX_{}'.format(i), vec_in, n_lin, n_lin)
                lin_y = layers.ResidualV2('VecLinY_{}'.format(i), vec_in, n_lin, n_lin)
                rot = layers.ResidualV3('VecRot_{}'.format(i), n_lin)

                kat_r = layers.Linear('KatR_{}'.format(i), vec_in, n_kat)
                kat_b = layers.Linear('KatB_{}'.format(i), vec_in, n_kat)

                layer_tuple = lin_x, lin_y, rot, kat_r, kat_b
                self.vec_layers.append(layer_tuple)
                vec_layers_flat.extend(layer_tuple)

                vec_in = n_lin + n_kat

            fc_in = 3 * vec_in + len(OTHER_FEATURES) + action2vec.vector_length
            times_in = fc_in + 2 * vec_in
            self.fc_layers = []
            for i, out_dim in enumerate(fc_cfg):
                if i != 0:
                    fc_in *= 2
                self.fc_layers.append([
                    layers.Linear('Lin{}_{}'.format(i, j), fc_in, out_dim)
                    for j in range(fc_n_parts)
                ])
                fc_in = out_dim

            times_out = fc_n_parts * fc_cfg[-1]
            self.times = layers.Linear('Times', times_in, times_out)
            self.logvalues = layers.Linear('LogValues', 3 * vec_in, 2 * vec_in)

            for lr in [*(lr for lst in self.fc_layers for lr in lst),
                       *vec_layers_flat,
                       self.times,
                       self.logvalues,
                       ]:
                self.var_list.extend(lr.var_list)

    def apply(self, state, action):
        return QualityFunction(self, state, action)


class QualityFunction:
    def __init__(self, model, state, action):
        """
        :param model: QualityFunctionModel
        :param state: [..., state_vector_len]
        :param action: [..., action_vector_len]
        """
        self.model = model  # type: QualityFunctionModel
        self.state = state  # type: tf.Tensor
        self.action = action  # type: tf.Tensor

        state = util.normalize_state(state)
        state += tf.zeros_like(action[..., :1])

        x0 = select_features(state, state2vec, *X_FEATURES)
        y0 = select_features(state, state2vec, *Y_FEATURES)
        r0 = tf.concat([
            tf.sqrt(tf.square(x0) + tf.square(y0)),
            select_features(state, state2vec, *R_FEATURES),
            tf.ones_like(select_features(state, state2vec, *A_FEATURES)),
            tf.ones_like(select_features(action, action2vec, *A_ACTIONS)),
            select_features(action, action2vec, *R_ACTIONS),
        ], -1)
        a0 = tf.concat([
            tf.atan2(y0, x0),
            tf.zeros_like(select_features(state, state2vec, *R_FEATURES)),
            select_features(state, state2vec, *A_FEATURES),
            select_features(state, state2vec, *A_FEATURES),
            tf.zeros_like(select_features(action, action2vec, *R_ACTIONS)),
        ], -1)

        r, a = r0, a0
        self.vec_nodes = []
        for i, (lin_x, lin_y, rot, kat_r, kat_b) in enumerate(model.vec_layers):
            # a = (a+pi)%(2*pi) - pi
            x = r * tf.cos(a)
            y = r * tf.sin(a)

            xln = lin_x.apply(x, tf.identity)
            yln = lin_y.apply(y, tf.identity)

            rlt = tf.sqrt(tf.square(xln.out) + tf.square(yln.out))
            alt = tf.asin(yln.out / (rlt + EPS))
            aln = rot.apply(alt, tf.nn.relu)

            rkn = kat_r.apply(r, tf.identity)
            bkn = kat_b.apply(r, lambda x: tf.nn.relu(x) + EPS)
            rkt = tf.clip_by_value(rkn.out, EPS, bkn.out)
            akt = tf.asin(rkt / (bkn.out + EPS))

            r = tf.concat([rlt, bkn.out], -1)
            a = tf.concat([aln.out, akt], -1)

            self.vec_nodes.append((xln, yln, aln, rkn, bkn))

        vector = tf.concat([
            r,
            a,
            tf.cos(a),
            select_features(state, state2vec, *OTHER_FEATURES),
            action,
        ], -1)

        self.logvalues = model.logvalues.apply(
            tf.concat([r, a, tf.cos(a)], -1),
            lambda x: tf.log(tf.maximum(1.0, x)),
        )
        self.times = model.times.apply(
            tf.concat([vector, self.logvalues.out], -1),
            tf.nn.relu,
        )

        self.fc_layers = []
        vector_array = [vector]
        for node_list in model.fc_layers:
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
            self.fc_layers.append(out_array)
        vector = tf.concat(vector_array, -1)

        self.features = vector * tf.exp(-self.times.out)
        finite_assert = tf.Assert(
            tf.reduce_all(tf.is_finite(self.features)),
            [tf.reduce_all(tf.is_finite(v)) for v in model.var_list],
        )
        with tf.control_dependencies([finite_assert]):
            self.quality = tf.reduce_mean(self.features, axis=-1)

    def get_quality(self):
        return self.quality

    def call(self, state, action, session):
        return session.run(self.quality, feed_dict={
            self.state: state,
            self.action: action,
        })
Model = QualityFunctionModel


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx:idx+1])
    return tf.concat(feature_tensors, -1)