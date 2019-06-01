import tensorflow as tf


class DNN:
    def __init__(
        self, input_dim, stem_units, head_units, heads, scope=None
    ):
        scope_prefix = (
            (scope + "/" if not scope.endswith("/") else scope) if scope else ""
        )
        self.stem = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units, activation)
                for n_units, activation in stem_units
            ],
            name=scope_prefix+"Stem",
        )
        self.controls = {
            head_name: tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(n_units, activation)
                    for n_units, activation in head_units.get(head_name, head_units[None])
                ]
                + [tf.keras.layers.Dense(out_dim, tf.identity)],
                name=scope_prefix+f"Head_{head_name}",
            )
            for head_name, out_dim in heads.items()
        }
        with tf.variable_scope(scope_prefix):
            self.stem.build((None, input_dim))
            control_inputs_dim = stem_units[-1][0] if stem_units else input_dim
            for mdl in self.controls.values():
                mdl.build((None, control_inputs_dim))

    def __call__(self, state):
        logits = {}
        features = self.stem(state)
        for ctl, net in self.controls.items():
            logits[ctl] = net(features)
        return logits


def make_v1(input_dim, heads, scope=None):
    return DNN(
        input_dim=input_dim,
        stem_units=[
            (80, tf.nn.leaky_relu),
            (70, tf.nn.leaky_relu),
            (60, tf.nn.leaky_relu),
            (50, tf.nn.leaky_relu),
        ],
        head_units={None: [(40, tf.nn.leaky_relu)]},
        heads=heads,
        scope=scope,
    )


def make_v2(input_dim, heads, scope=None):
    return DNN(
        input_dim=input_dim,
        stem_units=[(80, tf.nn.leaky_relu), (50, tf.nn.leaky_relu)],
        head_units={None: [(40, tf.nn.leaky_relu)]},
        heads=heads,
        scope=scope,
    )


def make_v3(input_dim, heads, scope=None):
    return DNN(
        input_dim=input_dim,
        stem_units=[(80, tf.nn.leaky_relu), (60, tf.nn.leaky_relu)],
        head_units={None: [(40, tf.nn.leaky_relu), (30, tf.nn.leaky_relu)]},
        heads=heads,
        scope=scope,
    )
