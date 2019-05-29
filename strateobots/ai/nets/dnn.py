import tensorflow as tf
from strateobots.ai.lib import data


class DNN:
    def __init__(self, stem_units, control_units, continuous_activations=None):
        continuous_activations = continuous_activations or {}
        self.stem = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units, activation)
                for n_units, activation in stem_units
            ],
            name="Stem",
        )
        self.controls = {
            ctl: tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(n_units, activation)
                    for n_units, activation in ctl_units
                ]
                + [
                    tf.keras.layers.Dense(
                        data.get_control_feature(ctl).dimension
                        if ctl in data.CATEGORICAL_CONTROLS
                        else 2,  # 2 is mean and std-dev
                        continuous_activations.get(ctl, tf.identity),
                    )
                ],
                name=f"Head_{ctl}",
            )
            for ctl, ctl_units in control_units.items()
        }

    def __call__(self, state):
        logits = {}
        predictions = {}
        features = self.stem(state)
        for ctl, net in self.controls.items():
            head = net(features)
            if ctl in data.CATEGORICAL_CONTROLS:
                logits[ctl] = head
                head = tf.nn.softmax(head)
            else:
                mean = head[..., 0]
                stddev = tf.nn.softplus(head[..., 1])
                logits[ctl + "_mean"] = mean
                logits[ctl + "_std"] = stddev
                head = tf.distributions.Normal(mean, stddev)
            predictions[ctl] = head
        return logits, predictions


def make_v1(controls):
    return DNN(
        stem_units=[
            (80, tf.nn.leaky_relu),
            (70, tf.nn.leaky_relu),
            (60, tf.nn.leaky_relu),
            (50, tf.nn.leaky_relu),
        ],
        control_units={ctl: [(40, tf.nn.leaky_relu)] for ctl in controls},
    )


def make_v2(controls):

    def angle_activation(x):
        mean, stddev = tf.unstack(x, axis=-1)
        mean = tf.atan(mean) * 2  # -pi .. +pi
        return tf.stack([mean, stddev], axis=-1)

    return DNN(
        stem_units=[
            (80, tf.nn.leaky_relu),
            (50, tf.nn.leaky_relu),
        ],
        control_units={ctl: [(40, tf.nn.leaky_relu)] for ctl in controls},
        continuous_activations={
            "orientation": angle_activation,
            "tower_orientation": angle_activation,
        }
    )
