import tensorflow as tf
from strateobots.ai.lib import data


class DNN:

    def __init__(self, stem_units, control_units):
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
                ] + [
                    tf.keras.layers.Dense(
                        data.get_control_feature(ctl).dimension,
                        tf.identity
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

