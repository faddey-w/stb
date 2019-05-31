import tensorflow as tf
import numpy as np
from strateobots.ai.lib import data


class DNN:
    def __init__(
        self, stem_units, head_units, heads, scope=None
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

    def __call__(self, state):
        logits = {}
        features = self.stem(state)
        for ctl, net in self.controls.items():
            # head = net(features)
            # if ctl in data.CATEGORICAL_CONTROLS:
            #     logits[ctl] = head
            #     head = tf.nn.softmax(head)
            # else:
            #     mean = head[..., 0]
            #     stddev = tf.nn.softplus(head[..., 1]) + 0.01
            #     logits[ctl + "_mean"] = mean
            #     logits[ctl + "_std"] = stddev
            #     head = tf.distributions.Normal(mean, stddev)
            logits[ctl] = net(features)
        return logits
        # return logits, predictions


def make_v1(heads, scope=None):
    return DNN(
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


def make_v2(heads, scope=None):
    return DNN(
        stem_units=[(80, tf.nn.leaky_relu), (50, tf.nn.leaky_relu)],
        head_units={None: [(40, tf.nn.leaky_relu)]},
        heads=heads,
        scope=scope,
    )


def make_v3(heads, scope=None):
    return DNN(
        stem_units=[(80, tf.nn.leaky_relu), (60, tf.nn.leaky_relu)],
        head_units={None: [(40, tf.nn.leaky_relu), (30, tf.nn.leaky_relu)]},
        heads=heads,
        scope=scope,
    )
