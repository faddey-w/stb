import tensorflow as tf
from strateobots.ai.lib import data


def postprocess_argmax(logits, predictions):
    output_nodes = {}
    for ctl, prediction in predictions.items():
        if ctl in data.CATEGORICAL_CONTROLS:
            prediction = tf.argmax(prediction)
            ctl_feature = data.get_control_feature(ctl)
            prediction = tf.constant(ctl_feature.categories)[prediction]
        else:
            if isinstance(prediction, tf.distributions.Distribution):
                prediction = prediction.mean()
        output_nodes[ctl] = prediction
    return output_nodes


def postprocess_probabilistic(logits, predictions):
    output_nodes = {}
    for ctl, prediction in predictions.items():
        if ctl in data.CATEGORICAL_CONTROLS:
            prediction = tf.random.categorical(logits[ctl], 1)[:, 0]
            ctl_feature = data.get_control_feature(ctl)
            ctl_values = tf.constant(ctl_feature.categories)
            prediction = tf.gather(ctl_values, prediction, axis=0)
        else:
            if isinstance(prediction, tf.distributions.Distribution):
                prediction = prediction.sample()

        output_nodes[ctl] = prediction
    return output_nodes
