import tensorflow as tf
import numpy as np
from strateobots.ai.lib import data, data_encoding, util


class ControlOutput:
    def __init__(self, control, state, logits_dict):
        self.control = control
        self.logits_dict = logits_dict
        self.state = state

    def choice(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    @staticmethod
    def heads(control):
        raise NotImplementedError


class CategoricalControlOutput(ControlOutput):
    def __init__(self, control, state, logits_dict):
        super(CategoricalControlOutput, self).__init__(control, state, logits_dict)
        self.n_categories = data.get_control_feature(self.control).dimension
        self._logits = self.logits_dict[self.control]

    def choice(self):
        return tf.argmax(self._logits, axis=-1)

    def sample(self):
        return tf.random.categorical(self._logits, 1)[:, 0]

    def entropy(self):
        prob = tf.nn.softmax(self._logits)
        log_prob = tf.nn.log_softmax(self._logits)
        return -prob * log_prob

    def log_prob(self, category_index):
        onehot = tf.one_hot(category_index, self.n_categories)
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=onehot, logits=self._logits
        )

    @staticmethod
    def heads(control):
        return {control: data.get_control_feature(control).dimension}


class ScalarOutput(ControlOutput):
    def __init__(self, control, state, logits_dict, min_value=None, max_value=None):
        super(ScalarOutput, self).__init__(control, state, logits_dict)
        self.min_value = min_value
        self.max_value = max_value
        self.mean = logits_dict[self.control + "_mean"][:, 0]
        self.std = logits_dict[self.control + "_std"][:, 0]
        self.std = tf.nn.softplus(self.std) + 0.2
        self._distr = tf.distributions.Normal(self.mean, self.std)

    @classmethod
    def Bounded(cls, min_value=None, max_value=None):
        def factory(control, state, logits_dict):
            return cls(control, state, logits_dict, min_value, max_value)

        factory.heads = cls.heads
        return factory

    def choice(self):
        return self._clip(self._distr.mean()[:, 0])

    def sample(self):
        return self._clip(self._distr.sample(1)[0])

    def entropy(self):
        return util.assert_finite(self._distr.entropy(), [self.control])

    def log_prob(self, x):
        return self._distr.log_prob(x)

    @staticmethod
    def heads(control):
        return {control + "_mean": 1, control + "_std": 1}

    def _clip(self, x):
        if self.min_value is not None:
            x = tf.maximum(x, self.min_value)
        if self.max_value is not None:
            x = tf.minimum(x, self.max_value)
        return x


class OrientationOutput(ScalarOutput):
    def __init__(self, control, state, logits_dict):
        assert control in ("gun_orientation", "orientation")
        super(OrientationOutput, self).__init__(
            control, state, logits_dict, -np.pi + 1e-3, np.pi - 1e-3
        )
        self.is_for_gun = control == "gun_orientation"
        ori, tow_ori = data_encoding.extract_orientations(state)[:2]
        self.base_angle = tow_ori if self.is_for_gun else ori

    @classmethod
    def Bounded(cls, min_value=None, max_value=None):
        raise TypeError

    def choice(self):
        x = super(OrientationOutput, self).choice()
        return self.base_angle + x

    def sample(self):
        x = super(OrientationOutput, self).sample()
        return self.base_angle + x

    def log_prob(self, x):
        return super(OrientationOutput, self).log_prob(x - self.base_angle)


def build_network(scope, network_factory, control_outputs):
    all_heads = {}
    for ctl, ctl_cls in control_outputs.items():
        all_heads.update(ctl_cls.heads(ctl))

    net = network_factory(all_heads, scope)
    return net


def build_inference(network, state, control_outputs):
    logits_dict = network(state)
    return {
        ctl: ctl_cls(ctl, state, logits_dict)
        for ctl, ctl_cls in control_outputs.items()
    }
