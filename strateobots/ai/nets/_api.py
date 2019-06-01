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

    def logits(self):
        return self._logits

    def probabilities(self):
        return tf.nn.softmax(self._logits)

    def choice(self):
        return tf.argmax(self._logits, axis=-1)

    def sample(self):
        return tf.random.categorical(self._logits, 1)[:, 0]

    def entropy(self):
        prob = tf.nn.softmax(self._logits)
        log_prob = tf.nn.log_softmax(self._logits)
        return -prob * log_prob

    def log_prob(self, category_index):
        max_logit = tf.maximum(0.0, tf.reduce_max(self._logits, axis=1, keep_dims=True))
        exp_safe = tf.exp(self._logits - max_logit)
        expsum = tf.reduce_sum(exp_safe, axis=1)
        batch_range = tf.range(tf.shape(self._logits)[0])
        gathering_indices = tf.stack([tf.cast(batch_range, tf.int32), tf.cast(category_index, tf.int32)], axis=1)
        tgt_logit = tf.gather_nd(self._logits, gathering_indices)
        logprob = tgt_logit - tf.log(expsum) - tf.squeeze(max_logit, axis=1)
        return logprob

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
        self.std = tf.nn.softplus(self.std) + 0.09
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


class OrientationOutput(ControlOutput):
    def __init__(self, control, state, logits_dict):
        assert control in ("gun_orientation", "orientation")
        super(OrientationOutput, self).__init__(control, state, logits_dict)

        diff_logit = logits_dict[self.control + "_diff"][:, 0]
        base_logit = logits_dict[self.control + "_base"][:, 0]

        # generate an unimodal beta-distribution
        base = tf.nn.softplus(base_logit) + 1e-3
        a = 1 + base + tf.nn.softplus(diff_logit)
        b = 1 + base + tf.nn.softplus(-diff_logit)

        self._distr = tf.distributions.Beta(a, b)
        self.is_for_gun = control == "gun_orientation"
        ori, tow_ori = data_encoding.extract_orientations(state)[:2]
        self.base_angle = tow_ori if self.is_for_gun else ori

    def choice(self):
        return self._to_angle(self._distr.mean())

    def sample(self):
        return self._to_angle(self._distr.sample(1)[0])

    def entropy(self):
        return util.assert_finite(self._distr.entropy(), [self.control])

    def log_prob(self, x):
        rel_angle = x - self.base_angle + np.pi
        rel_angle = tf.clip_by_value(rel_angle, 1e-2, 2 * np.pi - 1e-2)
        beta_value = rel_angle / (2 * np.pi)
        return self._distr.log_prob(beta_value)

    @staticmethod
    def heads(control):
        return {control + "_diff": 1, control + "_base": 1}

    def _to_angle(self, x):
        return self.base_angle + 2 * np.pi * (1 - 1e-3) * (x - 0.5)


def build_network(scope, network_factory, input_dim, control_outputs):
    all_heads = {}
    for ctl, ctl_cls in control_outputs.items():
        all_heads.update(ctl_cls.heads(ctl))

    net = network_factory(input_dim, all_heads, scope)
    return net


def build_inference(network, state, control_outputs):
    logits_dict = network(state)
    return {
        ctl: ctl_cls(ctl, state, logits_dict)
        for ctl, ctl_cls in control_outputs.items()
    }


class AnglenavModel:
    def __init__(self, network_factory, input_dim, scope=None):
        self.control_model = {
            "move": CategoricalControlOutput,
            "action": CategoricalControlOutput,
            "orientation": OrientationOutput,
            "gun_orientation": OrientationOutput,
        }
        self.policy_net = build_network(
            util.make_scope(scope, "Actor"), network_factory, input_dim, self.control_model
        )
        self.value_net = network_factory(
            input_dim, {"value": 1}, scope=util.make_scope(scope, "Critic")
        )

    def __call__(self, state_vectors):
        ctl_outs = build_inference(self.policy_net, state_vectors, self.control_model)
        value_tensor = self.value_net(state_vectors)["value"][:, 0]
        return value_tensor, ctl_outs
