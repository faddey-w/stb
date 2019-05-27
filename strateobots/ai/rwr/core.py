import tensorflow as tf
from strateobots.ai.lib import data


class RewardWeightedRegression:
    def __init__(self, model, batch_size=10, entropy_coeff=None):
        self.model = model
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff  # type: float

        self.state_ph = tf.placeholder(tf.float32, [batch_size, model.state_dimension])
        self.action_idx_ph = {
            ctl: tf.placeholder(
                tf.int32 if data.is_categorical(ctl) else tf.float32, [batch_size]
            )
            for ctl in self.model.control_set
        }
        self.reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.inference = self.model.apply(self.state_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

        self.loss_vectors = {}
        self.entropy = {}
        for ctl in self.model.control_set:
            value = self.inference.controls[ctl]
            if data.is_categorical(ctl):
                value_safe = value - tf.reduce_max(value, axis=1, keepdims=True)
                eps = +0.0001
                value_softmax = eps + (1 - 2 * eps) * tf.nn.softmax(value_safe, axis=1)
                loss = self.reward_ph * tf.log(
                    tf.gather_nd(
                        1 - value_softmax,
                        tf.stack(
                            [tf.range(batch_size), self.action_idx_ph[ctl]], axis=1
                        ),
                    )
                )
                entropy = tf.reduce_mean(-value_softmax * tf.log(value_softmax))
                self.entropy[ctl] = entropy
            else:
                loss = self.reward_ph * tf.square(value - self.action_idx_ph[ctl])

            self.loss_vectors[ctl] = loss
        self.full_entropy = tf.add_n(list(self.entropy.values()))
        self.full_loss_vector = tf.add_n(list(self.loss_vectors.values()))
        self.loss = tf.reduce_mean(self.full_loss_vector)

        loss = self.loss
        if self.entropy_coeff is not None:
            loss -= self.entropy_coeff * self.full_entropy
        self.grads_raw, _ = zip(*self.optimizer.compute_gradients(loss, model.var_list))
        self.grads_clip, self.grads_norm = tf.clip_by_global_norm(self.grads_raw, 20.0)
        self.vars_norm = tf.global_norm(model.var_list)

        self.train_step = self.optimizer.apply_gradients(
            list(zip(self.grads_clip, model.var_list))
        )

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def do_train_step(self, session, replay_memory, batch_index, extra_tensors=()):
        _, extra_results = self.compute_on_sample(
            session, replay_memory, [self.train_step, extra_tensors], batch_index
        )
        return extra_results

    def compute_on_sample(self, session, replay_memory, tensors, batch_index):

        props, actions, states, _ = replay_memory.get_prepared_epoch_batch(batch_index)
        reward = props[..., 0]

        return session.run(
            tensors,
            {
                self.state_ph: states,
                self.reward_ph: reward,
                **{
                    self.action_idx_ph[ctl]: actions[ctl]
                    for ctl in self.model.control_set
                },
            },
        )
