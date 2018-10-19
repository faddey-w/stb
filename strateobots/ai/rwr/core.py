import numpy as np
import tensorflow as tf

from strateobots.ai.lib.data import state2vec


class RewardWeightedRegression:

    def __init__(self, model, batch_size=10):
        self.model = model
        self.batch_size = batch_size

        self.state_ph = tf.placeholder(tf.float32, [batch_size, state2vec.vector_length])
        self.action_idx_ph = tf.placeholder(tf.int32, [batch_size])
        self.reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.qfunc = self.model.apply(self.state_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

        # self.entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.action_idx_ph,
        #     logits=self.qfunc.quality,
        # )
        # self.loss_vector = - self.reward_ph * self.entropy
        quality_safe = self.qfunc.quality - tf.reduce_max(
            self.qfunc.quality,
            axis=1, keepdims=True
        )
        self.loss_vector = self.reward_ph * tf.log(
            tf.clip_by_value(tf.gather_nd(
                1-tf.nn.softmax(quality_safe, axis=1),
                tf.stack([tf.range(batch_size), self.action_idx_ph], axis=1)
            ), 0.00001, 0.99999)
        )
        self.loss = tf.reduce_mean(self.loss_vector)

        self.train_step = self.optimizer.minimize(self.loss, var_list=model.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def do_train_step(self, session, replay_memory, extra_tensors=(),
                      **sampling_kwargs):
        _, extra_results = self.compute_on_sample(
            session,
            replay_memory,
            [self.train_step, extra_tensors],
            **sampling_kwargs
        )
        return extra_results

    def compute_on_sample(self, session, replay_memory, tensors,
                          n_seq_samples=0, seq_sample_size=0,
                          n_rnd_entries=0, n_last_entries=0):
        total = n_seq_samples * seq_sample_size + n_rnd_entries + n_last_entries
        if total != self.batch_size:
            raise ValueError("incorrect batch size: {}".format(total))

        states_before, actions, rewards = [], [], []

        for _ in range(n_seq_samples):
            st_before, act, reward, st_after = replay_memory.get_random_slice(seq_sample_size)
            states_before.append(st_before)
            actions.append(act)
            rewards.append(reward)

        if n_rnd_entries > 0:
            st_before, act, reward, st_after = replay_memory.get_random_sample(n_rnd_entries)
            states_before.append(st_before)
            actions.append(act)
            rewards.append(reward)

        if n_last_entries > 0:
            st_before, act, reward, st_after = replay_memory.get_last_entries(n_last_entries)
            states_before.append(st_before)
            actions.append(act)
            rewards.append(reward)

        states_before_sample = np.concatenate(states_before, axis=0)
        actions_sample = np.concatenate(actions, axis=0)[..., 0]
        reward_sample = np.concatenate(rewards, axis=0)

        return session.run(tensors, {
            self.state_ph: states_before_sample,
            self.action_idx_ph: actions_sample,
            self.reward_ph: reward_sample[..., 0],
        })
