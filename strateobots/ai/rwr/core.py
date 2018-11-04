import tensorflow as tf
from strateobots.ai.lib import data


class RewardWeightedRegression:

    def __init__(self, model, reward_function, batch_size=10):
        self.model = model
        self.batch_size = batch_size
        self.reward_function = reward_function

        self.state_ph = tf.placeholder(tf.float32, [batch_size, model.state_dimension])
        self.action_idx_ph = {
            ctl: tf.placeholder(tf.int32, [batch_size])
            for ctl in data.ALL_CONTROLS
        }
        self.reward_ph = tf.placeholder(tf.float32, [batch_size])
        self.inference = self.model.apply(self.state_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

        self.loss_vectors = {}
        for ctl in data.ALL_CONTROLS:
            quality = self.inference.controls[ctl]
            quality_safe = quality - tf.reduce_max(quality, axis=1, keepdims=True)
            loss_vector = self.reward_ph * tf.log(
                tf.clip_by_value(tf.gather_nd(
                    1-tf.nn.softmax(quality_safe, axis=1),
                    tf.stack([tf.range(batch_size), self.action_idx_ph[ctl]], axis=1)
                ), 0.00001, 0.99999)
            )
            self.loss_vectors[ctl] = loss_vector
        self.full_loss_vector = tf.add_n(list(self.loss_vectors.values()))
        self.loss = tf.reduce_mean(self.full_loss_vector)

        self.train_step = self.optimizer.minimize(self.loss, var_list=model.var_list)

        self.init_op = tf.variables_initializer(self.optimizer.variables())

    def do_train_step(self, session, replay_memory, batch_index, extra_tensors=()):
        _, extra_results = self.compute_on_sample(
            session,
            replay_memory,
            [self.train_step, extra_tensors],
            batch_index,
        )
        return extra_results

    def compute_on_sample(self, session, replay_memory, tensors, batch_index):

        ticks, actions, states, _ = replay_memory.get_prepared_epoch_batch(self.batch_size, batch_index)
        reward = self.reward_function(ticks)

        return session.run(tensors, {
            self.state_ph: states,
            self.reward_ph: reward,
            **{self.action_idx_ph[ctl]: actions[ctl]
               for ctl in data.ALL_CONTROLS},
        })
