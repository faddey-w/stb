import tensorflow as tf
from strateobots.ai.lib import data


class PolicyLearning:

    def __init__(self, model, batch_size=10):
        self.model = model
        self.batch_size = batch_size

        self.state_ph = tf.placeholder(tf.float32, [batch_size, model.state_dimension])
        self.action_idx_ph = {
            ctl: tf.placeholder(tf.int32, [batch_size])
            for ctl in data.ALL_CONTROLS
        }
        self.inference = self.model.apply(self.state_ph)

        # self.optimizer = tf.train.RMSPropOptimizer(0.001)
        self.optimizer = tf.train.AdamOptimizer()
        # self.optimizer = tf.train.GradientDescentOptimizer(0.001)

        self.loss_vectors = {}
        self.losses = {}
        self.accuracies = {}
        self.train_steps = {}
        self.vars_grads = {}
        for ctl in data.ALL_CONTROLS:
            quality = self.inference.controls[ctl]
            n_actions = tf.shape(quality)[1]
            action_labels = tf.one_hot(self.action_idx_ph[ctl], n_actions)
            loss_vector = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(action_labels),
                logits=quality,
            )
            self.loss_vectors[ctl] = loss_vector
            self.losses[ctl] = tf.reduce_mean(loss_vector)
            self.vars_grads[ctl] = self.optimizer.compute_gradients(self.losses[ctl], model.var_list)
            self.train_steps[ctl] = self.optimizer.apply_gradients(self.vars_grads[ctl])

            predicted_idx = tf.argmax(quality, 1, output_type=tf.int32)
            match_flags = tf.equal(predicted_idx, self.action_idx_ph[ctl])
            accuracy = tf.reduce_mean(tf.to_float(match_flags))
            self.accuracies[ctl] = accuracy

        # self.full_loss_vector = tf.add_n(list(self.loss_vectors.values()))
        # self.loss = tf.reduce_mean(self.full_loss_vector)

        self.train_step = tuple(self.train_steps.values())

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

        props, actions, states, _ = replay_memory.get_prepared_epoch_batch(batch_index)

        return session.run(tensors, {
            self.state_ph: states,
            **{self.action_idx_ph[ctl]: actions[ctl]
               for ctl in data.ALL_CONTROLS},
        })
