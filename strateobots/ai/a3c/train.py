import argparse
import os
import time
import threading
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
from strateobots.ai.lib import model_function, model_saving, data_encoding, data
from strateobots.ai import nets
from strateobots.engine import StbEngine, BotType
from strateobots.ai.lib.bot_initializers import random_bot_initializer
from strateobots import util

ENTROPY_WEIGHT = None  # 0.0000
POLICY_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.00001
REWARD_DISCOUNT = 0.99
GRADIENT_NORM_CLIP = 3
MODEL_SAVE_PERIOD_MINUTES = 3
N_WORKERS = cpu_count()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument(
        "--reset-step",
        action="store_true",
        help="sets training step to 0, use when starting with pretrained network",
    )
    parser.add_argument("--restore-prefix-map")
    parser.add_argument("--only-critic", action="store_true")
    opts = parser.parse_args(argv)

    _, model_ctor_name, encoder_name = model_saving.load_model_config(
        opts.model_dir
    )
    model_ctor = util.get_by_import_path(model_ctor_name)
    encoder, state_dim = data_encoding.get_encoder(encoder_name)

    nets.AnglenavModel(model_ctor, state_dim, "GlobalNet")
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "GlobalNet")
    assert model_vars

    train_step_t = tf.train.get_or_create_global_step()
    inc_step_op = tf.assign_add(train_step_t, 1)

    all_saveable_variables = model_vars + [train_step_t]
    saver = tf.train.Saver(
        all_saveable_variables, keep_checkpoint_every_n_hours=1, max_to_keep=5
    )
    if opts.restore_prefix_map is None:
        model_restorer = tf.train.Saver(model_vars)
    else:
        from_prefix, to_prefix = opts.restore_prefix_map.split(":")

        def _replace_prefix(variable_name):
            if variable_name.startswith(to_prefix):
                return from_prefix + variable_name[len(to_prefix) :]
            raise ValueError(variable_name)

        model_restorer = tf.train.Saver(
            {_replace_prefix(v.op.name): v for v in model_vars}
        )

    model_init_op = tf.variables_initializer(all_saveable_variables)

    workers = [
        Worker(
            worker_id=i + 1,
            encoder=encoder,
            global_model_vars=model_vars,
            inc_step_op=inc_step_op,
            model_constructor=model_ctor,
            state_dim=state_dim,
            train_only_critic=opts.only_critic,
        )
        for i in range(N_WORKERS)
    ]

    tf.get_default_graph().finalize()
    sess = tf.Session()

    ckpt = tf.train.latest_checkpoint(opts.model_dir)
    if ckpt:
        print("Restoring model from", ckpt)
        model_restorer.restore(sess, ckpt)
    else:
        print("Initializing untrained model")
        sess.run(model_init_op)
    summary_writer = tf.summary.FileWriter(opts.model_dir)

    opponent_function = util.get_object_by_config("config.ini", "ai.simple_longrange")
    for worker in workers:
        threading.Thread(
            target=lambda w: w.main(sess, opponent_function, summary_writer),
            args=[worker],
            daemon=True,
        ).start()

    def save_model():
        step_i = sess.run(train_step_t)
        ckpt = saver.save(sess, os.path.join(opts.model_dir, "model-ckpt"), step_i)
        print("Model is saved to", ckpt)

    try:
        while True:
            time.sleep(MODEL_SAVE_PERIOD_MINUTES * 60)
            save_model()
    except KeyboardInterrupt:
        print("Interrupted")
        save_model()


class Worker:
    def __init__(
        self,
        worker_id,
        model_constructor,
        global_model_vars,
        state_dim,
        encoder,
        train_only_critic,
        inc_step_op,
    ):
        self.scope = f"WorkerNet_{worker_id}"
        self.model = nets.AnglenavModel(model_constructor, state_dim, self.scope)
        self.id = worker_id
        self.state_vec_t = tf.placeholder(tf.float32, [1, state_dim])
        self.value_t, self.controls_outs = self.model(self.state_vec_t)
        self.controls_t = {
            ctl: out.sample()[0] for ctl, out in self.controls_outs.items()
        }
        self.encoder = encoder
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.controls = tuple(self.model.control_model.keys())
        self.inc_step_op = inc_step_op

        self.state_his_t = tf.placeholder(tf.float32, [None, state_dim])
        self.actions_his_t = {
            ctl: tf.placeholder(
                tf.int32 if ctl in data.CATEGORICAL_CONTROLS else tf.float32, [None]
            )
            for ctl in self.controls
        }
        self.reward_t = tf.placeholder(tf.float32, [None])
        value_prediction_t, policy_prediction = self.model(self.state_his_t)

        critic_loss = tf.losses.mean_squared_error(self.reward_t, value_prediction_t)
        advantage = tf.stop_gradient(self.reward_t - value_prediction_t)
        actor_losses = {}
        self.summaries = {"Loss/value": critic_loss}
        for ctl in self.controls:
            pred = policy_prediction[ctl]  # type: nets.ControlOutput
            action_his = self.actions_his_t[ctl]
            loss = tf.reduce_mean(-pred.log_prob(action_his) * advantage)
            entropy = tf.reduce_mean(pred.entropy())
            self.summaries["Loss/" + ctl] = loss
            self.summaries["Entropy/" + ctl] = entropy
            actor_losses[ctl] = loss
            if ENTROPY_WEIGHT is not None:
                actor_losses[ctl] -= ENTROPY_WEIGHT * entropy
        actor_loss = tf.add_n(list(actor_losses.values()))
        self.summaries["Loss/total"] = critic_loss + actor_loss
        _critic_loss_scale = CRITIC_LEARNING_RATE / POLICY_LEARNING_RATE
        if train_only_critic:
            training_loss = critic_loss * _critic_loss_scale
        else:
            training_loss = actor_loss + critic_loss * _critic_loss_scale

        grads = tf.gradients(training_loss, self.variables)
        grads, grad_norm_t = tf.clip_by_global_norm(grads, GRADIENT_NORM_CLIP)
        self.summaries["Loss/grad_norm"] = grad_norm_t

        optimizer = tf.train.RMSPropOptimizer(
            POLICY_LEARNING_RATE
        )  # critic LR applied via loss
        self.push_op = optimizer.apply_gradients(list(zip(grads, global_model_vars)))
        self.pull_op = [
            w_v.assign(g_v) for w_v, g_v in zip(self.variables, global_model_vars)
        ]
        self.init_op = tf.variables_initializer(optimizer.variables())

    def main(self, sess, opponent_function, summary_writer):
        sess.run(self.init_op)
        while True:
            sess.run(self.pull_op)
            engine, buffer_s, buffer_a, buffer_v, immediate_r = self._generate_game(
                sess, opponent_function
            )

            discounted_reward = _discount(immediate_r, REWARD_DISCOUNT)
            perf_stats = self._summarize_performance(
                engine, buffer_s, buffer_v, discounted_reward
            )

            feed = {
                self.state_his_t: buffer_s,
                self.reward_t: discounted_reward,
                **{self.actions_his_t[ctl]: buffer_a[ctl] for ctl in self.controls},
            }
            train_stats, _, game_i = sess.run(
                [self.summaries, self.push_op, self.inc_step_op], feed
            )
            print(f"Game {game_i + 1}: m_r={np.mean(discounted_reward):.3f}")

            smry = tf.Summary()
            for key, value in {**perf_stats, **train_stats}.items():
                smry.value.add(tag=key, simple_value=value)
            summary_writer.add_summary(smry, game_i)
            summary_writer.flush()

    def _summarize_performance(self, engine, buffer_s, buffer_v, discounted_reward):
        last_hp1, last_hp2 = _get_final_hp(engine)
        damage_dealt = 1 - last_hp2
        damage_taken = 1 - last_hp1
        episode_len = len(buffer_s)
        mean_reward = float(np.mean(discounted_reward))
        mean_value = float(np.mean(buffer_v))

        return {
            "Perf/Reward": mean_reward,
            "Perf/Length": episode_len,
            "Perf/Value": mean_value,
            "Perf/Advantage": mean_reward - mean_value,
            "Perf/DmgDealt": damage_dealt,
            "Perf/DmgTaken": damage_taken,
        }

    def _generate_game(self, sess, opponent_function):
        def agent_function(state):
            state_vec = self.encoder(state)
            ctl_vectors, value = sess.run(
                (self.controls_t, self.value_t),
                feed_dict={self.state_vec_t: [state_vec]},
            )
            ctl_dicts = model_function.predictions_to_ctls(ctl_vectors, state)
            buf_s.append(state_vec)
            buf_v.append(value)
            for ctl in self.controls:
                buf_a[ctl].append(ctl_vectors[ctl])
            return ctl_dicts

        buf_s = []
        buf_a = {ctl: [] for ctl in self.controls}
        buf_v = []

        bot_init = random_bot_initializer([BotType.Raider], [BotType.Raider])
        engine = StbEngine(
            ai1=agent_function,
            ai2=opponent_function,
            initialize_bots=bot_init,
            max_ticks=2000,
            wait_after_win_ticks=0,
            stop_all_after_finish=True,
            debug=True,
        )
        engine.play_all()

        imm_val = _extract_immediate_value(engine.replay, engine.team1, engine.team2)
        # immediate_rewards = imm_val[:-1]
        immediate_rewards = imm_val[1:] - imm_val[:-1]

        if len(buf_s) == len(immediate_rewards) + 1:
            # the case if game is stopped due to timeout
            buf_s = buf_s[1:]
            buf_v = buf_v[1:]
            buf_a = {ctl: arr[1:] for ctl, arr in buf_a.items()}

        assert {
            len(buf_s),
            *map(len, buf_a.values()),
            len(buf_v),
            len(immediate_rewards),
        } == {len(buf_s)}
        return engine, buf_s, buf_a, buf_v, immediate_rewards


def _extract_immediate_value(replay_data, t1, t2):
    vals = []
    for state in replay_data:
        bot = enemy = None
        if state["bots"].get(t1):
            bot = state["bots"][t1][0]
            bot_hp = bot["hp"]
        else:
            bot_hp = 0
        if state["bots"].get(t2):
            enemy = state["bots"][t2][0]
            enemy_hp = enemy["hp"]
        else:
            enemy_hp = 0
        val = -1.25 * enemy_hp
        if bot and enemy:
            dx = enemy["x"] - bot["x"]
            dy = enemy["y"] - bot["y"]
            dist = (dx * dx + dy * dy) ** 0.5
            gun_direction = bot["orientation"] + bot["tower_orientation"]
            gun_x = np.cos(gun_direction)
            gun_y = np.sin(gun_direction)
            val += 0.3 * (dx * gun_x + dy * gun_y) / dist
        # vals.append(1 - enemy_hp)
        # vals.append(100 * (bot_hp - enemy_hp))
        # vals.append(0.1 * bot_hp - 1.25 * enemy_hp)
        vals.append(val)
    return np.array(vals) * 100


def _discount(values, discount_factor):
    discounted = np.zeros_like(values)
    r_acc = 0
    for i, r in enumerate(values[::-1], 1):
        r_acc = r_acc * discount_factor + r
        discounted[-i] = r_acc
    return discounted


def _get_final_hp(engine):
    t1, t2 = engine.team1, engine.team2
    hp1 = hp2 = 0
    last_state = engine.replay[-1]
    if last_state["bots"].get(t1):
        hp1 = last_state["bots"][t1][0]["hp"]
    if last_state["bots"].get(t2):
        hp2 = last_state["bots"][t2][0]["hp"]
    return hp1, hp2


if __name__ == "__main__":
    main([".data/A3C/models/anglenav2"])
    # main()
