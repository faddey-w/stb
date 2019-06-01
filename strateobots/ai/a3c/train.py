import argparse
import os
import queue
import threading
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
from strateobots.ai.lib import model_function, model_saving, data_encoding, data
from strateobots.ai import nets
from strateobots.engine import StbEngine, BotType
from strateobots.ai.lib.bot_initializers import random_bot_initializer
from strateobots import util


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument(
        "--init-training",
        action="store_true",
        help="use when restoring first time from pretrained network",
    )
    parser.add_argument("--restore-prefix-map")
    parser.add_argument("--only-critic", action="store_true")
    opts = parser.parse_args(argv)

    controls, model_ctor_name, encoder_name = model_saving.load_model_config(
        opts.model_dir
    )
    model_ctor = util.get_by_import_path(model_ctor_name)
    encoder, state_dim = data_encoding.get_encoder(encoder_name)

    global_model = nets.AnglenavModel(model_ctor, "GlobalNet")
    worker_model = nets.AnglenavModel(model_ctor, "WorkerNet")
    assert set(controls) == set(global_model.control_model.keys())
    controls = tuple(global_model.control_model.keys())

    entropy_weight = None  # 0.0000
    policy_learning_rate = 0.001
    critic_learning_rate = 0.00001
    sync_each_n_games = 20
    reward_discount = 0.99
    advantage_scale = 1
    gradient_norm_clip = 3
    urgent_sync_gradnorm_threshold = float("inf")

    n_workers = cpu_count()
    state_vec_t = tf.placeholder(tf.float32, [1, state_dim])
    worker_value_t, worker_controls_outs = worker_model(state_vec_t)
    worker_controls_t = {
        ctl: out.sample()[0] for ctl, out in worker_controls_outs.items()
    }

    # make sure variable nodes for global net are created:
    global_model(state_vec_t)

    state_his_t = tf.placeholder(tf.float32, [None, state_dim])
    actions_his_t = {
        ctl: tf.placeholder(
            tf.int32 if ctl in data.CATEGORICAL_CONTROLS else tf.float32, [None]
        )
        for ctl in controls
    }
    reward_t = tf.placeholder(tf.float32, [None])
    value_prediction_t, policy_prediction = worker_model(state_his_t)

    critic_loss = tf.losses.mean_squared_error(reward_t, value_prediction_t)
    advantage = tf.stop_gradient(reward_t - value_prediction_t) * advantage_scale
    tf.summary.scalar("Loss/value", critic_loss)
    actor_losses = {}
    for ctl in controls:
        pred = policy_prediction[ctl]  # type: nets.ControlOutput
        action_his = actions_his_t[ctl]
        loss = tf.reduce_mean(-pred.log_prob(action_his) * advantage)
        entropy = tf.reduce_mean(pred.entropy())
        tf.summary.scalar("Loss/" + ctl, loss)
        tf.summary.scalar("Entropy/" + ctl, entropy)
        actor_losses[ctl] = loss
        if entropy_weight is not None:
            actor_losses[ctl] -= entropy_weight * entropy
    actor_loss = tf.add_n(list(actor_losses.values()))
    tf.summary.scalar("Loss/total", critic_loss + actor_loss)
    _critic_loss_scale = critic_learning_rate / policy_learning_rate
    if opts.only_critic:
        training_loss = critic_loss * _critic_loss_scale
    else:
        training_loss = actor_loss + critic_loss * _critic_loss_scale

    key = tf.GraphKeys.TRAINABLE_VARIABLES
    g_vars = tf.get_collection(key, "GlobalNet")
    w_vars = tf.get_collection(key, "WorkerNet")

    grads = tf.gradients(training_loss, w_vars)
    grads, grad_norm_t = tf.clip_by_global_norm(grads, gradient_norm_clip)
    tf.summary.scalar("Loss/grad_norm", grad_norm_t)

    opt = tf.train.RMSPropOptimizer(policy_learning_rate)  # critic LR applied via loss
    update_op = opt.apply_gradients(list(zip(grads, g_vars)))
    sync_op = [w_v.assign(g_v) for w_v, g_v in zip(w_vars, g_vars)]
    train_step_t = tf.train.get_or_create_global_step()
    inc_step_op = tf.assign_add(train_step_t, 1)

    training_process_variables = opt.variables() + [train_step_t]
    all_saveable_variables = g_vars + training_process_variables
    saver = tf.train.Saver(
        all_saveable_variables, keep_checkpoint_every_n_hours=1, max_to_keep=5
    )
    if opts.restore_prefix_map is None:
        model_restorer = tf.train.Saver(g_vars)
    else:
        from_prefix, to_prefix = opts.restore_prefix_map.split(":")

        def _replace_prefix(variable_name):
            if variable_name.startswith(to_prefix):
                return from_prefix + variable_name[len(to_prefix) :]
            raise ValueError(variable_name)

        model_restorer = tf.train.Saver({_replace_prefix(v.op.name): v for v in g_vars})

    training_state_restorer = tf.train.Saver(training_process_variables)
    model_init_op = tf.variables_initializer(all_saveable_variables)
    training_init_op = tf.variables_initializer(training_process_variables)

    train_summaries_t = tf.summary.merge_all()
    tf.get_default_graph().finalize()
    sess = tf.Session()

    ckpt = tf.train.latest_checkpoint(opts.model_dir)
    if ckpt:
        print("Restoring model from", ckpt)
        model_restorer.restore(sess, ckpt)
        if opts.init_training:
            print("Initializing zero training state!")
            sess.run(training_init_op)
        else:
            training_state_restorer.restore(sess, ckpt)
    else:
        print("Initializing untrained model")
        sess.run(model_init_op)
    sess.run(sync_op)
    summary_writer = tf.summary.FileWriter(opts.model_dir)

    replays_queue = queue.Queue(maxsize=n_workers * 3)

    opponent_function = util.get_object_by_config("config.ini", "ai.simple_longrange")
    for _ in range(n_workers):
        threading.Thread(
            target=lambda: play_worker(
                sess,
                worker_controls_t,
                worker_value_t,
                state_vec_t,
                controls,
                opponent_function,
                encoder,
                replays_queue,
            ),
            daemon=True,
        ).start()

    game_i = sess.run(train_step_t) - 1
    while True:
        game_i += 1

        engine, buf_s, buf_a, buf_v = replays_queue.get()
        t1 = engine.team1
        t2 = engine.team2

        imm_val = _extract_immediate_value(engine.replay, t1, t2)
        # immediate_rewards = imm_val[:-1]
        immediate_rewards = imm_val[1:] - imm_val[:-1]

        if len(buf_s) == len(immediate_rewards) + 1:
            # the case if game is stopped due to timeout
            buf_s = buf_s[1:]
            buf_v = buf_v[1:]
            buf_a = {ctl: arr[1:] for ctl, arr in buf_a.items()}
        episode_len = len(buf_s)

        discounted_reward = np.zeros([episode_len])
        r_acc = 0
        for i, r in enumerate(immediate_rewards[::-1], 1):
            r_acc = r_acc * reward_discount + r
            discounted_reward[-i] = r_acc
        assert {
            len(buf_s),
            *map(len, buf_a.values()),
            len(buf_v),
            len(discounted_reward),
        } == {len(buf_s)}
        last_hp1, last_hp2 = _get_final_hp(engine)
        damage_dealt = 1 - last_hp2
        damage_taken = 1 - last_hp1

        mean_reward = float(np.mean(discounted_reward))
        mean_value = float(np.mean(buf_v))
        perf_smr = tf.Summary()
        perf_smr.value.add(tag="Perf/Reward", simple_value=mean_reward)
        perf_smr.value.add(tag="Perf/Length", simple_value=episode_len)
        perf_smr.value.add(tag="Perf/Value", simple_value=mean_value)
        perf_smr.value.add(tag="Perf/Advantage", simple_value=mean_reward - mean_value)
        perf_smr.value.add(tag="Perf/DmgDealt", simple_value=damage_dealt)
        perf_smr.value.add(tag="Perf/DmgTaken", simple_value=damage_taken)
        summary_writer.add_summary(perf_smr, game_i)
        print(f"Game {game_i+1}: m_r={np.mean(discounted_reward):.3f}")

        actions_feed = {actions_his_t[ctl]: buf_a[ctl] for ctl in controls}
        feed = {state_his_t: buf_s, reward_t: discounted_reward, **actions_feed}
        train_smr, _, _, grad_norm = sess.run(
            [train_summaries_t, update_op, inc_step_op, grad_norm_t], feed
        )
        summary_writer.add_summary(train_smr, game_i)

        summary_writer.flush()

        do_sync = do_save = (game_i % sync_each_n_games == 0)
        if grad_norm > urgent_sync_gradnorm_threshold:
            print(f"Urgent sync: grad_norm={grad_norm:.3f}")
            do_sync = True
        if do_sync:
            sess.run(sync_op)
            print("sync workers...")
        if do_save:
            ckpt = saver.save(sess, os.path.join(opts.model_dir, "model-ckpt"), game_i)
            print("Model is saved to", ckpt)


def play_worker(
    sess,
    worker_controls_t,
    worker_value_t,
    state_vec_t,
    controls,
    opponent_function,
    encoder,
    out_queue,
):
    while True:

        def agent_function(state):
            state_vec = encoder(state)
            ctl_vectors, value = sess.run(
                (worker_controls_t, worker_value_t),
                feed_dict={state_vec_t: [state_vec]},
            )
            ctl_dicts = model_function.predictions_to_ctls(ctl_vectors, state)
            buffer_s.append(state_vec)
            buffer_v.append(value)
            for ctl in controls:
                buffer_a[ctl].append(ctl_vectors[ctl])
            return ctl_dicts

        buffer_s = []
        buffer_a = {ctl: [] for ctl in controls}
        buffer_v = []

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

        out_queue.put((engine, buffer_s, buffer_a, buffer_v))


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
            dx = enemy['x'] - bot['x']
            dy = enemy['y'] - bot['y']
            dist = (dx*dx + dy*dy) ** 0.5
            gun_direction = bot['orientation'] + bot['tower_orientation']
            gun_x = np.cos(gun_direction)
            gun_y = np.sin(gun_direction)
            val += 0.3 * (dx * gun_x + dy * gun_y) / dist
        # vals.append(1 - enemy_hp)
        # vals.append(100 * (bot_hp - enemy_hp))
        # vals.append(0.1 * bot_hp - 1.25 * enemy_hp)
        vals.append(val)
    return np.array(vals) * 100


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
    # main([".data/A3C/models/direct", ".data/A3C/logs/direct/try1"])
    main()
