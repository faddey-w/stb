import argparse
import os
import queue
import threading
import random
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
    parser.add_argument("--init-training", action="store_true",
                        help="use when restoring first time from pretrained network")
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

    batch_size = 3000
    entropy_weight = 0.1
    learning_rate = 0.001
    sync_each_n_games = 1
    reward_discount = 0.95
    replay_subsample_rate = 2
    save_each_n_steps = 50
    n_workers = cpu_count()
    state_vec_t = tf.placeholder(tf.float32, [1, state_dim])
    worker_value_t, worker_controls_outs = worker_model(state_vec_t)
    worker_controls_t = {
        ctl: out.sample()[0]
        for ctl, out in worker_controls_outs.items()
    }

    # make sure variable nodes for global net are created:
    global_model(state_vec_t)

    state_his_t = tf.placeholder(tf.float32, [batch_size, state_dim])
    actions_his_t = {
        ctl: tf.placeholder(
            tf.int32 if ctl in data.CATEGORICAL_CONTROLS else tf.float32, [batch_size]
        )
        for ctl in controls
    }
    reward_t = tf.placeholder(tf.float32, [batch_size])
    value_prediction_t, policy_prediction = worker_model(state_his_t)

    critic_loss = tf.losses.mean_squared_error(reward_t, value_prediction_t)
    advantage = tf.stop_gradient(reward_t - value_prediction_t)
    tf.summary.scalar("Loss/value", critic_loss)
    actor_losses = {}
    for ctl in controls:
        pred = policy_prediction[ctl]  # type: nets.ControlOutput
        action_his = actions_his_t[ctl]
        loss = tf.reduce_mean(-pred.log_prob(action_his) * advantage)
        entropy = tf.reduce_mean(pred.entropy())
        tf.summary.scalar("Loss/" + ctl, loss)
        tf.summary.scalar("Entropy/" + ctl, entropy)
        actor_losses[ctl] = loss - entropy_weight * entropy
    actor_loss = tf.add_n(list(actor_losses.values()))
    total_loss = critic_loss + actor_loss
    tf.summary.scalar("Loss/total", total_loss)

    key = tf.GraphKeys.TRAINABLE_VARIABLES
    g_vars = tf.get_collection(key, "GlobalNet")
    w_vars = tf.get_collection(key, "WorkerNet")

    grads = tf.gradients(total_loss, w_vars)
    grads, grad_norm = tf.clip_by_global_norm(grads, 40)
    tf.summary.scalar("Loss/grad_norm", grad_norm)

    opt = tf.train.RMSPropOptimizer(learning_rate)
    update_op = opt.apply_gradients(list(zip(grads, g_vars)))
    sync_op = [w_v.assign(g_v) for w_v, g_v in zip(w_vars, g_vars)]
    train_step_t = tf.train.get_or_create_global_step()
    inc_step_op = tf.assign_add(train_step_t, 1)

    training_process_variables = opt.variables() + [train_step_t]
    all_saveable_variables = g_vars + training_process_variables
    saver = tf.train.Saver(
        all_saveable_variables, keep_checkpoint_every_n_hours=1, max_to_keep=5
    )
    model_restorer = tf.train.Saver(g_vars)
    training_state_restorer = tf.train.Saver(training_process_variables)
    model_init_op = tf.variables_initializer(all_saveable_variables)
    training_init_op = tf.variables_initializer(training_process_variables)

    train_summaies_t = tf.summary.merge_all()
    tf.get_default_graph().finalize()
    sess = tf.Session()

    ckpt = tf.train.latest_checkpoint(opts.model_dir)
    if ckpt:
        print("Restoring model from", ckpt)
        model_restorer.restore(sess, ckpt)
        if opts.init_training:
            print("Initializing training from scratch!")
            sess.run(training_init_op)
        else:
            training_state_restorer.restore(sess, ckpt)
    else:
        print("Initializing untrained model")
        sess.run(model_init_op)
    sess.run(sync_op)
    summary_writer = tf.summary.FileWriter(opts.model_dir)

    buffer_s = []
    buffer_r = []
    buffer_a = {ctl: [] for ctl in controls}
    buffer_v = []
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
            daemon=True
        ).start()

    need_to_sync = False
    need_to_save = False
    init_step = sess.run(train_step_t)
    game_i = init_step-1
    while True:
        game_i += 1
        if game_i % sync_each_n_games == 0:
            need_to_sync = True
        if game_i % save_each_n_steps == 0:
            need_to_save = True

        engine, buf_s, buf_a, buf_v = replays_queue.get()
        t1 = engine.team1
        t2 = engine.team2

        imm_val = _extract_immediate_value(engine.replay, t1, t2)
        immediate_rewards = imm_val[1:] - imm_val[:-1]
        # immediate_rewards = imm_val[1:]
        discounted_reward_reverted = []
        r_acc = 0
        for r in immediate_rewards[::-1]:
            r_acc = r_acc * reward_discount + r
            discounted_reward_reverted.append(r_acc)
        discounted_reward = discounted_reward_reverted[::-1]
        if len(buf_s) == len(discounted_reward) + 1:
            # the case if game stopped due to timeout
            buf_s = buf_s[1:]
            buf_v = buf_v[1:]
            buf_a = {ctl: arr[1:] for ctl, arr in buf_a.items()}
        assert len(set(map(len, [buf_s, *buf_a.values(), buf_v, discounted_reward]))) == 1, (
            tuple(map(len, [buf_s, *buf_a.values(), buf_v, discounted_reward]))
        )
        episode_len = len(buf_s)
        last_hp1, last_hp2 = _get_final_hp(engine)
        damage_dealt = 1 - last_hp2
        damage_taken = 1 - last_hp1

        mean_reward = float(np.mean(discounted_reward_reverted))
        mean_value = float(np.mean(buf_v))
        perf_smr = tf.Summary()
        perf_smr.value.add(tag="Perf/Reward", simple_value=mean_reward)
        perf_smr.value.add(tag="Perf/Length", simple_value=episode_len)
        perf_smr.value.add(tag="Perf/Value", simple_value=mean_value)
        perf_smr.value.add(tag="Perf/Advantage", simple_value=mean_reward - mean_value)
        perf_smr.value.add(tag="Perf/DmgDealt", simple_value=damage_dealt)
        perf_smr.value.add(tag="Perf/DmgTaken", simple_value=damage_taken)
        summary_writer.add_summary(perf_smr, game_i)
        print(f"Game {game_i+1}: m_r={mean_reward:.3f}")

        sample_indices = random.sample(range(episode_len), episode_len // replay_subsample_rate)
        buffer_r.extend(_sample(discounted_reward, sample_indices))
        buffer_s.extend(_sample(buf_s, sample_indices))
        buffer_v.extend(_sample(buf_v, sample_indices))
        for ctl in controls:
            buffer_a[ctl].extend(_sample(buf_a[ctl], sample_indices))
        sess.run(inc_step_op)

        train_step_was_done = False
        while len(buffer_s) >= batch_size:
            train_step_was_done = True

            actions_feed = {
                actions_his_t[ctl]: buffer_a[ctl][:batch_size] for ctl in controls
            }
            feed = {
                state_his_t: buffer_s[:batch_size],
                reward_t: buffer_r[:batch_size],
                **actions_feed,
            }
            train_smr, _ = sess.run([train_summaies_t, update_op], feed)

            buffer_r = buffer_r[batch_size:]
            buffer_s = buffer_s[batch_size:]
            buffer_v = buffer_v[batch_size:]
            buffer_a = {ctl: buffer_a[ctl][batch_size:] for ctl in controls}

            summary_writer.add_summary(train_smr, game_i)

        summary_writer.flush()

        if train_step_was_done:
            if need_to_sync:
                sess.run(sync_op)
                print("sync workers...")
                need_to_sync = False
            if need_to_save:
                ckpt = saver.save(sess, os.path.join(opts.model_dir, "model-ckpt"), game_i)
                print("Model is saved to", ckpt)
                need_to_save = False


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


def _sample(values, indices):
    return np.array(values)[indices]


def _extract_immediate_value(replay_data, t1, t2):
    vals = []
    for state in replay_data:
        if state["bots"].get(t1):
            bot_hp = state["bots"][t1][0]["hp"]
        else:
            bot_hp = 0
        if state["bots"].get(t2):
            enemy_hp = state["bots"][t2][0]["hp"]
        else:
            enemy_hp = 0
        # vals.append(1 - enemy_hp)
        vals.append(100 * (bot_hp - enemy_hp))
    return np.array(vals)


def _get_final_hp(engine):
    t1, t2 = engine.team1, engine.team2
    hp1 = hp2 = 0
    last_state = engine.replay[-1]
    if last_state['bots'].get(t1):
        hp1 = last_state['bots'][t1][0]['hp']
    if last_state['bots'].get(t2):
        hp2 = last_state['bots'][t2][0]['hp']
    return hp1, hp2


if __name__ == "__main__":
    # main([".data/A3C/models/direct", ".data/A3C/logs/direct/try1"])
    main()
