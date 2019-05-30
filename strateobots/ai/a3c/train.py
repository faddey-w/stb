import argparse
import os
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
    parser.add_argument("log_dir")
    opts = parser.parse_args(argv)

    controls, model_ctor_name, encoder_name = model_saving.load_model_config(
        opts.model_dir
    )
    model_ctor = util.get_by_import_path(model_ctor_name)
    encoder, state_dim = data_encoding.get_encoder(encoder_name)

    control_model = {
        "move": nets.CategoricalControlOutput,
        "action": nets.CategoricalControlOutput,
        "orientation": nets.OrientationOutput,
        "gun_orientation": nets.OrientationOutput,
    }

    global_actor = nets.build_network("GlobalNet/Actor", model_ctor, control_model)
    global_critic = model_ctor({"value": 1}, scope="GlobalNet/Critic")

    worker_actor = nets.build_network("WorkerNet/Actor", model_ctor, control_model)
    worker_critic = model_ctor({"value": 1}, scope="WorkerNet/Critic")

    batch_size = 200
    entropy_weight = 0.1
    learning_rate = 0.01
    sync_each_n_games = 5
    reward_discount = 0.95
    replay_subsample_rate = 5
    total_games = 500
    state_vec_t = tf.placeholder(tf.float32, [1, state_dim])
    worker_controls_t = {
        ctl: out.sample()[0]
        for ctl, out in nets.build_inference(worker_actor, state_vec_t, control_model).items()
    }
    worker_value_t = worker_critic(state_vec_t)["value"][0]

    # create variable nodes for global nets:
    global_actor(state_vec_t)
    global_critic(state_vec_t)

    state_his_t = tf.placeholder(tf.float32, [batch_size, state_dim])
    actions_his_t = {
        ctl: tf.placeholder(
            tf.int32 if ctl in data.CATEGORICAL_CONTROLS else tf.float32,
            [batch_size, data.get_control_feature(ctl).dimension],
        )
        for ctl in controls
    }
    reward_t = tf.placeholder(tf.float32, [batch_size])
    policy_prediction = nets.build_inference(worker_actor, state_his_t, control_model)
    value_prediction_t = worker_critic(state_his_t)["value"][:, 0]

    critic_loss = tf.losses.mean_squared_error(reward_t, value_prediction_t)
    advantage = tf.stop_gradient(reward_t - value_prediction_t)
    tf.summary.scalar("Loss/value", critic_loss)
    actor_losses = {}
    for ctl in controls:
        pred = policy_prediction[ctl]  # type: nets.ControlOutput
        action_his = actions_his_t[ctl]
        loss = tf.reduce_mean(- pred.log_prob(action_his) * advantage)
        entropy = tf.reduce_mean(pred.entropy())
        tf.summary.scalar("Loss/" + ctl, loss)
        tf.summary.scalar("Entropy/" + ctl, entropy)
        actor_losses[ctl] = loss - entropy_weight * entropy
        # actor_losses[ctl] = tf.maximum(loss, 0) - entropy_weight * tf.maximum(entropy, 0)
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
    train_step_t = tf.train.get_or_create_global_step()
    update_op = opt.apply_gradients(
        list(zip(grads, g_vars)), train_step_t
    )
    sync_op = [w_v.assign(g_v) for w_v, g_v in zip(w_vars, g_vars)]

    init_op = tf.variables_initializer(g_vars + opt.variables()), train_step_t.initializer
    train_summaies_t = tf.summary.merge_all()
    tf.get_default_graph().finalize()
    sess = tf.Session()

    sess.run(init_op)
    os.makedirs(opts.log_dir, exist_ok=True)
    summary_writer = tf.summary.FileWriter(opts.log_dir)

    ctl_features = {ctl: data.get_control_feature(ctl) for ctl in controls}

    bot_init = random_bot_initializer([BotType.Raider], [BotType.Raider])

    def agent_function(state):
        state_vec = encoder(state)
        ctl_vectors, value = sess.run(
            (worker_controls_t, worker_value_t), feed_dict={state_vec_t: [state_vec]}
        )
        ctl_dicts = model_function.predictions_to_ctls(ctl_vectors, state)
        ctl_dict = ctl_dicts[0]
        buffer_s.append(state_vec)
        buffer_v.append(value)
        for ctl in controls:
            buffer_a[ctl].append(ctl_features[ctl](ctl_dict))
        return ctl_dicts

    opponent_function = util.get_object_by_config("config.ini", "ai.simple_longrange")

    buffer_s = []
    buffer_r = []
    buffer_a = {ctl: [] for ctl in controls}
    buffer_v = []

    for game_i in range(total_games):
        if game_i % sync_each_n_games == 0:
            sess.run(sync_op)

        offs = len(buffer_s)

        engine = StbEngine(
            ai1=agent_function,
            ai2=opponent_function,
            initialize_bots=bot_init,
            max_ticks=2000,
            wait_after_win_ticks=0,
            stop_all_after_finish=True,
            debug=True
        )
        engine.play_all()
        t1 = engine.team1
        t2 = engine.team2

        imm_val = _extract_immediate_value(engine.replay, t1, t2)
        # immediate_rewards = imm_val[1:] - imm_val[:-1]
        immediate_rewards = imm_val[1:]
        discounted_reward_reverted = []
        r_acc = 0
        for r in immediate_rewards[::-1]:
            r_acc = r_acc * reward_discount + r
            discounted_reward_reverted.append(r_acc)
        buffer_r.extend(discounted_reward_reverted[::-1])

        mean_reward = float(np.mean(buffer_r[offs:]))
        mean_value = float(np.mean(buffer_v[offs:]))
        perf_smr = tf.Summary()
        perf_smr.value.add(tag="Perf/Reward", simple_value=mean_reward)
        perf_smr.value.add(tag="Perf/Length", simple_value=len(buffer_r[offs:]))
        perf_smr.value.add(tag="Perf/Value", simple_value=mean_value)
        perf_smr.value.add(tag="Perf/Advantage", simple_value=mean_reward - mean_value)
        summary_writer.add_summary(perf_smr, game_i)
        print(f"Game {game_i+1}/{total_games}: m_r={mean_reward:.3f}")

        if len(buffer_s) >= replay_subsample_rate * batch_size:
            subsample = slice(
                None, replay_subsample_rate * batch_size, replay_subsample_rate
            )
            remain_slice = slice(replay_subsample_rate * batch_size, None)

            actions_feed = {
                actions_his_t[ctl]: buffer_a[ctl][subsample] for ctl in controls
            }
            feed = {
                state_his_t: buffer_s[subsample],
                reward_t: buffer_r[subsample],
                **actions_feed,
            }
            train_smr, _, train_step = sess.run(
                [train_summaies_t, update_op, train_step_t], feed
            )

            buffer_r = buffer_r[remain_slice]
            buffer_s = buffer_s[remain_slice]
            buffer_v = buffer_v[remain_slice]
            buffer_a = {ctl: buffer_a[ctl][remain_slice] for ctl in controls}

            summary_writer.add_summary(train_smr, train_step)

        summary_writer.flush()


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
        vals.append(bot_hp - enemy_hp)
    return np.array(vals)


if __name__ == "__main__":
    # main([".data/A3C/models/direct", ".data/A3C/logs/direct/try1"])
    main([".data/A3C/models/anglenav2", ".data/A3C/logs/anglenav2/try1"])
