import argparse
import datetime
import logging
import os

import tensorflow as tf

from strateobots import REPO_ROOT
from strateobots.engine import BotType
from .core import get_session, ReinforcementLearning, control_noise
from ..lib import replay, model_saving, handcrafted
from ..lib.data import state2vec, action2vec
from . import ai, model

tf.reset_default_graph()
log = logging.getLogger(__name__)


def noised(trainer_function, noise_prob):
    def function(bot, enemy, ctl):
        trainer_function(bot, enemy, ctl)
        control_noise(ctl, noise_prob)
    return function


class Config:
    
    memory_capacity = 100000
    memory_capacity_per_class = 1000

    new_model_cls = model.semisparse_fc.QualityFunctionModel

    model_params = dict(
        # coord_cfg=[8] * 4,
        # angle_cfg=[8] * 4,
        # fc_cfg=[20] * (8 - 1) + [8],
        # exp_layers=[4],

        # fc_cfg=[8]*7 + [6]*16 + [8],
        # exp_layers=[],
        # pool_layers=[2, 4, 5, 9, 12],
        # join_point=7,

        # lin_h=20,
        # log_h=20,
        # lin_o=40,
        # n_evt=30,

        # linear_cfg=[(30, 50), (30, 50), (20, 20)],
        # linear_cfg=[(30, 50), (30, 50), (30, 50), (30, 50), (20, 20)],
        # logical_cfg=[40, 40, 20],
        # values_cfg=[(10, 20)],

        # vec2d_cfg=[(7, 11)] * 10,
        # fc_cfg=[23, 17, 13],

        n_parts=30,
        cfg=[4] * 10,
    )

    batch_size = 120
    sampling = dict(
        n_seq_samples=0,
        seq_sample_size=0,
        n_rnd_entries=100,
        n_last_entries=20
    )
    reward_prediction = 0.95
    select_random_prob_decrease = 0.03
    select_random_max_prob = 0.1
    select_random_min_prob = 0.1
    self_play = False

    bot_type = BotType.Raider
        
    modes = [
        ai.NotMovingMode(),
        ai.LocateAtCircleMode(),
        ai.NoShieldMode(),
        # ai.NotBodyRotatingMode(),
        # ai.BackToCenter(),
    ]
    trainer_function = staticmethod(noised(handcrafted.turret_behavior, 0.1))
    # trainer_function = staticmethod(noised(handcrafted.distance_attack, 0.1))
    # trainer_function = staticmethod(noised(handcrafted.short_range_attack, 0.1))


def memory_keyfunc(state_before, action, state_after):
    x0 = state_before[state2vec[0, 'x']]
    y0 = state_before[state2vec[0, 'y']]
    x1 = state_before[state2vec[1, 'x']]
    y1 = state_before[state2vec[1, 'y']]
    x_flag = x0 > x1
    y_flag = y0 > y1
    d_flag = ((x1 - x0) ** 2 + (y1 - y0) ** 2) > 200 ** 2
    e_dmg = state_before[state2vec[1, 'hp_ratio']] - state_after[state2vec[1, 'hp_ratio']]
    b_dmg = state_before[state2vec[0, 'hp_ratio']] - state_after[state2vec[0, 'hp_ratio']]
    return x_flag, y_flag, d_flag, e_dmg > 0.001, b_dmg > 0.001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--max-games', default=None, type=int)
    opts = parser.parse_args()
    cfg = Config()

    if opts.save_dir is None:
        run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        opts.save_dir = os.path.join(REPO_ROOT, '_data', run_name)
    logs_dir = os.path.join(opts.save_dir, 'logs')
    model_dir = os.path.join(opts.save_dir, 'model', '')
    replay_dir = os.path.join(opts.save_dir, 'replay')

    logging.basicConfig(level=logging.INFO)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    sess = get_session()
    try:
        model_mgr = model_saving.ModelManager.load_existing_model(model_dir)
        model_mgr.load_vars(sess)
        model = model_mgr.model  # type: model.QualityFunctionModel
    except Exception as exc:
        log.info("Cannot load model (%r), so creating a new one", exc)
        model = cfg.new_model_cls(**cfg.model_params)
        model_mgr = model_saving.ModelManager(model, model_dir)
        model_mgr.init_vars(sess)
    # replay_memory = replay.ReplayMemory(
    #     cfg.memory_capacity,
    #     state2vec.vector_length,
    #     action2vec.vector_length,
    #     state2vec.vector_length,
    # )
    replay_memory = replay.BalancedMemory(
        memory_keyfunc,
        cfg.memory_capacity_per_class,
        [state2vec.vector_length,
         action2vec.vector_length,
         state2vec.vector_length]
    )
    rl = ReinforcementLearning(
        model,
        batch_size=cfg.batch_size,
        reward_prediction=cfg.reward_prediction,
        self_play=cfg.self_play,
    )
    sess.run(rl.init_op)
    try:
        replay_memory.load(replay_dir)
        log.info('replay memory buffer loaded from %s', replay_dir)
    except FileNotFoundError:
        log.info('collecting new replay memory buffer')

    try:
        i = model_mgr.step_counter

        if opts.save:
            emu_writer = tf.summary.FileWriter(
                os.path.join(logs_dir, 'emu'),
                sess.graph
            )
            train_writer = tf.summary.FileWriter(
                os.path.join(logs_dir, 'train'),
                sess.graph
            )
        else:
            emu_writer = None
            train_writer = None

        while True:
            for mode in cfg.modes:
                mode.reset()
            select_random_prob = max(
                cfg.select_random_min_prob,
                cfg.select_random_max_prob - cfg.select_random_prob_decrease * model_mgr.step_counter
            )
            ai1_factory = ai.DQNDuelAI.parametrize(
                bot_type=cfg.bot_type,
                modes=cfg.modes
            )
            ai2_factory = ai.DQNDuelAI.parametrize(
                bot_type=cfg.bot_type,
                modes=cfg.modes,
                trainer_function=cfg.trainer_function if not cfg.self_play else None,
            )

            i += 1
            rl.run(
                frameskip=2,
                max_ticks=3000,
                world_size=1000,
                replay_memory=replay_memory,
                ai1_cls=ai1_factory,
                ai2_cls=ai2_factory,
                n_games=1,
                select_random_prob=select_random_prob,
                emu_writer=emu_writer,
                train_writer=train_writer,
                **cfg.sampling
            )
            if opts.save:
                model_mgr.save_vars(sess)
                replay_memory.save(replay_dir)
            if opts.max_games is not None and i >= opts.max_games:
                break
    except KeyboardInterrupt:
        if opts.save:
            model_mgr.save_vars(sess)
            replay_memory.save(replay_dir)


if __name__ == '__main__':
    main()
