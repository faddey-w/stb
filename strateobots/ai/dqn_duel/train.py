import argparse
import datetime
import logging
import os
import shlex

import tensorflow as tf

from strateobots import REPO_ROOT
from strateobots.engine import BotType
from .core import ReinforcementLearning, ModelbasedFunction
from ..lib import replay, model_saving, handcrafted
from ..lib.data import state2vec, action2vec
from . import ai, model, runner

tf.reset_default_graph()
log = logging.getLogger(__name__)


class Config:
    
    memory_capacity = 100000

    new_model_cls = model.classic.QualityFunctionModelset.AllTheSame
    # new_model_cls = model.eventbased.QualityFunctionModelset.AllTheSame

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
        # values_cfg=[(10, 20), (10, 20)],

        # vec2d_cfg=[(7, 11)] * 3,
        # fc_cfg=[40, 60],

        # n_parts=30,
        # cfg=[4] * 10,

        # cfg=[5],

        angle_sections=20,
        layer_sizes=[50, 50, 50, 1]
    )

    batch_size = 120
    sampling = dict(
        n_seq_samples=0,
        seq_sample_size=0,
        n_rnd_entries=120,
        n_last_entries=0,
    )
    reward_prediction = 0.97
    steps_between_games = 100

    bot_type = BotType.Raider


class GameReporter:

    def __init__(self):
        self.step = 0
        self.last_loss = 0
        self.final_hp1 = None
        self.final_hp2 = None

        self.final_hp1_ph = tf.placeholder(tf.float32)
        self.final_hp2_ph = tf.placeholder(tf.float32)
        self.summaries = tf.summary.merge([
            tf.summary.scalar('hp1', self.final_hp1_ph),
            tf.summary.scalar('hp2', self.final_hp2_ph),
        ])

    def __call__(self, engine):
        if engine.is_finished:
            log.info('Game #{}: hp1={:.2f} hp2={:.2f} loss={:.4f}'.format(
                self.step, engine.ai1.bot.hp_ratio, engine.ai2.bot.hp_ratio,
                self.last_loss,
            ))
            self.final_hp1 = engine.ai1.bot.hp_ratio
            self.final_hp2 = engine.ai2.bot.hp_ratio

    def write_summaries(self, session, writer):
        sumry = session.run(self.summaries, {
            self.final_hp1_ph: self.final_hp1,
            self.final_hp2_ph: self.final_hp2,
        })
        writer.add_summary(sumry, self.step)


def entrypoint(save_model, save_logs, save_dir, max_games):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    cfg = Config()

    if save_dir is None:
        run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_dir = os.path.join(REPO_ROOT, '_data', 'DQN', run_name)
    logs_dir = os.path.join(save_dir, 'logs')
    model_dir = os.path.join(save_dir, 'model', '')
    replay_dir = os.path.join(save_dir, 'replay')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    sess = tf.Session()
    try:
        model_mgr = model_saving.ModelManager.load_existing_model(model_dir)
        model_mgr.load_vars(sess)
        model = model_mgr.model  # type: model.QualityFunctionModel
    except Exception as exc:
        log.info("Cannot load model (%r), so creating a new one", exc)
        model = cfg.new_model_cls(**cfg.model_params)
        model_mgr = model_saving.ModelManager(model, model_dir)
        model_mgr.init_vars(sess)
    replay_memory = replay.ReplayMemory(
        cfg.memory_capacity,
        state2vec.vector_length,
        action2vec.vector_length,
        state2vec.vector_length,
    )
    try:
        replay_memory.load(replay_dir)
        log.info('replay memory buffer loaded from %s', replay_dir)
    except FileNotFoundError:
        log.info('collect new replay memory buffer')

    log.info('construct computation graphs')
    rl = ReinforcementLearning(
        model,
        batch_size=cfg.batch_size,
        reward_prediction=cfg.reward_prediction,
    )
    nn_func = ModelbasedFunction(model, sess)

    log.info('initialize model variables')
    sess.run(rl.init_op)

    log.info('start training')
    try:
        i = model_mgr.step_counter

        if save_logs:
            log_writer = tf.summary.FileWriter(
                os.path.join(logs_dir, 'train'),
                sess.graph
            )
        else:
            log_writer = None

        reporter = GameReporter()

        while True:

            reporter.step = i
            runner.run_one_game(
                replay_memory=replay_memory,
                ai1_func=nn_func,
                ai2_func=handcrafted.short_range_attack,
                report=reporter,
                frames_per_action=3,
            )
            reporter.last_loss = 0
            if save_logs:
                reporter.write_summaries(sess, log_writer)

            for _ in range(cfg.steps_between_games):
                i += 1
                [loss, sumry] = rl.do_train_step(
                    sess, replay_memory,
                    extra_tensors=[
                        rl.total_loss,
                        rl.summaries,
                    ],
                    **cfg.sampling,
                )
                if save_logs:
                    log_writer.add_summary(sumry, i)
                reporter.last_loss += loss
            reporter.last_loss /= cfg.steps_between_games

            if save_model:
                model_mgr.step_counter = i
                model_mgr.save_vars(sess)
                replay_memory.save(replay_dir)
            if max_games is not None and i >= max_games:
                break
    except KeyboardInterrupt:
        pass


def main(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--max-games', default=None, type=int)
    opts = parser.parse_args(cmdline)
    entrypoint(
        save_model=opts.save,
        save_logs=opts.save,
        save_dir=opts.save_dir,
        max_games=opts.max_games,
    )


if __name__ == '__main__':
    main()

