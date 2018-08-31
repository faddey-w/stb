import argparse
import datetime
import logging
import os

import tensorflow as tf

from strateobots import REPO_ROOT
from strateobots.engine import BotType
from .core import ReinforcementLearning
from .core import noised_ai_func, compute_reward_from_engine
from ..lib import replay, model_saving, handcrafted
from ..lib.data import state2vec, action2vec
from . import model, runner

tf.reset_default_graph()
log = logging.getLogger(__name__)


class Config:
    
    memory_capacity = 20000

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

        angle_sections=36,
        layer_sizes=[75, 75, 75, 1]
    )

    batch_size = 50
    sampling = dict(
        n_seq_samples=0,
        seq_sample_size=0,
        n_rnd_entries=50,
        n_last_entries=0,
    )
    reward_prediction = 0.97
    steps_between_games = 100

    bot_type = BotType.Raider


class GameReporter:

    def __init__(self, reward_prediction):
        self.step = 0

        self.loss_sum = 0.0
        self.loss_n = 0

        self.reward_prediction = reward_prediction
        self._rewards = []
        self._rew_comp = None
        self._q_values = []

        self.final_hp1 = None
        self.final_hp2 = None

        self.final_hp1_ph = tf.placeholder(tf.float32)
        self.final_hp2_ph = tf.placeholder(tf.float32)
        self.loss_ph = tf.placeholder(tf.float32)
        self.q_dev_ph = tf.placeholder(tf.float32)
        self.summaries = tf.summary.merge([
            tf.summary.scalar('hp1', self.final_hp1_ph),
            tf.summary.scalar('hp2', self.final_hp2_ph),
            tf.summary.scalar('loss', self.loss_ph),
            tf.summary.scalar('Q_dev', self.q_dev_ph),
        ])

    def __call__(self, engine):
        if self._rew_comp is None:
            self._rew_comp = compute_reward_from_engine(engine)
        else:
            self._rewards.append(self._rew_comp.get_next(engine))

        self._q_values.append(engine.ai1.function.max_q)

        if engine.is_finished:
            self.final_hp1 = engine.ai1.bot.hp_ratio
            self.final_hp2 = engine.ai2.bot.hp_ratio

    def write_summaries(self, session, writer):

        q_target_sum = 0.0
        full_rew = 0.0
        rp = self.reward_prediction
        for rew in self._rewards[::-1]:
            full_rew *= rp
            full_rew += rew
            q_target_sum += full_rew
        q_target_avg = q_target_sum / len(self._rewards)
        q_nn_avg = sum(self._q_values) / len(self._q_values)

        sumry = session.run(self.summaries, {
            self.final_hp1_ph: self.final_hp1,
            self.final_hp2_ph: self.final_hp2,
            self.loss_ph: self.loss,
            self.q_dev_ph: q_nn_avg - q_target_avg,
        })
        writer.add_summary(sumry, self.step)

    def add_loss(self, value):
        self.loss_sum += value
        self.loss_n += 1

    @property
    def loss(self):
        return self.loss_sum / max(1, self.loss_n)

    def reset(self):
        self.loss_sum = 0.0
        self.loss_n = 0
        self._rewards[:] = []
        self._q_values[:] = []

    def print_report(self):
        log.info('Game #{}: hp1={:.2f} hp2={:.2f} loss={:.4f}'.format(
            self.step,
            self.final_hp1,
            self.final_hp2,
            self.loss,
        ))


def entrypoint(save_model, save_logs, save_dir, max_games, eval_train_ratio,
               n_full_explore, n_decreasing_explore):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    cfg = Config()

    save_dir_was_none = save_dir is None
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
    except Exception:
        if save_dir_was_none:
            log.info("Creating new model at %s", save_dir)
        else:
            log.exception("Cannot load model, so creating a new one")
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
    bot_func = model.make_function(sess)
    noised_bot_func = noised_ai_func(bot_func, 0.1)

    log.info('initialize model variables')
    sess.run(rl.init_op)

    log.info('start training')
    n_evals, n_total_cycle = eval_train_ratio
    try:
        if save_logs:
            train_log_writer = tf.summary.FileWriter(
                os.path.join(logs_dir, 'train'),
                sess.graph
            )
            eval_log_writer = tf.summary.FileWriter(
                os.path.join(logs_dir, 'eval'),
                sess.graph
            )
        else:
            train_log_writer = None
            eval_log_writer = None

        reporter = GameReporter(cfg.reward_prediction)
        reporter.step = model_mgr.step_counter

        while True:
            reporter.reset()

            if reporter.step % n_total_cycle < n_evals:
                log_writer = eval_log_writer
                run_func = bot_func
            else:
                prob = 1.0 - (model_mgr.step_counter - n_full_explore) / n_decreasing_explore
                noised_bot_func.noise_prob = max(0.1, min(1.0, prob))
                log_writer = train_log_writer
                run_func = noised_bot_func

            runner.run_one_game(
                replay_memory=replay_memory,
                ai1_func=run_func,
                ai2_func=handcrafted.short_range_attack,
                report=reporter,
                frames_per_action=3,
            )

            for _ in range(cfg.steps_between_games):
                [loss] = rl.do_train_step(
                    sess, replay_memory,
                    extra_tensors=[
                        rl.total_loss,
                    ],
                    **cfg.sampling,
                )
                reporter.add_loss(loss)
            if save_logs:
                reporter.write_summaries(sess, log_writer)
            reporter.print_report()

            if save_model:
                replay_memory.save(replay_dir)
                model_mgr.save_vars(sess)
            reporter.step += 1
            if max_games is not None and model_mgr.step_counter >= max_games:
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

