import logging
import argparse
import shutil
import os
import tensorflow as tf
import numpy as np
from strateobots.engine import StbEngine, BotType
from strateobots import util
from strateobots.ai.lib.bot_initializers import random_bot_initializer
from strateobots.ai.lib import model_function, replay
from strateobots.ai.models import ff_direct
from strateobots.ai import simple_duel
from strateobots.ai.rwr.core import RewardWeightedRegression


log = logging.getLogger(__name__)


def reward_function(rd, decay=0.995):
    n = rd.ticks.size - 1
    team = rd.team
    opp_team = rd.opponent_team
    value = 0
    if rd.json_data[-1]['bots'][team]:
        value += _get_hp(rd, team, -1)
    if rd.json_data[-1]['bots'][opp_team]:
        value -= _get_hp(rd, opp_team, -1)
    reward = np.zeros([n])
    # reward += value * 500 / n
    r = value / decay

    for i in range(n-1, -1, -1):
        hp_next = _get_hp(rd, opp_team, i+1)
        hp_prev = _get_hp(rd, opp_team, i)
        imm_r = hp_prev - hp_next
        r = decay * r + imm_r
        reward[i] += r

    reward *= 100
    return reward


def load_predicate(metadata, t1, t2):
    return metadata['team1'] == str(t1)


def _get_hp(rd, team, tick):
    if rd.json_data[tick]['bots'][team]:
        return rd.json_data[tick]['bots'][team][0]['hp']
    else:
        return 0.0


class RWRTraining:

    def __init__(self, session, storage_directory, logs_dir=None):
        self.sess = session
        self.model = ff_direct.Model('RWRModel')
        self.training_step = 0
        self.replay_memory = replay.ReplayMemory(storage_directory, self.model,
                                                 reward_function,
                                                 controls=self.model.control_set,
                                                 max_games_keep=20)
        self.replay_memory.reload(load_predicate=load_predicate)
        self.rwr = rwr = RewardWeightedRegression(
            self.model,
            batch_size=500,
            entropy_coeff=0.1,
        )
        self.function = model_function.ModelAiFunction(self.model, self.sess)

        self.bot_init = random_bot_initializer([BotType.Raider], [BotType.Raider])
        self.bot_init_name = 'Random RvR'
        self.n_batches_per_loop = 1

        if logs_dir is not None:
            actions_entropy = [
                tf.summary.scalar('Losses/Entropy_{}'.format(ctl), entropy)
                for ctl, entropy in rwr.entropy.items()
            ]
            self.train_summary_op = tf.summary.merge([
                tf.summary.scalar('Losses/Loss', rwr.loss),
                tf.summary.scalar('Losses/Entropy', rwr.full_entropy),
                tf.summary.scalar('Losses/VarsNorm', rwr.vars_norm),
                tf.summary.scalar('Losses/GradsNorm', rwr.grads_norm),
                *actions_entropy,
            ])
            self.log_writer = tf.summary.FileWriter(logs_dir)
        else:
            self.log_writer = None
            self.train_summary_op = None

    def run_game(self, opponent):
        engine = StbEngine(
            ai1=self.function,
            ai2=opponent['function'],
            initialize_bots=self.bot_init,
            max_ticks=2000,
            wait_after_win_ticks=0,
            stop_all_after_finish=True,
        )
        metadata = util.make_metadata_before_game(
            init_name=self.bot_init_name,
            ai1_module=self.__class__.__module__.rpartition('.')[0],
            ai1_name='{} step={}'.format(self.model.name, self.training_step),
            ai2_module=opponent['module'],
            ai2_name=opponent['name'],
        )
        while not engine.is_finished:
            engine.tick()
        util.fill_metadata_after_game(metadata, engine)

        t1_alive = len(engine.replay[-1]['bots'][engine.team1]) > 0
        t2_alive = len(engine.replay[-1]['bots'][engine.team2]) > 0
        if t1_alive:
            t1_hp = engine.replay[-1]['bots'][engine.team1][0]['hp']
        else:
            t1_hp = 0
        if t2_alive:
            t2_hp = engine.replay[-1]['bots'][engine.team2][0]['hp']
        else:
            t2_hp = 0
        log.info('GAME #{}: T1={} T2={}'.format(
            self.training_step,
            '{}%hp'.format(int(100*t1_hp)) if t1_alive else 'lost',
            '{}%hp'.format(int(100*t2_hp)) if t2_alive else 'lost',
        ))

        with util.interrupt_atomic():
            return self.replay_memory.add_replay(metadata, engine.replay,
                                                 load_predicate=load_predicate)

    def train_once(self, n_batches):
        # win_batch = int(self.rwr.batch_size * self.win_lost_proportion)
        # lost_batch = self.rwr.batch_size - win_batch
        self.replay_memory.prepare_epoch(self.rwr.batch_size, n_batches, shuffle=True)

        summary = None
        for i in range(n_batches):
            summary = self.rwr.do_train_step(
                self.sess, self.replay_memory, i,
                self.train_summary_op)
        self.training_step += 1
        return summary

    def game_train_loop(self, n_games):
        log.info('Start training')
        opponent = {
            'function': simple_duel.CloseDistanceAttack(),
            'name': 'Close distance attack',
            'module': simple_duel.__name__,
        }
        episode_count = 0
        while True:
            episode_count += 1
            [rd] = self.run_game(opponent)
            try:
                train_summary = self.train_once(self.n_batches_per_loop)
            except replay.NotEnoughData:
                log.info('not enough data')
            else:
                if train_summary:
                    self.log_writer.add_summary(train_summary, episode_count)

                perf_summary = tf.summary.Summary()
                perf_summary.value.add(
                    tag='Perf/Length',
                    simple_value=float(len(rd.json_data)),
                )
                perf_summary.value.add(
                    tag='Perf/BotHP',
                    simple_value=float(_get_hp(rd, rd.team, -1)),
                )
                perf_summary.value.add(
                    tag='Perf/EnemyHP',
                    simple_value=float(_get_hp(rd, rd.opponent_team, -1)),
                )
                perf_summary.value.add(
                    tag='Perf/Reward',
                    simple_value=float(np.mean(reward_function(rd))),
                )
                self.log_writer.add_summary(perf_summary, episode_count)

                self.log_writer.flush()
            if n_games is not None:
                n_games -= 1
                if n_games <= 0:
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop-data', action='store_true')
    parser.add_argument('--drop-logs', action='store_true')
    parser.add_argument('--drop-all', action='store_true')
    parser.add_argument('--data-dir', default='_data/RWR')
    parser.add_argument('--logs-dir', default='_data/logs/RWR')
    parser.add_argument('--n-games', '-n', type=int)
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    session = tf.Session()

    if opts.drop_data or opts.drop_all:
        if os.path.exists(opts.data_dir):
            shutil.rmtree(opts.data_dir)
    if opts.drop_logs or opts.drop_all:
        if os.path.exists(opts.logs_dir):
            shutil.rmtree(opts.logs_dir)
    os.makedirs(opts.logs_dir, exist_ok=True)

    rwr_train = RWRTraining(session, opts.data_dir, opts.logs_dir)
    session.run(rwr_train.model.init_op)
    session.run(rwr_train.rwr.init_op)
    rwr_train.game_train_loop(opts.n_games)
    # with session.as_default():
    #     import code; code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
