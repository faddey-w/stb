import logging
import argparse
import shutil
import os
import tensorflow as tf
import numpy as np
from strateobots.engine import StbEngine, BotType
from strateobots import util
from strateobots.ai.lib.bot_initializers import duel_bot_initializer
from strateobots.ai.lib import integration, replay
from strateobots.ai.models import simple_ff
from strateobots.ai import simple_duel
from strateobots.ai.rwr.core import RewardWeightedRegression


log = logging.getLogger(__name__)


def reward_function(replay_data, team, is_win,
                    decay=0.99):
    n = len(replay_data)-1
    opp_team = (set(replay_data[0]['bots']) - {team}).pop()
    reward = np.zeros([n])
    reward += 100 / n * (+1 if is_win else -1)
    r = 0.0

    def get_hp(tick):
        bots = replay_data[tick]['bots'][opp_team]
        if bots:
            return bots[0]['hp']
        else:
            return 0.0 if tick > 0 else 1.0

    for i in range(n-1, -1, -1):
        hp_next = get_hp(i+1)
        hp_prev = get_hp(i)
        imm_r = hp_prev - hp_next
        r = decay * r + imm_r
        reward[i] += r

    return reward


class RWRTraining:

    def __init__(self, session, storage_directory):
        self.sess = session
        self.model = simple_ff.Model('RWRModel')
        self.training_step = 0
        self.replay_memory = replay.ReplayMemory(storage_directory, self.model,
                                                 reward_function,
                                                 load_winner_data=True)
        self.rwr = RewardWeightedRegression(
            self.model,
            batch_size=100
        )
        self.function = integration.ModelAiFunction(self.model, self.sess)

        self.bot_init = duel_bot_initializer(BotType.Raider, BotType.Raider, 0.8)
        self.bot_init_name = 'Duel RvR'
        self.n_batches_per_loop = 10
        self.win_lost_proportion = 0.0

    def run_game(self, opponent):
        engine = StbEngine(
            world_width=1000,
            world_height=1000,
            ai1=self.function,
            ai2=opponent['function'],
            initialize_bots=self.bot_init,
            max_ticks=2000,
            wait_after_win_ticks=0,
            stop_all_after_finish=True,
        )
        metadata = dict(
            init_name=self.bot_init_name,
            ai1_module=self.__class__.__module__.rpartition('.')[0],
            ai1_name='{} step={}'.format(self.model.name, self.training_step),
            ai2_module=opponent['module'],
            ai2_name=opponent['name'],
        )
        while not engine.is_finished:
            engine.tick()
        metadata['nticks'] = engine.nticks

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
            self.replay_memory.add_replay(metadata, engine.replay)

    def train_once(self, n_batches):
        win_batch = int(self.rwr.batch_size * self.win_lost_proportion)
        lost_batch = self.rwr.batch_size - win_batch
        self.replay_memory.prepare_epoch(win_batch, lost_batch, n_batches, True)

        for i in range(n_batches):
            self.rwr.do_train_step(self.sess, self.replay_memory, i)
        self.training_step += 1

    def game_train_loop(self, n_games):
        log.info('Start training')
        opponent = {
            'function': simple_duel.short_range_attack,
            'name': 'Close distance attack',
            'module': simple_duel.__name__,
        }
        while True:
            self.run_game(opponent)
            try:
                self.train_once(self.n_batches_per_loop)
            except replay.NotEnoughData:
                log.info('not enough data')
            if n_games is not None:
                n_games -= 1
                if n_games <= 0:
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop-data', action='store_true')
    parser.add_argument('--data-dir', default='_data/RWR')
    parser.add_argument('--n-games', '-n', type=int)
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    session = tf.Session()

    if opts.drop_data:
        if os.path.exists(opts.data_dir):
            shutil.rmtree(opts.data_dir)

    rwr_train = RWRTraining(session, opts.data_dir)
    session.run(rwr_train.model.init_op)
    session.run(rwr_train.rwr.init_op)
    rwr_train.game_train_loop(opts.n_games)


if __name__ == '__main__':
    main()
