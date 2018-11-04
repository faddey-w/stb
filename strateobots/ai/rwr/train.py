import logging
import tensorflow as tf
import numpy as np
from functools import partial
from strateobots.engine import StbEngine, BotType
from strateobots import util
from strateobots.ai.lib.bot_initializers import duel_bot_initializer
from strateobots.ai.lib import integration, replay
from strateobots.ai.models import simple_ff
from strateobots.ai import simple_duel
from strateobots.ai.rwr.core import RewardWeightedRegression


log = logging.getLogger(__name__)


class RWRTraining:

    def __init__(self, session, storage_directory):
        self.sess = session
        self.model = simple_ff.Model('RWRModel')
        self.training_step = 0
        self.replay_memory = replay.ReplayMemory(storage_directory, self.model)
        self.rwr = RewardWeightedRegression(
            self.model,
            partial(reward_function, decay=0.99),
            batch_size=100
        )
        self.function = integration.ModelAiFunction(self.model, self.sess)

        self.bot_init = duel_bot_initializer(BotType.Raider, BotType.Raider, 0.8)
        self.bot_init_name = 'Duel RvR'
        self.n_batches_per_loop = 10

    def run_game(self, opponent):
        engine = StbEngine(
            world_width=1000,
            world_height=1000,
            ai1=self.function,
            ai2=opponent['function'],
            initialize_bots=self.bot_init,
            max_ticks=2000,
            wait_after_win=1,
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
        self.replay_memory.prepare_epoch(n_batches * self.rwr.batch_size, True)

        for i in range(n_batches):
            self.rwr.do_train_step(self.sess, self.replay_memory, i)
        self.training_step += 1

    def game_train_loop(self):
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
                pass


def reward_function(ticks, decay):
    ticks_remaining = ticks[..., 1] - ticks[..., 0]
    result = np.zeros_like(ticks_remaining)
    r = result[-1] = -1
    for i in range(np.size(ticks_remaining) - 2, -1, -1):
        next_r = decay * r
        r = result[i] = next_r
    return result


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    session = tf.Session()
    rwr_train = RWRTraining(session, '_data/RWR')
    session.run(rwr_train.model.init_op)
    session.run(rwr_train.rwr.init_op)
    rwr_train.game_train_loop()


if __name__ == '__main__':
    main()
