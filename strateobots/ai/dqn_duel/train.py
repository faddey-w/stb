import argparse
import datetime
import logging
import os

import tensorflow as tf

from strateobots.engine import BotType
from .core import get_session, ReinforcementLearning
from .model import QualityFunction
from ..lib import replay, model_saving, handcrafted
from ..lib.data import state2vec, action2vec
from . import ai

tf.reset_default_graph()
log = logging.getLogger(__name__)
REPO_ROOT = os.path.dirname(os.path.dirname(__import__('strateobots').__file__))


class Config:
    
    memory_capacity = 20000

    model_layers = (10, 8, 6, 4)

    batch_size = 80
    reward_prediction = 0.05
    select_random_prob_decrease = 0.005
    select_random_min_prob = 0.03
    self_play = False

    bot_type = BotType.Raider
        
    def make_common_modes(self):
        return [
            ai.NotMovingMode(),
            ai.LocateAtCircleMode(),
        ]
    
    def make_ai1_modes(self):
        return [
            ai.PassiveMode()
        ]

    def make_ai2_modes(self):
        return [
            ai.TrainerMode([handcrafted.turret_behavior])
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'play'], default='train')
    parser.add_argument('--no-save', action='store_false', dest='save')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--max-games', default=None, type=int)
    opts = parser.parse_args()

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
    cfg = Config()

    sess = get_session()
    try:
        model_mgr = model_saving.ModelManager.load_existing_model(model_dir)
        model_mgr.load_vars(sess)
        model = model_mgr.model  # type: QualityFunction.Model
    except:
        model = QualityFunction.Model(cfg.model_layers)
        model_mgr = model_saving.ModelManager(model, model_dir)
        model_mgr.init_vars(sess)
    replay_memory = replay.ReplayMemory(
        capacity=cfg.memory_capacity,
        state_size=state2vec.vector_length,
        action_size=action2vec.vector_length
    )
    rl = ReinforcementLearning(
        model,
        batch_size=cfg.batch_size,
        n_games=1,
        reward_prediction=cfg.reward_prediction,
        select_random_prob_decrease=cfg.select_random_prob_decrease,
        select_random_min_prob=cfg.select_random_min_prob,
        self_play=cfg.self_play,
        qfunc_class=model.__class__.QualityFunction
    )
    sess.run(rl.init_op)
    try:
        replay_memory.load(replay_dir)
        log.info('replay memory buffer loaded from %s', replay_dir)
    except FileNotFoundError:
        log.info('collecting new replay memory buffer')

    if opts.action == 'play':
        import code
        with sess.as_default():
            code.interact(local=dict(globals(), **locals()))
    if opts.action == 'train':
        try:
            i = 0
            while True:
                ai1_factory = ai.DQNDuelAI.parametrize(
                    bot_type=cfg.bot_type,
                    modes=cfg.make_common_modes() + cfg.make_ai1_modes()
                )
                ai2_factory = ai.DQNDuelAI.parametrize(
                    bot_type=cfg.bot_type,
                    modes=cfg.make_common_modes() + cfg.make_ai2_modes()
                )
                
                i += 1
                rl.run(
                    frameskip=2,
                    max_ticks=2000,
                    world_size=1000,
                    replay_memory=replay_memory,
                    log_root_dir=logs_dir if opts.save else None,
                    ai1_cls=ai1_factory,
                    ai2_cls=ai2_factory,
                    step_counter=model_mgr.step_counter
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
