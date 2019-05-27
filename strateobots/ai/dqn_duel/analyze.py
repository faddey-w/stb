import argparse
import datetime
import logging
import os

import tensorflow as tf

from strateobots import REPO_ROOT
from strateobots.engine import BotType
from .core import ReinforcementLearning, ModelbasedFunction
from .core import noised_ai_func, compute_reward_from_engine
from ..lib import replay, model_saving, handcrafted
from ..lib.data import state2vec, action2vec
from . import model, runner

tf.reset_default_graph()
log = logging.getLogger(__name__)


class Config:
    memory_capacity = 100000

    batch_size = 120
    sampling = dict(
        n_seq_samples=0, seq_sample_size=0, n_rnd_entries=120, n_last_entries=0
    )
    reward_prediction = 0.97


def entrypoint(save_dir):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()

    model_dir = os.path.join(save_dir, "model", "")
    replay_dir = os.path.join(save_dir, "replay")

    sess = tf.Session()
    model_mgr = model_saving.ModelManager.load_existing_model(model_dir)
    model_mgr.load_vars(sess)
    mdl = model_mgr.model  # type: mdl.QualityFunctionModel
    mem = replay.ReplayMemory(
        cfg.memory_capacity,
        state2vec.vector_length,
        action2vec.vector_length,
        state2vec.vector_length,
    )
    mem.load(replay_dir)
    log.info("replay memory buffer loaded from %s", replay_dir)

    log.info("construct computation graphs")
    rl = ReinforcementLearning(
        mdl, batch_size=cfg.batch_size, reward_prediction=cfg.reward_prediction
    )
    bot_func = ModelbasedFunction(mdl, sess)

    log.info("initialize model variables")
    sess.run(rl.init_op)

    import code

    code.interact(local=dict(globals(), **locals()))


def main(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    opts = parser.parse_args(cmdline)
    entrypoint(save_dir=opts.save_dir)


if __name__ == "__main__":
    main()
