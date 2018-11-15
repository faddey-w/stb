import logging
import argparse
import tensorflow as tf
import numpy as np
import time
from collections import defaultdict
from strateobots.ai.lib import replay, data, model_saving
from strateobots.ai.policy_learning.core import PolicyLearning
from strateobots.ai.models import simple_ff, classic


log = logging.getLogger(__name__)


class PLTraining:

    def __init__(self, session, storage_directory):
        self.sess = session

        self.model = simple_ff.Model('PLModel')
        # self.model = classic.Model('PLModel')

        self.replay_memory = replay.ReplayMemory(storage_directory, self.model,
                                                 _props_function,
                                                 load_winner_data=True,
                                                 load_loser_data=False,
                                                 )
        import code; code.interact(local=dict(**locals(), **globals()))
        self.pl = PolicyLearning(
            self.model,
            batch_size=300
        )

    def train_once(self, step_idx, n_batches):
        self.replay_memory.prepare_epoch(self.pl.batch_size, 0, n_batches, True)
        extra_t = self.pl.action_idx_ph, self.pl.inference.controls
        accuracies = defaultdict(float)

        for i in range(n_batches):
            act_idx, ctl_val = self.pl.do_train_step(self.sess, self.replay_memory, i, extra_t)
            for ctl in data.ALL_CONTROLS:
                pred_idx = np.argmax(ctl_val[ctl], 1)
                correct = act_idx[ctl] == pred_idx
                accuracies[ctl] += np.sum(correct) / np.size(pred_idx) / n_batches
        message = '#{}: {}'.format(step_idx, ' '.join(
            '{}={:.2f}'.format(ctl, accuracies[ctl])
            for ctl in data.ALL_CONTROLS
        ))
        log.info(message)
        return accuracies

    def train_loop(self, target_accuracy=None, max_iterations=None, max_time=None):
        def iterator():
            i = 0
            while i < max_iterations:
                yield i
                i += 1
        if max_iterations is None:
            max_iterations = float('inf')
        if max_time is not None:
            stop_time = time.time() + max_time
        else:
            stop_time = None
        log.info('Start training')
        for i in iterator():
            accuracies = self.train_once(i, 20)
            if target_accuracy is not None:
                if min(accuracies.values()) >= target_accuracy:
                    break
            if stop_time is not None:
                if time.time() >= stop_time:
                    log.info('stopped by timer')
                    break


def _props_function(replay_data, team, is_win):
    return np.zeros(len(replay_data)-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='_data/RWR')
    parser.add_argument('--n-batches', '-n', type=int)
    parser.add_argument('--max-accuracy', '-A', type=float)
    parser.add_argument('--max-time', '-t', type=float)
    parser.add_argument('--save-dir', '-S')
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    session = tf.Session()

    pl_train = PLTraining(session, opts.data_dir)
    session.run(pl_train.model.init_op)
    session.run(pl_train.pl.init_op)
    try:
        pl_train.train_loop(opts.max_accuracy, opts.n_batches, opts.max_time)
    except KeyboardInterrupt:
        pass

    if opts.save_dir:
        mgr = model_saving.ModelManager(pl_train.model, opts.save_dir)
        mgr.save_definition()
        mgr.save_vars(session)


if __name__ == '__main__':
    main()
