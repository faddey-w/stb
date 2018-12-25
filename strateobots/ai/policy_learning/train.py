import logging
import argparse
import tensorflow as tf
import numpy as np
import time
from collections import defaultdict
from strateobots.ai.lib import replay, data, model_saving
from strateobots.ai.policy_learning.core import PolicyLearning
from strateobots.ai.models import simple_ff, classic, radar


log = logging.getLogger(__name__)


class PLTraining:

    def __init__(self, session, storage_directory):
        self.sess = session

        model_module = radar
        src_path = model_module.__file__

        self.model = model_module.Model('PLModel')

        cache_key = model_saving.generate_model_hash(self.model, src_path)

        self.replay_memory = replay.ReplayMemory(
            storage_directory, self.model, _props_function,
            load_winner_data=True,
            load_loser_data=False,
            cache_key=cache_key,
            load_predicate=lambda md: 'Close distance attack' in (md['ai1_name'], md['ai2_name'])
        )
        # import code; code.interact(local=dict(**locals(), **globals()))
        self.pl = PolicyLearning(
            self.model,
            batch_size=300
        )

    def train_once(self, step_idx, n_batches):
        self.replay_memory.prepare_epoch(self.pl.batch_size, 0, n_batches, True)
        extra_t = self.pl.action_idx_ph, self.pl.inference.controls
        accuracies = defaultdict(float)

        for i in range(n_batches):
            _, (act_idx, ctl_val) = self.pl.compute_on_sample(
                self.sess,
                self.replay_memory,
                [self.pl.train_steps['tower_rotate'], extra_t],
                i,
            )
            for ctl in data.ALL_CONTROLS:
                pred_idx = np.argmax(ctl_val[ctl], 1)
                correct = act_idx[ctl] == pred_idx
                accuracies[ctl] += np.sum(correct) / np.size(pred_idx) / n_batches
        message = '#{}: {}'.format((step_idx+1) * n_batches, ' '.join(
            '{}={:.2f}'.format(ctl, accuracies[ctl])
            for ctl in data.ALL_CONTROLS
        ))
        log.info(message)
        return accuracies

    def __call__(self, tensor, prepare_new=False):
        if prepare_new:
            self.replay_memory.prepare_epoch(self.pl.batch_size, 0, 1, True)
        return self.pl.compute_on_sample(self.sess, self.replay_memory, tensor, 0)

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
    parser.add_argument('--data-dir', default='_data/PL')
    parser.add_argument('--n-batches', '-n', type=int)
    parser.add_argument('--max-accuracy', '-A', type=float)
    parser.add_argument('--max-time', '-t', type=float)
    parser.add_argument('--save-dir', '-S')
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--interactive-after', '-a', action='store_true')
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    session = tf.Session()

    pl_train = PLTraining(session, opts.data_dir)
    session.run(pl_train.model.init_op)
    session.run(pl_train.pl.init_op)

    plt = pl_train
    if opts.interactive:
        interactive(globals(), locals())
    try:
        pl_train.train_loop(opts.max_accuracy, opts.n_batches, opts.max_time)
    except KeyboardInterrupt:
        pass
    if opts.interactive_after:
        interactive(globals(), locals())

    if opts.save_dir:
        mgr = model_saving.ModelManager(pl_train.model, opts.save_dir)
        mgr.save_definition()
        mgr.save_vars(session)


def interactive(*nsdicts):
    ns = {}
    for d in nsdicts:
        ns.update(d)

    class Resume(SystemExit):
        pass

    def resume():
        raise Resume
    ns['resume'] = resume

    import code
    try:
        code.interact(local=ns)
    except Resume:
        pass


if __name__ == '__main__':
    main()
