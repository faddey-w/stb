import logging
import argparse
import tensorflow as tf
import numpy as np
import time
from collections import defaultdict
from strateobots.ai.lib import replay, data, model_saving
from strateobots.ai.policy_learning.core import PolicyLearning
from strateobots.ai.models import ff_aim_angle, classic, radar, visual


log = logging.getLogger(__name__)


class PLTraining:
    def __init__(self, session, storage_directory, batch_size=50, epoch_size=50):
        self.sess = session
        self.batch_size = batch_size
        self.epoch_size = epoch_size

        # model_module = visual
        # model_module = radar
        model_module = ff_aim_angle
        src_path = model_module.__file__

        # import pdb; pdb.set_trace()
        self.model = model_module.Model()

        cache_key = model_saving.generate_model_hash(self.model, src_path)

        def load_predicate(metadata, t1, t2):
            return t1 == "56576"
            # winner = metadata['winner']
            # if winner is None:
            #     return False
            # team_id = 1 if winner == metadata['team1'] else 2
            # return 'Close distance attack' == metadata['ai{}_name'.format(team_id)]

        self.replay_memory = replay.ReplayMemory(
            storage_directory,
            self.model,
            _props_function,
            cache_key=cache_key,
            controls=self.model.control_set,
        )
        self.replay_memory.reload(load_predicate=load_predicate)
        # import code; code.interact(local=dict(**locals(), **globals()))
        self.pl = PolicyLearning(self.model, batch_size=self.batch_size)

    def train_once(self, step_idx, n_batches):
        self.replay_memory.prepare_epoch(self.pl.batch_size, n_batches, True)
        accuracies = defaultdict(float)
        loss, regularizer = 0, 0

        for i in range(n_batches):
            l, r, _, acc = self.pl.compute_on_sample(
                self.sess,
                self.replay_memory,
                [
                    self.pl.loss,
                    self.pl.regularizer,
                    self.pl.train_step,
                    self.pl.accuracies,
                ],
                i,
            )
            for ctl in self.model.control_set:
                accuracies[ctl] += acc[ctl] / n_batches
            loss += l / n_batches
            regularizer += r / n_batches
        message = "#{}: loss={:.4f}+{:.4f} {}".format(
            (step_idx + 1) * n_batches,
            loss,
            regularizer,
            " ".join(
                "{}={:.4f}".format(ctl, accuracies[ctl])
                for ctl in self.model.control_set
            ),
        )
        log.info(message)
        return accuracies

    def __call__(self, tensor, prepare_new=False, shuffle=True):
        if prepare_new:
            self.replay_memory.prepare_epoch(self.pl.batch_size, 0, 1, shuffle)
        return self.pl.compute_on_sample(self.sess, self.replay_memory, tensor, 0)

    def train_loop(self, target_accuracy=None, n_batches=None, max_time=None):
        def iterator():
            i = 0
            while i < max_iterations:
                yield i
                i += 1

        if n_batches is None:
            max_iterations = float("inf")
        else:
            max_iterations = n_batches / self.epoch_size
        if max_time is not None:
            stop_time = time.time() + max_time
        else:
            stop_time = None
        log.info("Start training")
        for i in iterator():
            accuracies = self.train_once(i, self.epoch_size)
            if target_accuracy is not None:
                if min(accuracies.values()) >= target_accuracy:
                    break
            if stop_time is not None:
                if time.time() >= stop_time:
                    log.info("stopped by timer")
                    break


def _props_function(rd):
    return np.zeros(rd.ticks.size - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="_data/PL")
    parser.add_argument("--n-batches", "-n", type=int)
    parser.add_argument("--epoch-size", "-E", type=int, default=50)
    parser.add_argument("--batch-size", "-B", type=int, default=50)
    parser.add_argument("--max-accuracy", "-A", type=float)
    parser.add_argument("--max-time", "-t", type=float)
    parser.add_argument("--save-dir", "-S")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--interactive-after", "-a", action="store_true")
    opts = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    session = tf.Session()

    pl_train = PLTraining(
        session, opts.data_dir, batch_size=opts.batch_size, epoch_size=opts.epoch_size
    )
    session.run(pl_train.model.init_op)
    session.run(pl_train.pl.init_op)

    plt = pl_train
    if opts.interactive:
        with session.as_default():
            interactive(globals(), locals())
    try:
        pl_train.train_loop(opts.max_accuracy, opts.n_batches, opts.max_time)
    except KeyboardInterrupt:
        log.info("interrupted")
    if opts.interactive_after:
        with session.as_default():
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

    ns["resume"] = resume

    import code

    try:
        code.interact(local=ns)
    except Resume:
        pass


if __name__ == "__main__":
    main()
