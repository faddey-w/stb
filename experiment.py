import argparse
import importlib
import subprocess
import threading
import itertools
import sys
from strateobots.engine import StbEngine
from strateobots.models import BotType
from strateobots import replay, util, bot_initializers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker', 'supervisor'])
    opts, argv = parser.parse_known_args()

    if opts.mode == 'worker':
        worker(**worker_argparse(argv))
    elif opts.mode == 'supervisor':
        supervisor(**supervisor_argparse(argv))


def supervisor_argparse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('storage_dir')
    parser.add_argument('n_workers', type=int)
    parser.add_argument('--suffix-first', type=int, default=1)
    parser.add_argument('--suffix-last', type=int, default=5)
    return vars(parser.parse_args(argv))


def supervisor(storage_dir, n_workers, suffix_first, suffix_last):
    ai_names = {
        'strateobots.ai.simple_duel': [
            'short',
            'distance',
            'ramming'
        ],
        'strateobots.ai.treesearch': [
            # 'MCTS-1',
            'MCTS-2',
            # 'MCTS-3',
        ],
    }
    one_ai_cases = [
        (ai_path, ai_name, t)
        for ai_path, ai_list in ai_names.items()
        for ai_name in ai_list
        for t in 'RTL'
    ]
    cases = [
        (t1, t2, ai1_path, ai1_name, ai2_path, ai2_name, suffix)
        for (ai1_path, ai1_name, t1), (ai2_path, ai2_name, t2) in itertools.combinations(one_ai_cases, 2)
        for suffix in map(str, range(suffix_first, suffix_last+1))
    ]
    counter = AtomicInteger()
    total = len(cases)

    semaphore = threading.Semaphore(n_workers)
    threads = []
    for case in cases:
        cmd = [sys.executable, sys.argv[0], 'worker', storage_dir, *case]
        t = threading.Thread(target=_subprocess_waiter,
                             args=(semaphore, cmd, counter, total))
        t.setDaemon(True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def _subprocess_waiter(semaphore, command, counter, total):
    semaphore.acquire()
    subprocess.check_call(command)
    semaphore.release()
    counter.inc()
    print('{} / {}'.format(counter.value, total))


def worker_argparse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('storage_dir')
    parser.add_argument('t1')
    parser.add_argument('t2')
    parser.add_argument('ai1_path')
    parser.add_argument('ai1_name')
    parser.add_argument('ai2_path')
    parser.add_argument('ai2_name')
    parser.add_argument('suffix')
    return vars(parser.parse_args(argv))


def worker(storage_dir, t1, t2, ai1_path, ai1_name, ai2_path, ai2_name, suffix):

    description = '{}v{}_{}_{}_{}_{}_{}'.format(
        t1, t2,
        ai1_path.rpartition('.')[2].replace('_', '-'), ai1_name,
        ai2_path.rpartition('.')[2].replace('_', '-'), ai2_name,
        suffix
    )

    typemap = {'R': BotType.Raider, 'T': BotType.Heavy, 'L': BotType.Sniper}

    initializer = bot_initializers.DuelInitializer(typemap[t1], typemap[t2], 0.7)

    ai1_mod = importlib.import_module(ai1_path).AIModule()
    ai1 = ai1_mod.construct_ai_function(StbEngine.TEAMS[0], ai1_name)

    ai2_mod = importlib.import_module(ai2_path).AIModule()
    ai2 = ai2_mod.construct_ai_function(StbEngine.TEAMS[1], ai2_name)

    engine = StbEngine(
        ai1=ai1,
        ai2=ai2,
        initialize_bots=initializer,
        max_ticks=2000,
        wait_after_win=1,
        stop_all_after_finish=True,
    )
    while not engine.is_finished:
        engine.tick()

    h1 = h2 = 0
    for bot in engine.iter_bots():
        if bot.team == engine.team1:
            h1 = max(h1, bot.hp_ratio)
        else:
            h2 = max(h2, bot.hp_ratio)

    with util.interrupt_atomic():
        storage = replay.ReplayDataStorage(storage_dir)
        storage.save_replay(description, dict(
            init_name='Duel {}v{}'.format(t1, t2),
            ai1_module=ai1_mod.name,
            ai1_name=ai1_name,
            ai2_module=ai2_mod.name,
            ai2_name=ai2_name,
            descriptor=(t1, t2, ai1_path, ai1_name, ai2_path, ai2_name),
            winner=+1 if h1 > h2 else (-1 if h1 < h2 else 0),
            nticks=engine.nticks,
        ), engine.replay)

        print('DONE: {}'.format(description))


class AtomicInteger:
    def __init__(self, value=0):
        self._value = value
        self._lock = threading.Lock()

    def inc(self):
        with self._lock:
            self._value += 1
            return self._value

    def dec(self):
        with self._lock:
            self._value -= 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = v


if __name__ == '__main__':
    main()
