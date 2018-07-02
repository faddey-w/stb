import os
import logging
import datetime
import time
import itertools
import tensorflow as tf
from strateobots import REPO_ROOT
from strateobots.ai.lib import replay, data, handcrafted, model_saving
from strateobots.engine import BotType
from . import core, ai, model


class Config:

    n_games = 2
    batch_size = 80

    params = {}


def main():
    logging.basicConfig(level=logging.INFO)

    # cfg = Config()
    # cfg.params = dict(
    #     coord_cfg=(16, 16, 16, 16),
    #     angle_cfg=(16, 16, 16, 16),
    #     fc_cfg=(10, 10, 10, 10, 10, 10, 8),
    #     exp_layers=[],
    # )
    # train(cfg)
    search_params()


def C(n, k):
    def fact(x1, x2):
        r = 1
        for i in range(x1, x2+1):
            r *= i
        return r
    return fact(k+1, n) // fact(2, n-k)


def search_params():
    vec2d_sizes = [4, 8, 16, 32]
    fc_sizes = [6, 10, 16, 20]
    vec2d_depths = [2, 4, 6]
    fc_depths = [2, 4, 6, 8]
    exp_amounts = [0, 1]

    all_combinations = list(itertools.product(
        vec2d_sizes, fc_sizes,
        vec2d_depths, fc_depths,
        exp_amounts
    ))
    total_amount = len(all_combinations) // (len(exp_amounts) * len(fc_depths))
    total_amount *= sum(
        C(n, x) if n < 6 else C(n // 2, x)
        for x in exp_amounts for n in fc_depths
    )
    print("Total amount of attempts:", total_amount)
    print("Estimated time for all:", 5 * total_amount // 60,
          "hours (assuming that one run takes 5mins)")
    if input("proceed?  ") != 'YES':
        return

    i = 0
    skip = 82
    for vec_sz, fc_sz, vec_dep, fc_dep, n_exp in all_combinations:
        for exp_lrs in itertools.combinations(range(fc_dep), n_exp):
            if fc_dep >= 6 and any(x % 2 == 1 for x in exp_lrs):
                continue
            i += 1
            if i <= skip:
                continue

            graph = tf.Graph()
            graph.seed = 666
            core.reset_session()

            exp_lrs = list(exp_lrs)
            params = dict(
                coord_cfg=[vec_sz] * vec_dep,
                angle_cfg=[vec_sz] * vec_dep,
                fc_cfg=[fc_sz] * (fc_dep - 1) + [8],
                exp_layers=exp_lrs,
            )
            print('RUN #{}:'.format(i), ' '.join('{}={}'.format(k, v) for k, v in params.items()))
            started_at = time.time()

            cfg = Config()
            cfg.params = params
            with graph.as_default():
                run_name, loss = train(cfg, print_each=1000)
            duration = int(time.time() - started_at)

            better_name = 'loss:{:5.3f}__vec:{}-{}__fc:{}-{}__exp:{}'.format(
                loss,
                vec_dep, vec_sz,
                fc_dep, fc_sz,
                '-'.join(map(str, exp_lrs))
            )
            os.rename(PATH_PREFIX + run_name, PATH_PREFIX + better_name)
            print('done in {}:{} with loss={:5.3f}'.format(
                duration // 60,
                duration % 60,
                loss,
            ))


PATH_PREFIX = os.path.join(REPO_ROOT, '_data', '_overfit', '')


def train(cfg, print_each=17):
    mdl = model.QualityFunctionModel(**cfg.params)
    sess = core.get_session()
    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mem = replay.ReplayMemory(
        capacity=2000 * cfg.n_games,
        action_size=data.action2vec.vector_length,
        state_size=data.state2vec.vector_length,
    )
    rl = core.ReinforcementLearning(
        model=mdl,
        batch_size=cfg.batch_size,
        reward_prediction=0.95,
        self_play=False,
    )

    ai1_factory = ai.DQNDuelAI.parametrize(
        bot_type=BotType.Raider,
        modes=[
            ai.NotMovingMode(),
            ai.LocateAtCircleMode(),
            ai.NoShieldMode(),
        ]
    )
    ai2_factory = ai.DQNDuelAI.parametrize(
        bot_type=BotType.Raider,
        modes=[
            ai.NotMovingMode(),
            ai.LocateAtCircleMode(),
            ai.NoShieldMode(),
            ai.TrainerMode([handcrafted.turret_behavior])
        ]
    )
    data_dir = PATH_PREFIX + 'replay'
    model_dir = PATH_PREFIX + run_name + '/model'
    logs_dir = PATH_PREFIX + run_name + '/logs'

    model_saving.ModelManager(mdl, model_dir)  # just save model source code
    writer = tf.summary.FileWriter(logs_dir)
    sess.run(tf.variables_initializer(mdl.var_list))
    sess.run(rl.init_op)

    try:
        mem.load(data_dir)
    except:
        os.makedirs(data_dir, exist_ok=True)
        for _ in range(cfg.n_games):
            for mode in ai1_factory.modes + ai2_factory.modes:
                mode.reset()
            rl.run(
                replay_memory=mem,
                n_games=1,
                random_batch_size=cfg.batch_size,
                select_random_prob=0.8,
                world_size=1000,
                max_ticks=2000,
                frameskip=2,
                ai1_cls=ai1_factory,
                ai2_cls=ai2_factory,
            )
        mem.save(data_dir)

    loss_sum = 0
    loss_n = 0
    last_losses = []
    for i in range(7001):
        [loss, sumry] = rl.do_train_step(
            sess, mem,
            random_batch_size=cfg.batch_size,
            extra_tensors=[
                rl.loss,
                rl.train_summaries,
            ],
        )
        if i % 11 == 0:
            writer.add_summary(sumry, i)
        loss_sum += loss
        loss_n += 1

        if i % print_each == 0:
            print('#{}:\tloss={:6.4f}'.format(i, loss_sum / loss_n))
            loss_sum = 0
            loss_n = 0
        last_losses.append(loss)
        if len(last_losses) > 300:
            last_losses.pop(0)
    return run_name, sum(last_losses) / max(1, len(last_losses))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()

