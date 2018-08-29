import os
import datetime
import tensorflow as tf
from strateobots import REPO_ROOT
from strateobots.ai.lib import replay, data, handcrafted, model_saving
from . import core, model, runner


PATH_PREFIX = os.path.join(REPO_ROOT, '_data', '_overfit', '')


def train():
    n_games = 10
    batch_size = 80
    print_each = 17

    sess = tf.Session()
    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    print('Constructing model')
    # mdl = model.eventbased.QualityFunctionModelset.AllTheSame(
    #     linear_cfg=[(30, 30), (15, 30), (15, 30), (15, 30), (10, 10)],
    #     logical_cfg=[30, 30, 10],
    #     values_cfg=[(5, 10), (5, 10), (5, 10)],
    # )
    # mdl = model.classic.QualityFunctionModelset.AllTheSame(
    #     angle_sections=10,
    #     layer_sizes=[200, 100],
    # )
    # mdl = model.classic.QualityFunctionModelset.AllTheSame(
    #     angle_sections=15,
    #     layer_sizes=[100, 50],
    # )
    mdl = model.simple_logexp.QualityFunctionModelset.AllTheSame(
        cfg=[50, 50, 50, 50, 50],
    )
    # mdl = model.vec2d_v2.QualityFunctionModel(
    #     vec2d_cfg=[(17, 17)] * 6,
    #     fc_cfg=[11] * 6,
    # )
    # mdl = model.vec2d_v2.QualityFunctionModel(
    #     vec2d_cfg=[(11, 19)] * 2,
    #     fc_cfg=[31, 23, 17, 13, 7],
    # )
    # mdl = model.vec2d_v3.QualityFunctionModel(
    #     vec2d_cfg=[(9, 17)] * 6,
    #     fc_n_parts=20,
    #     fc_cfg=[3] * 20,
    # )
    # mdl = model.classic.QualityFunctionModel(
    #     angle_sections=10,
    #     layer_sizes=[200, 100]
    # )
    print('Allocate replay memory')
    mem = replay.ReplayMemory(
        2000 * n_games,
        data.state2vec.vector_length,
        data.action2vec.vector_length,
        data.state2vec.vector_length,
    )
    print('Constructing compute graph')
    rl = core.ReinforcementLearning(
        modelset=mdl,
        batch_size=batch_size,
        reward_prediction=0.97,
    )

    print('Initialization')
    ai1_function = core.ModelbasedFunction(mdl, sess)
    ai2_function = handcrafted.short_range_attack

    data_dir = PATH_PREFIX + 'replay'
    model_dir = PATH_PREFIX + run_name + '/model'
    logs_dir = PATH_PREFIX + run_name + '/logs'

    model_saving.ModelManager(mdl, model_dir)  # just save model source code
    writer = tf.summary.FileWriter(logs_dir)
    summary_op = tf.summary.scalar('loss', rl.loss)
    sess.run(tf.variables_initializer(mdl.var_list))
    sess.run(rl.init_op)

    try:
        print('Trying to load replay')
        mem.load(data_dir)
        print('Replay loaded from', data_dir)
    except:
        print('Generating new replay')
        os.makedirs(data_dir, exist_ok=True)
        for i in range(n_games):
            print('Game #{}/{}'.format(i+1, n_games))
            runner.run_one_game(
                replay_memory=mem,
                ai1_func=ai1_function,
                ai2_func=ai2_function,
            )
        mem.save(data_dir)

    print('Start training')
    loss_sum = 0
    loss_n = 0
    last_losses = []
    for i in range(2501):
        [loss, sumry] = rl.do_train_step(
            sess, mem,
            n_rnd_entries=batch_size,
            extra_tensors=[
                rl.loss,
                summary_op,
            ],
        )
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
        train()
    except KeyboardInterrupt:
        print('')

