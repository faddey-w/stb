import tensorflow as tf
import numpy as np
import math
import random
import time

from strateobots.ai.lib import data, datagen, nn, replay
from strateobots.util import objedict
from strateobots.ai.simple_duel import navigate_gun, norm_angle


num_labels = 3


def scale(scale_rate):
    return lambda x: x * scale_rate


space_data_schema = data.FeatureSet([
    data.Feature(['bot', 'x'], scale(0.001)),
    data.Feature(['bot', 'y'], scale(0.001)),
    data.Feature(['bot', 'orientation'], math.sin),
    data.Feature(['bot', 'orientation'], math.cos),
    data.Feature(['bot', 'tower_orientation'], math.sin),
    data.Feature(['bot', 'tower_orientation'], math.cos),
    data.Feature(['enemy', 'x'], scale(0.001)),
    data.Feature(['enemy', 'y'], scale(0.001)),
    data.Feature(['extra', 'angle_to_enemy'], math.sin),
    data.Feature(['extra', 'angle_to_enemy'], math.cos),
    data.Feature(['extra', 'gun_orientation'], math.sin),
    data.Feature(['extra', 'gun_orientation'], math.cos),
])
other_data_schema = data.FeatureSet([
    data.Feature(['bot', 'orientation'], norm_angle),
    data.Feature(['bot', 'tower_orientation'], norm_angle),
    data.Feature(['extra', 'angle_to_enemy'], norm_angle),
    data.Feature(['extra', 'gun_orientation'], norm_angle),
])


def encode(item):
    item = item.copy()
    angle_to_enemy = math.atan2(
        item['enemy']['y'] - item['bot']['y'],
        item['enemy']['x'] - item['bot']['x'],
    )
    gun_orientation = item['bot']['orientation'] + item['bot'][
        'tower_orientation']
    # delta_angle_nonnorm = angle_to_enemy - item['bot']['orientation'] - \
    #                       item['bot']['tower_orientation']
    # delta_angle = norm_angle(angle_to_enemy) - norm_angle(
    #     item['bot']['orientation']) - norm_angle(
    #     item['bot']['tower_orientation'])
    item['extra'] = {
        'angle_to_enemy': angle_to_enemy,
        # 'delta_angle_nonnorm': delta_angle_nonnorm,
        # 'delta_angle': delta_angle,
        'gun_orientation': gun_orientation,
        # 'answer': navigate_gun(**objedict(item)),
    }
    other_data = other_data_schema(item)
    vec1d = space_data_schema(item)
    vec2d = np.expand_dims(vec1d, 0) * np.expand_dims(vec1d, 1)
    vec2d = np.reshape(vec2d, [vec2d.size])
    vec3d = np.expand_dims(vec2d, 0) * np.expand_dims(vec1d, 1)
    vec3d = np.reshape(vec3d, [vec3d.size])
    return np.concatenate([
        vec1d,
        vec2d,
        vec3d,
        other_data,
    ])


class Model:

    _d = space_data_schema.dimension
    _o = other_data_schema.dimension
    state_dimension = _o + _d + _d**2 + _d**3

    def __init__(self):
        sd = space_data_schema.dimension
        n_exp = 0
        self.le = nn.Linear('LE', sd, n_exp)
        self.l1 = nn.Linear('L1', self.state_dimension + n_exp, 20)
        self.l2 = nn.Linear('L2', self.l1.out_dim, 20)
        self.l3 = nn.Linear('L3', self.l2.out_dim, num_labels)
        self.var_list = [
            *self.l1.var_list,
            *self.l2.var_list,
            *self.l3.var_list,
            *self.le.var_list,
        ]

    def apply(self, dataset_t):
        sd = space_data_schema.dimension
        space_data_t = dataset_t[..., :sd]
        exp_node = self.le.apply(tf.log(1 + tf.abs(space_data_t)), tf.exp)
        lin_input = tf.concat([dataset_t, exp_node.out], -1)
        node1 = self.l1.apply(lin_input, tf.tanh)
        node2 = self.l2.apply(node1.out, tf.nn.leaky_relu)
        node3 = self.l3.apply(node2.out, tf.identity)
        return (
            node1,
            node2,
            node3,
            exp_node,
        ), node3.out

    @staticmethod
    def encode_prev_state(**item):
        return np.array([], np.float)

    @staticmethod
    def encode_state(**item):
        return encode({'bot': item['bot'], 'enemy': item['enemy']})


# dataset_loading_policy = 'generate'
dataset_loading_policy = 'load'
if dataset_loading_policy == 'generate':

    gen_item = datagen.structure({
        'bot': {
            'x': datagen.value(0, 1000),
            'y': datagen.value(0, 1000),
            'orientation': datagen.value(-10 * math.pi, 10 * math.pi),
            'tower_orientation': datagen.value(-10 * math.pi, 10 * math.pi),
        },
        'enemy': {
            'x': datagen.value(0, 1000),
            'y': datagen.value(0, 1000),
        }
    })

    dataset_json = [
        gen_item() for i in range(1000)
    ]
    answers = np.array([navigate_gun(**objedict(item)) for item in dataset_json])
    labels = answers + 1
    dataset = np.array(list(map(encode, dataset_json)))
elif dataset_loading_policy == 'load':
    rm = replay.ReplayMemory('_data/RWR', Model,
                             lambda replay_data, team, is_win: np.zeros(
                                 len(replay_data) - 1),
                             load_winner_data=True,
                             load_loser_data=False)
    rm.prepare_epoch(rm.total_items(), 0, 1)
    _, ctl_data, state_data, _ = rm.get_prepared_epoch_batch(0)
    labels = ctl_data['tower_rotate']
    dataset = state_data
    # import code; code.interact(local=globals())
else:
    raise NotImplementedError(dataset_loading_policy)

graph = tf.Graph()
with graph.as_default():
    model = Model()

    tf_dataset = tf.placeholder(tf.float32, [None, model.state_dimension])
    tf_label_idx = tf.placeholder(tf.int64, [None])
    tf_labels = tf.one_hot(tf_label_idx, num_labels)

    nodes, logits = model.apply(tf_dataset)
    train_prediction = tf.nn.softmax(logits)

    loss_vector = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_labels)
    # mask = tf.stop_gradient(tf.to_float(tf.not_equal(tf.argmax(logits, 1), tf_label_idx)))
    # loss = tf.reduce_mean(mask * loss_vector)
    loss = tf.reduce_mean(loss_vector)
    # regularizer = tf.add_n([tf.nn.l2_loss(v) for v in model.var_list])
    # beta = 0.01
    # loss = tf.reduce_mean(loss + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    init_op = tf.global_variables_initializer()


num_steps = 2001


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
            / predictions.shape[0])

session = tf.Session(graph=graph)

session.run(init_op)
print('Initialized')
start = time.time()
spent_on_measurements = 0
for step in range(num_steps):
    sample = tuple(random.sample(range(dataset.shape[0]), 300))
    _, l = session.run([optimizer, loss], {
        tf_dataset: dataset[sample, ],
        tf_label_idx: labels[sample, ],
    })
    if step % 50 == 0:
        measure_start = time.time()
        l, predictions = session.run([loss, train_prediction], {
            tf_dataset: dataset,
            tf_label_idx: labels,
        })
        acc = accuracy(predictions, labels)
        now = time.time()
        spent_on_measurements += now - measure_start
        elapsed = now - start - spent_on_measurements
        print('#{}: t={:6.2f} loss={:.4f} acc={:.1f}'.format(step, elapsed, l, acc))
        if acc >= 95:
            break


def check(item, tensors=train_prediction):
    itemdata = encode(item)
    itemlabel = navigate_gun(**objedict(item)) + 1
    return session.run(tensors, {
        tf_dataset: [itemdata],
        tf_label_idx: [itemlabel],
    })


def check_everything(rd_items):
    skipped = []
    for g, rd in enumerate(rd_items):
        wt = rd.winner_team
        lt = rd.loser_team
        if None in (wt, lt):
            skipped.append((g, None))
            continue
        for t, tick in enumerate(rd.json_data):
            try:
                bot = tick['bots'][wt][0]
                enemy = tick['bots'][lt][0]
                actual = tick['controls'][wt][0]['tower_rotate']
            except IndexError:
                skipped.append((g, t))
                continue
            else:
                expected = navigate_gun(objedict(bot), objedict(enemy))
                if expected != actual:
                    print('g={} t={}'.format(g, t))
    return skipped
