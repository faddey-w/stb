import tensorflow as tf
from strateobots.engine import BulletModel


class Average:
    def __init__(self):
        self.value = 0
        self.n = 0

    def reset(self):
        self.value = 0
        self.n = 0

    def add(self, x):
        self.value += x
        self.n += 1

    def get(self):
        return self.value / max(1, self.n)


def normalize_state(state):
    normalizer = tf.one_hot(
        [
            state2vec[0, "x"],
            state2vec[0, "y"],
            state2vec[1, "x"],
            state2vec[1, "y"],
            state2vec[2, "x"],
            state2vec[2, "y"],
            state2vec[3, "x"],
            state2vec[3, "y"],
        ],
        depth=state2vec.vector_length,
        on_value=1.0 / 1000,
        off_value=1.0,
    )
    normalizer = tf.reduce_min(normalizer, 0)
    state *= normalizer

    normalizer = tf.one_hot(
        [
            state2vec[0, "vx"],
            state2vec[0, "vy"],
            state2vec[1, "vx"],
            state2vec[1, "vy"],
        ],
        depth=state2vec.vector_length,
        on_value=1.0 / 10,
        off_value=1.0,
    )
    normalizer = tf.reduce_min(normalizer, 0)
    state *= normalizer
    return state


def find_bullets(engine, bots):
    bullets = {bullet.origin_id: bullet for bullet in engine.iter_bullets()}
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0)) for bot in bots
    ]


def make_states(engine):
    bot1, bot2 = engine.ai1.bot, engine.ai2.bot
    bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
    state1 = state2vec((bot1, bot2, bullet1, bullet2))
    state2 = state2vec((bot2, bot1, bullet2, bullet1))
    return state1, state2


def shape_to_list(shape):
    if hasattr(shape, "as_list"):
        return shape.as_list()
    else:
        return list(shape)


def add_batch_shape(x, batch_shape):
    if hasattr(batch_shape, "as_list"):
        batch_shape = batch_shape.as_list()
    tail_shape = x.get_shape().as_list()
    newshape = [1] * len(batch_shape) + tail_shape
    x = tf.reshape(x, newshape)
    newshape = batch_shape + tail_shape
    return x * tf.ones(newshape)


def select_features(tensor, mapper, *feature_names):
    feature_tensors = []
    for ftr_name in feature_names:
        idx = mapper[ftr_name]
        feature_tensors.append(tensor[..., idx : idx + 1])
    return tf.concat(feature_tensors, -1)


def assert_finite(tensor, watches=(), summarize=50):
    assertion = tf.Assert(tf.reduce_all(tf.is_finite(tensor)), watches, summarize=summarize)
    with tf.control_dependencies([assertion]):
        return tf.identity(tensor)
