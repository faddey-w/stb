import tensorflow as tf
from strateobots.engine import BotControl
from strateobots.ai.lib import data


class ModelAiFunction:

    def __init__(self, model, session):
        state_ph = tf.placeholder(tf.float32, [1, data.state2vec.vector_length])
        inference = model.apply(state_ph)

        def function(engine, bot, enemy, ctl):
            bullet_b, bullet_e = find_bullets(engine, [bot, enemy])
            state = data.state2vec((bot, enemy, bullet_b, bullet_e))
            feeds = {state_ph: [state]}
            if exploration_feed is not None:
                feeds.update(exploration_feed)
            prediction = session.run(inference.action_prediction, feeds)[0]
            decode_prediction(prediction, ctl, **alt_dict)

    def __call__(self, engine, bot, enemy, ctl):
        pass


def find_bullets(engine, bots):
    bullets = {
        bullet.origin_id: bullet
        for bullet in engine.iter_bullets()
    }
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0))
        for bot in bots
    ]


def make_states(engine):
    bot1, bot2 = engine.ai1.bot, engine.ai2.bot
    bullet1, bullet2 = find_bullets(engine, [bot1, bot2])
    state1 = data.state2vec((bot1, bot2, bullet1, bullet2))
    state2 = data.state2vec((bot2, bot1, bullet2, bullet1))
    return state1, state2
