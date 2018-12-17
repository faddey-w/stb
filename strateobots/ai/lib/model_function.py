import tensorflow as tf
import numpy as np
from strateobots.ai.lib import data


class ModelAiFunction:

    def __init__(self, model, session):
        self.state_ph = tf.placeholder(tf.float32, [1, model.state_dimension])
        self.inference = model.apply(self.state_ph)
        self.model = model
        self.session = session
        self.encoder = model.data_encoder()

    def __call__(self, state):
        bot_data = state['friendly_bots'][0]
        state_vector = encode_vector_for_model(self.encoder, state)

        ctl_vectors = self.session.run(
            self.inference.controls, feed_dict={
                self.state_ph: [state_vector],
            }
        )
        ctl_dict = {
            'id': bot_data['id'],
            'move': data.ctl_move.decode(ctl_vectors['move'][0]),
            'rotate': data.ctl_rotate.decode(ctl_vectors['rotate'][0]),
            'tower_rotate': data.ctl_tower_rotate.decode(ctl_vectors['tower_rotate'][0]),
            'shield': data.ctl_shield.decode(ctl_vectors['shield'][0]),
            'fire': data.ctl_fire.decode(ctl_vectors['fire'][0]),
        }
        return [ctl_dict]

    def on_new_game(self):
        self.encoder = self.model.data_encoder()


def encode_vector_for_model(encoder, state, team=None, opponent_team=None):
    if team is None:
        bot_data = state['friendly_bots'][0]
        enemy_data = state['enemy_bots'][0]
    else:
        if opponent_team is None:
            opponent_team = (set(state['bots'].keys()) - {team}).pop()
        bot_data = state['bots'][team][0]
        enemy_data = state['bots'][opponent_team][0]
    bot_bullet_data = None
    enemy_bullet_data = None

    for bullet in state['bullets']:
        if bullet['origin_id'] == bot_data['id']:
            bot_bullet_data = bullet
        elif bullet['origin_id'] == enemy_data['id']:
            enemy_bullet_data = bullet

    return encoder(
        bot_data,
        enemy_data,
        bot_bullet_data,
        enemy_bullet_data,
    )


class VectorGeneratorDataEncoderMixin:
    @data.generator_encoder
    def data_encoder(self):
        state = yield
        prev_state_vector = self._encode_prev_state(*state)
        while True:
            curr_state_vector = self._encode_state(*state)
            state_vector = np.concatenate(
                [prev_state_vector, curr_state_vector])
            prev_state_vector = self._encode_prev_state(*state)
            state = yield state_vector

    @staticmethod
    def _encode_prev_state(bot, enemy, bot_bullet, enemy_bullet):
        raise NotImplementedError

    @staticmethod
    def _encode_state(bot, enemy, bot_bullet, enemy_bullet):
        raise NotImplementedError
