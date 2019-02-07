import tensorflow as tf
import numpy as np
from math import pi, atan2
from strateobots.ai.lib import data
from strateobots.engine import BotType


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

        rotate = ctl_vectors.get('rotate', [None])[0]
        tower_rotate = ctl_vectors.get('tower_rotate', [None])[0]
        orientation = ctl_vectors.get('orientation', [None])[0]
        gun_orientation = ctl_vectors.get('gun_orientation', [None])[0]

        try:
            move_aim_x = ctl_vectors['move_aim_x'][0]
            move_aim_y = ctl_vectors['move_aim_y'][0]
        except KeyError:
            move_aim_x = move_aim_y = None
        else:
            orientation = atan2(move_aim_y - bot_data['y'],
                                move_aim_x - bot_data['x'])

        try:
            gun_aim_x = ctl_vectors['gun_aim_x'][0]
            gun_aim_y = ctl_vectors['gun_aim_y'][0]
        except KeyError:
            gun_aim_x = gun_aim_y = None
        else:
            gun_orientation = atan2(gun_aim_y - bot_data['y'],
                                    gun_aim_x - bot_data['x'])

        rotate, tower_rotate = _optimal_rotations(
            rotate, tower_rotate,
            orientation, gun_orientation,
            bot_data,
        )

        ctl_dict = {
            'id': bot_data['id'],
            'move': data.ctl_move.decode(ctl_vectors['move'][0]),
            'rotate': rotate,
            'tower_rotate': tower_rotate,
            'action': data.ctl_action.decode(ctl_vectors['action'][0]),
        }
        if move_aim_x is not None:
            ctl_dict['move_aim_x'] = move_aim_x
            ctl_dict['move_aim_y'] = move_aim_y
        elif orientation is not None:
            ctl_dict['orientation'] = orientation
        if gun_aim_x is not None:
            ctl_dict['gun_aim_x'] = gun_aim_x
            ctl_dict['gun_aim_y'] = gun_aim_y
        elif gun_orientation is not None:
            ctl_dict['gun_orientation'] = gun_orientation
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


class TwoStepDataEncoderMixin:
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


def _optimal_rotations(rotate, tower_rotate, orientation, gun_orientation, bot):
    if rotate is None:
        delta_angle = (orientation - bot['orientation']) % (2 * pi)
        rotate = -1 if delta_angle > pi else +1

    if tower_rotate is None:
        curr_gun_orientation = bot['orientation'] + bot['tower_orientation']
        # delta_angle = (gun_orientation - curr_gun_orientation) % (2 * pi)
        # tower_rotate = -1 if delta_angle > pi else +1
        bottype = BotType.by_code(bot['type'])

        right_gun_rot_speed = bottype.gun_rot_speed + rotate * bottype.rot_speed
        right_path = (gun_orientation - curr_gun_orientation) % (2 * pi)
        if right_gun_rot_speed < 0:
            right_path = 2 * pi - right_path
            right_gun_rot_speed = -right_gun_rot_speed
        right_time = right_path / max(right_gun_rot_speed, 0.0001)

        left_gun_rot_speed = -bottype.gun_rot_speed + rotate * bottype.rot_speed
        left_path = (curr_gun_orientation - gun_orientation) % (2 * pi)
        if left_gun_rot_speed < 0:
            left_path = 2 * pi - left_path
            left_gun_rot_speed = -left_gun_rot_speed
        left_time = left_path / max(left_gun_rot_speed, 0.0001)

        tower_rotate = +1 if right_time < left_time else -1

    return rotate, tower_rotate


if __name__ == '__main__':
    r = BotType.Raider.code
    assert (+1, +1) == _optimal_rotations(
        None, None, 1, 1,
        {'type': r, 'orientation': 0, 'tower_orientation': 0}
    )
    assert (+1, -1) == _optimal_rotations(
        None, None, 1, -0.1,
        {'type': r, 'orientation': 0, 'tower_orientation': 0}
    )
