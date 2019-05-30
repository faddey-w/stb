import tensorflow as tf
import numpy as np
import os
from math import pi, atan2, sin, cos
from strateobots.ai.lib import data, model_saving, data_encoding
from strateobots.engine import BotType


class ModelAiFunction:
    def __init__(self, session, state_vector_ph, control_tensors, encoder):
        self.state_vector_ph = state_vector_ph
        self.control_tensors = control_tensors
        self.session = session
        self.encoder = encoder

    @classmethod
    def from_exported_model(cls, exported_model_dir):
        inference_pb_path = os.path.join(exported_model_dir, "inference.pb")
        _, _, encoder_name = model_saving.load_model_config(exported_model_dir)
        encoder, state_vec_dim = data_encoding.get_encoder(encoder_name)

        with tf.gfile.GFile(inference_pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name=""
            )

        def find_nodes(scope):
            prefix = scope if scope.endswith("/") else scope + "/"
            return {
                node.name[len(prefix):]: graph.get_tensor_by_name(node.name + ":0")
                for node in graph.as_graph_def().node
                if node.name.startswith(prefix)
            }

        state_vector_ph = find_nodes("Input")["state_vector"]
        ctl_tensors = find_nodes("Output")
        session = tf.Session(graph=graph)
        return cls(session, state_vector_ph, ctl_tensors, encoder)

    @classmethod
    def from_legacy_model_object(cls, model, session):
        raise NotImplementedError

    def __call__(self, state):
        state_vector = self.encoder(state)
        ctl_vectors = self.session.run(
            self.control_tensors, feed_dict={self.state_vector_ph: state_vector}
        )
        return predictions_to_ctls(ctl_vectors, state)

    def on_new_game(self):
        pass


def predictions_to_ctls(predictions, state):
    bot_data = state["friendly_bots"][0]
    ctl_vectors = predictions
    rotate = ctl_vectors.get("rotate", None)
    tower_rotate = ctl_vectors.get("tower_rotate", None)
    orientation = ctl_vectors.get("orientation", None)
    gun_orientation = ctl_vectors.get("gun_orientation", None)

    if True:
        # for debugging direct control models
        if orientation is None and rotate is not None:
            orientation = bot_data["orientation"] + pi / 3 * rotate
        if gun_orientation is None and tower_rotate is not None:
            gun_orientation = bot_data["orientation"] + bot_data["tower_orientation"] + pi / 3 * tower_rotate

    try:
        move_aim_x = ctl_vectors["move_aim_x"]
        move_aim_y = ctl_vectors["move_aim_y"]
    except KeyError:
        if orientation is None:
            move_aim_x = move_aim_y = None
        else:
            move_aim_x = bot_data["x"] + 100 * cos(orientation)
            move_aim_y = bot_data["y"] + 100 * sin(orientation)
    else:
        if orientation is None:
            orientation = atan2(
                move_aim_y - bot_data["y"], move_aim_x - bot_data["x"]
            )

    try:
        gun_aim_x = ctl_vectors["gun_aim_x"]
        gun_aim_y = ctl_vectors["gun_aim_y"]
    except KeyError:
        if gun_orientation is None:
            gun_aim_x = gun_aim_y = None
        else:
            gun_aim_x = bot_data["x"] + 100 * cos(gun_orientation)
            gun_aim_y = bot_data["y"] + 100 * sin(gun_orientation)
    else:
        if gun_orientation is None:
            gun_orientation = atan2(
                gun_aim_y - bot_data["y"], gun_aim_x - bot_data["x"]
            )

    rotate, tower_rotate = _nearest_rotation(
        rotate, tower_rotate, orientation, gun_orientation, bot_data
    )

    ctl_dict = {
        "id": bot_data["id"],
        "move": ctl_vectors["move"],
        "rotate": rotate,
        "tower_rotate": tower_rotate,
        "action": ctl_vectors["action"],
    }
    if move_aim_x is not None:
        ctl_dict["move_aim_x"] = move_aim_x
        ctl_dict["move_aim_y"] = move_aim_y
    if orientation is not None:
        ctl_dict["orientation"] = orientation
    if gun_aim_x is not None:
        ctl_dict["gun_aim_x"] = gun_aim_x
        ctl_dict["gun_aim_y"] = gun_aim_y
    if gun_orientation is not None:
        ctl_dict["gun_orientation"] = gun_orientation
    return [ctl_dict]


class TwoStepDataEncoderMixin:
    @data.generator_encoder
    def data_encoder(self):
        state = yield
        prev_state_vector = self._encode_prev_state(*state)
        while True:
            curr_state_vector = self._encode_state(*state)
            state_vector = np.concatenate([prev_state_vector, curr_state_vector])
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
        delta_angle = (orientation - bot["orientation"]) % (2 * pi)
        rotate = -1 if delta_angle > pi else +1

    if tower_rotate is None:
        curr_gun_orientation = bot["orientation"] + bot["tower_orientation"]
        # delta_angle = (gun_orientation - curr_gun_orientation) % (2 * pi)
        # tower_rotate = -1 if delta_angle > pi else +1
        bottype = BotType.by_code(bot["type"])

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


def _nearest_rotation(rotate, tower_rotate, orientation, gun_orientation, bot):
    if rotate is None:
        delta_angle = (orientation - bot["orientation"]) % (2 * pi)
        rotate = -1 if delta_angle > pi else +1

    if tower_rotate is None:
        curr_gun_orientation = bot["orientation"] + bot["tower_orientation"]
        right_path = (gun_orientation - curr_gun_orientation) % (2 * pi)

        tower_rotate = -1 if right_path > pi else +1

    return rotate, tower_rotate


if __name__ == "__main__":
    r = BotType.Raider.code
    assert (+1, +1) == _optimal_rotations(
        None, None, 1, 1, {"type": r, "orientation": 0, "tower_orientation": 0}
    )
    assert (+1, -1) == _optimal_rotations(
        None, None, 1, -0.1, {"type": r, "orientation": 0, "tower_orientation": 0}
    )
