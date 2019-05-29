import numpy as np
from collections import defaultdict
from strateobots.ai.lib import data
from strateobots.ai.simple_duel import norm_angle


def _coord_scale(x):
    return x * 0.001


def _velocity_scale(x):
    return x * 0.01


_bot_visible_fields = data.FeatureSet(
    [
        data.Feature(["x"], _coord_scale),
        data.Feature(["y"], _coord_scale),
        data.Feature(["hp"]),
        data.Feature(["orientation"], norm_angle),
        data.Feature(["orientation"], np.sin),
        data.Feature(["orientation"], np.cos),
        data.Feature(["tower_orientation"], norm_angle),
        data.Feature(["tower_orientation"], np.sin),
        data.Feature(["tower_orientation"], np.cos),
        data.Feature(["shield"]),
        data.Feature(["has_shield"]),
        data.Feature(["is_firing"]),
    ]
)
_bot_private_fields = data.FeatureSet(
    [
        data.Feature(["vx"], _velocity_scale),
        data.Feature(["vy"], _velocity_scale),
        data.Feature(["load"]),
        data.Feature(["shot_ready"]),
        data.Feature(["shield_warmup"]),
    ]
)
_bullet_fields = data.FeatureSet(
    [
        data.Feature(["present"]),
        data.Feature(["x"], _coord_scale),
        data.Feature(["y"], _coord_scale),
        data.Feature(["orientation"], norm_angle),
        data.Feature(["orientation"], np.sin),
        data.Feature(["orientation"], np.cos),
    ]
)
_extra_fields = data.FeatureSet(
    [
        data.Feature(["bot_gun_orientation"], norm_angle),
        data.Feature(["bot_gun_orientation"], np.sin),
        data.Feature(["bot_gun_orientation"], np.cos),
        data.Feature(["enemy_gun_orientation"], norm_angle),
        data.Feature(["enemy_gun_orientation"], np.sin),
        data.Feature(["enemy_gun_orientation"], np.cos),
        data.Feature(["angle_to_enemy"], norm_angle),
        data.Feature(["angle_to_enemy"], np.sin),
        data.Feature(["angle_to_enemy"], np.cos),
        data.Feature(["distance_to_enemy"], _coord_scale),
        data.Feature(["recipr_distance_to_enemy"]),
    ]
)


def encode_1vs1_fully_visible(state, team=None, opponent_team=None):
    bot, enemy, bot_bullet, enemy_bullet = _get_1vs1_data(state, team, opponent_team)

    # prepare engineered features
    dx = enemy["x"] - bot["x"]
    dy = enemy["y"] - bot["y"]
    distance_to_enemy = np.sqrt(dx * dx + dy * dy)
    angle_to_enemy = np.arctan2(dy, dx)
    bot_gun_orientation = bot["orientation"] + bot["tower_orientation"]
    enemy_gun_orientation = enemy["orientation"] + enemy["tower_orientation"]
    extra = {
        "bot_gun_orientation": bot_gun_orientation,
        "enemy_gun_orientation": enemy_gun_orientation,
        "angle_to_enemy": angle_to_enemy,
        "distance_to_enemy": distance_to_enemy,
        "recipr_distance_to_enemy": 1 / _coord_scale(distance_to_enemy),
    }

    # do encoding
    bot_vector = np.concatenate([_bot_visible_fields(bot), _bot_private_fields(bot)])
    enemy_vector = np.concatenate(
        [_bot_visible_fields(enemy), _bot_private_fields(enemy)]
    )
    bot_bullet_vector = _bullet_fields(bot_bullet)
    enemy_bullet_vector = _bullet_fields(enemy_bullet)
    extra_vector = _extra_fields(extra)

    return np.concatenate(
        [bot_vector, enemy_vector, bot_bullet_vector, enemy_bullet_vector, extra_vector]
    )


dimension_1vs1_fully_visible = (
    _bot_visible_fields.dimension * 2
    + _bot_private_fields.dimension * 2
    + _bullet_fields.dimension * 2
    + _extra_fields.dimension
)


_control_fields_full = data.FeatureSet(
    [getattr(data, "ctl_" + ctlname) for ctlname in data.ALL_CONTROLS_FULL]
)


def encode_controls_full(state, team, opponent_team=None):
    bot, _, _, _ = _get_1vs1_data(state, team, opponent_team)
    ctl = _get_controls(state, team, bot["id"])
    return _control_fields_full(ctl)


dimension_controls_full = _control_fields_full.dimension


def extract_control(controls_array, name):
    return controls_array[..., _control_fields_full.get_slice([name])]


def get_encoder(name):
    if name == "1vs1_fully_visible":
        return encode_1vs1_fully_visible, dimension_1vs1_fully_visible
    if name == "controls_full":
        return encode_controls_full, dimension_controls_full
    raise KeyError(name)


def _get_1vs1_data(state, team, opponent_team):
    # support both training and performance modes
    if team is None:
        bot_data = state["friendly_bots"][0]
        enemy_data = state["enemy_bots"][0]
    else:
        if opponent_team is None:
            opponent_team = (set(state["bots"].keys()) - {team}).pop()
        bot_data = state["bots"][team][0]
        enemy_data = state["bots"][opponent_team][0]
    bot_bullet_data = None
    enemy_bullet_data = None

    for bullet in state["bullets"]:
        if bullet["origin_id"] == bot_data["id"]:
            bot_bullet_data = bullet
        elif bullet["origin_id"] == enemy_data["id"]:
            enemy_bullet_data = bullet

    if bot_bullet_data is None:
        bot_bullet_data = defaultdict(float, present=False)
    else:
        bot_bullet_data = {"present": True, **bot_bullet_data}
    if enemy_bullet_data is None:
        enemy_bullet_data = defaultdict(float, present=False)
    else:
        enemy_bullet_data = {"present": True, **enemy_bullet_data}

    return bot_data, enemy_data, bot_bullet_data, enemy_bullet_data


def _get_controls(state, team, bot_id):
    controls_list = state["controls"][team]
    for ctl in controls_list:
        if bot_id == ctl["id"]:
            return ctl
