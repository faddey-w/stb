import pytest
import numpy as np
from strateobots.ai.coding import (
    bot_type_coder,
    bot_full_coder,
    bot_visible_coder,
    bullet_coder,
    ray_coder,
    control_coder,
)
from strateobots.models import BotModel, BulletModel, BotType, BotControl, Action
from strateobots.engine import StbEngine


def _get_decoded(coder, original):
    code = coder.encode(original)
    assert isinstance(code, np.ndarray)
    assert code.ndim == 1
    return coder.decode(code)


@pytest.mark.parametrize("bot_type", BotType.get_list())
def test_bot_type_coder(bot_type):
    assert _get_decoded(bot_type_coder, bot_type) is bot_type


def test_bot_full_coder():
    bot = BotModel(
        id=1,
        team=StbEngine.TEAMS[0],
        type=BotType.Heavy,
        x=100.0,
        y=200.0,
        orientation=0.5,
        tower_orientation=1.5,
        shield_warmup=0.73,
        hp=320.0,
        shield=850.0,
        load=0.12,
        vx=25.6,
        vy=-109.4,
        rot_speed=0.1,
        tower_rot_speed=-0.2,
        is_firing=False,
    )
    decoded = _get_decoded(bot_full_coder, bot)
    decoded.id = bot.id
    assert bot == decoded


def test_bot_visible_coder():
    bot = BotModel(
        id=1,
        team=StbEngine.TEAMS[0],
        type=BotType.Heavy,
        x=100.0,
        y=200.0,
        orientation=0.5,
        tower_orientation=1.5,
        shield_warmup=0.73,
        hp=320.0,
        shield=850.0,
        load=0.12,
        vx=25.6,
        vy=-109.4,
        rot_speed=0.1,
        tower_rot_speed=-0.2,
        is_firing=False,
    )
    decoded = _get_decoded(bot_visible_coder, bot)
    fields_to_check = set(BotModel.__annotations__.keys()) & set(BotModel.VISIBLE_FIELDS)
    orig_dict = {f: bot.__dict__[f] for f in fields_to_check}
    result_dict = {f: decoded.__dict__[f] for f in fields_to_check}
    result_dict["id"] = bot.id
    assert orig_dict == result_dict


def test_bullet_coder():
    bullet = BulletModel(
        type=BotType.Raider,
        origin_id=1,
        orientation=0.123,
        x=1000.0,
        y=9876.0,
        range=157.0,
        remaining_range=123.456,
    )
    decoded = _get_decoded(bullet_coder, bullet)
    decoded.origin_id = bullet.origin_id
    assert bullet.serialize() == decoded.serialize()


def test_ray_coder():
    assert ray_coder is bullet_coder


def test_control_coder():
    ctl = BotControl(move=+1, rotate=-1, tower_rotate=0, action=Action.SHIELD_REGEN)
    assert ctl == _get_decoded(control_coder, ctl)


def test_control_coder_decoding_intermediate_values():
    vector = control_coder.encode(BotControl())
    vector[control_coder.get_slice("move")[0]] = 0.1
    vector[control_coder.get_slice("rotate")[0]] = 0.55
    vector[control_coder.get_slice("tower_rotate")[0]] = -0.75
    ctl = control_coder.decode(vector)
    assert ctl == BotControl(move=0, rotate=+1, tower_rotate=-1)
