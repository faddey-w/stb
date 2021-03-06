import pytest
import copy
import random
import numpy as np
from itertools import chain
from stb.ai.datacoding import (
    bot_full_coder,
    bot_visible_coder,
    bullet_coder,
    ray_coder,
    control_coder,
    WorldStateCodes,
)
from stb.models import BotModel, BulletModel, BotType, BotControl, Action
from stb.engine import StbEngine
from stb.bot_initializers import RandomInitializer
from stb.ai.replay_generator import ReplayGenerator
from stb.util import dist_points


def _get_decoded(coder, original):
    code = coder.encode(original.serialize())
    assert isinstance(code, np.ndarray)
    assert code.ndim == 1
    assert code.dtype == "float64"
    return original.__class__.from_serialized(coder.decode(code))


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
    vector = control_coder.encode(BotControl().serialize())
    vector[control_coder.get_slice("move")[0]] = 0.1
    vector[control_coder.get_slice("rotate")[0]] = 0.55
    vector[control_coder.get_slice("tower_rotate")[0]] = -0.75
    ctl = BotControl.from_serialized(control_coder.decode(vector))
    assert ctl == BotControl(move=0, rotate=+1, tower_rotate=-1)


def test_world_state_codes_encodes_controls_in_the_same_order_as_bots():
    engine = StbEngine()
    RandomInitializer([BotType.Raider] * 10, [BotType.Heavy] * 10)(engine)
    for bot in engine.iter_bots():
        ctl = engine.get_control(bot)
        ctl.action = random.choice(Action.ALL)
        ctl.move = random.choice([-1, 0, +1])
        ctl.rotate = random.choice([-1, 0, +1])
        ctl.tower_rotate = random.choice([-1, 0, +1])
    wsc = WorldStateCodes.from_engine(engine, with_controls=True)

    decoded = wsc.to_replay_item()
    for team in decoded["bots"].keys():
        for bot_dec, ctl_dec in zip(decoded["bots"][team], decoded["controls"][team]):
            bot_orig = _find_most_similar_bot(engine, bot_dec)
            ctl_orig = engine.get_control(bot_orig)
            assert ctl_dec == ctl_orig.serialize()


@pytest.mark.parametrize('repeat_arg', range(5))
def test_world_state_codes_and_replay_item_conversion(repeat_arg):
    replay = ReplayGenerator(2, 2, 5, max_ticks=500).generate()
    replay_item = random.choice(replay[100:])
    state = WorldStateCodes.from_replay_item(replay_item, with_controls=True)
    restored = state.to_replay_item()

    expected = copy.deepcopy(replay_item)
    del expected["explosions"]
    for botlist in expected["bots"].values():
        for bot in botlist:
            # IDs are removed from state vectors:
            del bot["id"]
            # computed fields are not restored:
            del bot["shot_ready"]
            del bot["has_shield"]
    for bulletlike in chain(expected["bullets"], expected["rays"]):
        # IDs are removed from state vectors:
        del bulletlike["origin_id"]
        # computed fields are not restored:
        del bulletlike["sin"]
        del bulletlike["cos"]
    for ctllist in expected["controls"].values():
        for ctl in ctllist:
            # IDs are removed from state vectors:
            del ctl["id"]
            # drop all metadata/debugging outputs
            for field in list(ctl.keys()):
                if field not in BotControl.FIELDS:
                    del ctl[field]

    assert restored == expected


def _find_most_similar_bot(engine, bot_tgt):
    bot_orig = None
    best_dist = float("inf")
    for bot in engine.iter_bots():
        dist = dist_points(bot.x, bot.y, bot_tgt["x"], bot_tgt["y"])
        if dist < best_dist:
            bot_orig = bot
            best_dist = dist
    return bot_orig
