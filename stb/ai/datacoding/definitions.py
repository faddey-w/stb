from stb import models, engine
from stb.ai.datacoding.core import ScalarValue, OneHotCoder, FieldCoder

bot_full_coder = FieldCoder.from_dataclass(
    models.BotModel,
    _exclude=["id"],
    team=OneHotCoder(engine.StbEngine.TEAMS),
    type=ScalarValue(integer=True),
    hp=ScalarValue(0, 1),
    shield=ScalarValue(0, 1),
    shot_ready=ScalarValue(0, 1, integer=True),
    has_shield=ScalarValue(0, 1, integer=True),
    _dict_mode=True,
)
bot_visible_coder = FieldCoder.from_dataclass(
    models.BotModel,
    _exclude=["id", *models.BotModel.HIDDEN_FIELDS],
    team=OneHotCoder(engine.StbEngine.TEAMS),
    type=ScalarValue(integer=True),
    hp=ScalarValue(0, 1),
    shield=ScalarValue(0, 1),
    has_shield=ScalarValue(0, 1, integer=True),
    _dict_mode=True,
)
bullet_coder = FieldCoder(
    {
        field: ScalarValue(integer=(field == "type"))
        for field in models.BulletModel.FIELDS
        if field != "origin_id"
    },
    read_only=["sin", "cos"],
)
ray_coder = bullet_coder
control_coder = FieldCoder.from_dataclass(
    models.BotControl,
    move=ScalarValue(min_value=-1, max_value=+1, integer=True),
    rotate=ScalarValue(min_value=-1, max_value=+1, integer=True),
    tower_rotate=ScalarValue(min_value=-1, max_value=+1, integer=True),
    _dict_mode=True,
)
