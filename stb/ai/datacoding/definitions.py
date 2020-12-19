from stb import models, engine
from stb.ai.datacoding.core import ScalarValue, OneHotCoder, FieldCoder

bot_type_coder = FieldCoder(
    {"code": OneHotCoder([bt.code for bt in models.BotType.get_list()])},
    getter=getattr,
    constructor=lambda d: models.BotType.by_code(d["code"]),
)
bot_full_coder = FieldCoder.from_dataclass(
    models.BotModel,
    _exclude=["id"],
    team=OneHotCoder(engine.StbEngine.TEAMS),
    type=bot_type_coder,
    hp_ratio=ScalarValue(0, 1),
    shield_ratio=ScalarValue(0, 1),
    shot_ready=ScalarValue(0, 1, integer=True),
    has_shield=ScalarValue(0, 1, integer=True),
)
bot_visible_coder = FieldCoder.from_dataclass(
    models.BotModel,
    _exclude=["id", *models.BotModel.HIDDEN_FIELDS],
    team=OneHotCoder(engine.StbEngine.TEAMS),
    type=bot_type_coder,
    hp_ratio=ScalarValue(0, 1),
    shield_ratio=ScalarValue(0, 1),
    has_shield=ScalarValue(0, 1, integer=True),
)
bullet_coder = FieldCoder(
    {
        field: ScalarValue() if field != "type" else bot_type_coder
        for field in models.BulletModel.FIELDS
        if field != "origin_id"
    },
    getter=getattr,
    constructor=lambda d: models.BulletModel(origin_id=None, **d),
    read_only=["sin", "cos"],
)
ray_coder = bullet_coder
control_coder = FieldCoder.from_dataclass(
    models.BotControl,
    move=ScalarValue(min_value=-1, max_value=+1, integer=True),
    rotate=ScalarValue(min_value=-1, max_value=+1, integer=True),
    tower_rotate=ScalarValue(min_value=-1, max_value=+1, integer=True),
)
