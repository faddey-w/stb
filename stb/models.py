import enum
import dataclasses
from types import MappingProxyType
from math import pi, sin, cos


class Constants:
    world_width = 1000
    world_height = 1000
    epsilon = 0.000001
    bot_radius = 25
    shield_damage_absorption = 4 / 5
    ray_min_load_required = 0.33
    bullet_speed = 500
    # friction_factor = 0
    friction_factor = 175
    collision_factor = 0.0002
    rotation_smoothness = 5
    min_collision_speed = 15
    ticks_per_sec = 50
    load_with_action = 0.4
    minimum_shield_warmup = 0.75
    shield_half_leak_period = 0.5
    full_information = True


@dataclasses.dataclass(frozen=True)
class BotTypeProperties:
    code: int
    max_hp: int
    mass: int
    reload_period: float
    acc: float
    bonus_acc: float
    max_ahead_speed: float  # points / sec
    bonus_max_speed: float  # points / sec
    max_back_speed: float  # points / sec
    rot_speed: float  # radian / sec
    bonus_rot_speed: float  # radian / sec
    gun_rot_speed: float  # radian / sec
    shots_ray: bool
    shot_range: int
    shot_energy: float  # for rays it is discharge per sec
    fire_scatter: float  # sigma of bullet direction distribution
    damage: int  # per-shot or per-second if ray
    shield_warmup_period: float  # sec
    shield_energy: int
    shield_regen: int  # hp / sec
    bonus_shield_regen: int  # hp / sec


class BotType:

    Heavy = BotTypeProperties(
        code=1,
        max_hp=1000,
        mass=100,
        reload_period=5,
        acc=17,
        bonus_acc=10,
        max_ahead_speed=55,
        max_back_speed=50,
        bonus_max_speed=20,
        rot_speed=pi / 4,
        bonus_rot_speed=pi / 4,
        gun_rot_speed=1.1 * 2 * pi / 3,
        shots_ray=False,
        shot_range=250,
        shot_energy=0.99,
        fire_scatter=2 * pi / 180,
        damage=200,
        shield_warmup_period=0.5,
        shield_energy=1200,
        shield_regen=50,
        bonus_shield_regen=20,
    )
    Raider = BotTypeProperties(
        code=2,
        max_hp=400,
        mass=50,
        reload_period=0.75,
        acc=75,
        bonus_acc=25,
        max_ahead_speed=160,
        max_back_speed=30,
        bonus_max_speed=40,
        rot_speed=pi,
        bonus_rot_speed=pi / 2,
        gun_rot_speed=pi,
        shots_ray=False,
        shot_range=200,
        shot_energy=0.33,
        fire_scatter=4 * pi / 180,
        damage=16,
        shield_warmup_period=0.8,
        shield_energy=200,
        shield_regen=20,
        bonus_shield_regen=20,
    )
    Sniper = BotTypeProperties(
        code=3,
        max_hp=250,
        mass=80,
        reload_period=10,
        acc=15,
        bonus_acc=12,
        max_ahead_speed=80,
        max_back_speed=40,
        bonus_max_speed=10,
        rot_speed=pi / 8,
        bonus_rot_speed=pi / 12,
        gun_rot_speed=pi / 6,
        shots_ray=True,
        shot_range=400,
        shot_energy=1,
        fire_scatter=0,
        damage=500,
        shield_warmup_period=0.1,
        shield_energy=500,
        shield_regen=75,
        bonus_shield_regen=10,
    )

    __members__ = MappingProxyType({"Heavy": Heavy, "Raider": Raider, "Sniper": Sniper})

    @classmethod
    def by_code(cls, code):
        for name, bt in cls.__members__.items():
            if bt.code == code:
                return bt
        raise ValueError

    @classmethod
    def get_list(cls):
        return list(cls.__members__.values())


@dataclasses.dataclass
class BotModel:
    id: int
    team: int
    type: BotTypeProperties
    x: float
    y: float
    orientation: float
    tower_orientation: float = 0.0
    shield_warmup: float = 0.0
    hp: float = None
    shield: float = None
    load: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    rot_speed: float = 0.0
    tower_rot_speed: float = 0.0
    is_firing: bool = False

    HIDDEN_FIELDS = "load", "rot_speed", "tower_rot_speed", "shot_ready"
    ALL_FIELDS = ()  # defined after class
    VISIBLE_FIELDS = ()  # defined after class

    def __post_init__(self):
        if self.hp is None:
            self.hp = self.type.max_hp
        if self.shield is None:
            self.shield = self.type.shield_energy

    @property
    def hp_ratio(self):
        return self.hp / self.type.max_hp

    @property
    def shield_ratio(self):
        return self.shield / self.type.shield_energy

    @property
    def shot_ready(self):
        return self.load >= self.type.shot_energy

    @property
    def has_shield(self):
        return self.shield_warmup > Constants.minimum_shield_warmup and self.shield > 0

    def serialize(self, with_hidden=True):
        if with_hidden:
            dct = _to_dict(self, BotModel.ALL_FIELDS)
        else:
            dct = _to_dict(self, BotModel.VISIBLE_FIELDS)
        dct["type"] = dct["type"].code
        dct["hp"] = dct.pop("hp_ratio")
        dct["shield"] = dct.pop("shield_ratio")
        return dct

    def __hash__(self):
        return hash(self.id)


BotModel.ALL_FIELDS = tuple(
    [
        *BotModel.__annotations__.keys(),
        *(name for name, value in vars(BotModel).items() if isinstance(value, property)),
    ]
)
BotModel.VISIBLE_FIELDS = tuple(set(BotModel.ALL_FIELDS) - set(BotModel.HIDDEN_FIELDS))


class BulletModel:

    __slots__ = [
        "type",
        "origin_id",
        "_orientation",
        "x",
        "y",
        "range",
        "remaining_range",
        "cos",
        "sin",
    ]
    FIELDS = tuple({*__slots__, "orientation"} - {"_orientation"})

    def __init__(self, type, origin_id, orientation, x, y, range, remaining_range=None):
        self.origin_id = origin_id
        self.type: BotTypeProperties = type
        self._orientation = orientation
        self.x = x
        self.y = y
        self.range = range
        self.remaining_range = range if remaining_range is None else remaining_range
        self.cos = cos(orientation)
        self.sin = sin(orientation)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value
        self.cos = cos(value)
        self.sin = sin(value)

    def serialize(self):
        dct = _to_dict(self, self.FIELDS)
        dct["type"] = dct["type"].code
        return dct


@dataclasses.dataclass
class ExplosionModel:

    x: float
    y: float
    duration: float
    size: float
    t: float = 0

    FIELDS = ()  # defined after class

    @property
    def t_ratio(self):
        return self.t / self.duration

    @property
    def is_ended(self):
        return self.t >= self.duration

    def serialize(self):
        return _to_dict(self, ExplosionModel.FIELDS)


ExplosionModel.FIELDS = tuple(ExplosionModel.__annotations__.keys())


class Action(int, enum.Enum):
    IDLE = 0
    FIRE = 1
    SHIELD_WARMUP = 2
    SHIELD_REGEN = 3
    ACCELERATION = 4


Action.ALL = tuple(Action.__members__.values())
Action.NAMES = tuple(name.lower() for name in Action.__members__.keys())


@dataclasses.dataclass
class BotControl:
    move: int = 0
    rotate: int = 0
    tower_rotate: int = 0
    action: Action = Action.IDLE

    FIELDS = ()  # defined after class


BotControl.FIELDS = tuple(BotControl.__annotations__.keys())


def _to_dict(obj, fields):
    return {field: getattr(obj, field) for field in fields}
