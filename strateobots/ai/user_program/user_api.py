

class _SimpleObject:
    def __init__(self, selfdict):
        self.__dict__.update(selfdict)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.__dict__))


def _act(bot, key, value):
    print('\0\1{};{};{}'.format(bot.id, key, repr(value)), flush=True)


def move_ahead(bot):
    _act(bot, 'move', +1)


def move_back(bot):
    _act(bot, 'move', -1)


def stay(bot):
    _act(bot, 'move', 0)


# NOTE: rotation +1 does rotation clockwise in coordinate system,
# but on visualizer image is flipped (Y axis directed to bottom)
# so on visualizer it will look like counter-clockwise
# so "left" (which is intuitively counter-clockwise) must actually send +1
# and vice versa "right" should send -1.


def rotate_left(bot):
    _act(bot, 'rotate', +1)


def rotate_right(bot):
    _act(bot, 'rotate', -1)


def no_rotate(bot):
    _act(bot, 'rotate', 0)


def rotate_gun_left(bot):
    _act(bot, 'tower_rotate', +1)


def rotate_gun_right(bot):
    _act(bot, 'tower_rotate', -1)


def no_rotate_gun(bot):
    _act(bot, 'tower_rotate', 0)


class _EngineActionConstants:
    IDLE = 0
    FIRE = 1
    SHIELD_WARMUP = 2
    SHIELD_REGEN = 3
    ACCELERATION = 4


def idle(bot):
    _act(bot, 'action', _EngineActionConstants.IDLE)


def fire(bot):
    _act(bot, 'action', _EngineActionConstants.FIRE)


def shield_regen(bot):
    _act(bot, 'action', _EngineActionConstants.SHIELD_REGEN)


def shield_warmup(bot):
    _act(bot, 'action', _EngineActionConstants.SHIELD_WARMUP)


def accelerate(bot):
    _act(bot, 'action', _EngineActionConstants.ACCELERATION)


def new_object(**attrs):
    return _SimpleObject(attrs)


class _DummyImportable:

    def __getattr__(self, item):
        if item == '__all__':
            return []
        return getattr(__builtins__, item)


def __import__(*args, **kwargs):
    """Just to help user to write code with smart editors"""
    return _DummyImportable()
__builtins__['__import__'] = __import__
del __import__

data = {}
persistent = {}
bots = []
enemies = []
bullets = []
rays = []
time = 0.0
