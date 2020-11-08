from strateobots.engine import BulletModel


class Average:
    def __init__(self):
        self.value = 0
        self.n = 0

    def reset(self):
        self.value = 0
        self.n = 0

    def add(self, x):
        self.value += x
        self.n += 1

    def get(self):
        return self.value / max(1, self.n)


def find_bullets(engine, bots):
    bullets = {bullet.origin_id: bullet for bullet in engine.iter_bullets()}
    return [
        bullets.get(bot.id, BulletModel(None, None, 0, bot.x, bot.y, 0)) for bot in bots
    ]


def make_scope(prefix, suffix):
    if prefix is None:
        return suffix
    if prefix.endswith("/"):
        return prefix + suffix
    else:
        return prefix + "/" + suffix


def map_struct(function, structure):
    if isinstance(structure, dict):
        return {key: map_struct(function, value) for key, value in structure.items()}
    if isinstance(structure, (list, tuple, set, frozenset)):
        typ = structure.__class__
        return typ(map_struct(function, item) for item in structure)
    return function(structure)
