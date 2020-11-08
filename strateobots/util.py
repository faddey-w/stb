import functools
import time
import sys
import contextlib
import signal
from collections import UserDict
import configparser
import importlib
from math import sin, cos, sqrt


def vec_rotate(x, y, angle):
    sin_a = sin(angle)
    cos_a = cos(angle)
    nx = x * cos_a - y * sin_a
    ny = x * sin_a + y * cos_a
    return nx, ny


def vec_sum(vec, *vecs):
    rx, ry = vec
    for x, y in vecs:
        rx += x
        ry += y
    return rx, ry


def vec_dot(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def half_chord_len(radius, distance):
    return sqrt(radius * radius - distance * distance)


def dist_line(point_x, point_y, line_cos, line_sin, line_x, line_y):
    q = line_sin * (point_x - line_x) - line_cos * (point_y - line_y)
    return abs(q)


def dist_points(x1, y1, x2, y2):
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def vec_len(x, y):
    return sqrt(x * x + y * y)


def vec_len2(x, y):
    return x * x + y * y


def quantize(x, q):
    r = x % q
    if r >= q / 2:
        r -= q
    return x - r


def _cached_with_timeout_impl(timeout, keyfunc):
    def decorator(function):
        cache = {}

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            key = keyfunc(args, kwargs)
            entry = cache.get(key)
            ok = False
            value = None
            exc_info = None
            now = time.time()
            if entry:
                value, exc_info, deadline = entry
                ok = now <= deadline
            if not ok:
                try:
                    value = function(*args, **kwargs)
                except:
                    exc_info = sys.exc_info()
                cache[key] = value, exc_info, now + timeout
            if exc_info:
                e, v, tb = exc_info
                raise v.with_traceback(tb)
            return value

        return wrapper

    return decorator


def cached_with_timeout(timeout):
    def keyfunc(args, kwargs):
        return args, frozenset(kwargs.items())

    return _cached_with_timeout_impl(timeout, keyfunc)


def cached_with_timeout_m(timeout):
    def keyfunc(args, kwargs):
        return args[1:], frozenset(kwargs.items())

    return _cached_with_timeout_impl(timeout, keyfunc)


def replay_descriptor_from_simulation(simulation):
    return {
        "id": simulation.sim_id,
        "finished": False,
        "nticks": simulation.engine.nticks,
        **simulation.metadata,
    }


def replay_descriptor_from_storage(storage, key):
    return {"id": key, "finished": True, **storage.load_metadata(key)}


class objedict(UserDict):
    def __getitem__(self, item):
        return _objedict_wrap_nested(super(objedict, self).__getitem__(item))

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        try:
            self.__dict__["data"][key] = value
        except KeyError:
            super(objedict, self).__setattr__(key, value)

    @property
    def __class__(self):
        return dict


DictWithAttrAccess = objedict


def _objedict_wrap_nested(value):
    typ = type(value)
    if typ is dict:
        value = objedict(value)
    elif typ in (list, tuple, set, frozenset):
        value = typ(map(_objedict_wrap_flat, value))
    return value


def _objedict_wrap_flat(value):
    if type(value) is dict:
        value = objedict(value)
    return value


@contextlib.contextmanager
def interrupt_atomic():
    def handler(*args, **kwargs):
        print("Will interrupt after atomic operation")
        nonlocal interrupted
        interrupted = True

    interrupted = False

    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    yield
    signal.signal(signal.SIGINT, prev_handler)
    if interrupted:
        raise KeyboardInterrupt


def make_metadata_before_game(init_name, ai1_module, ai1_name, ai2_module, ai2_name):
    return dict(
        init_name=init_name,
        ai1_module=ai1_module,
        ai1_name=ai1_name,
        ai2_module=ai2_module,
        ai2_name=ai2_name,
    )


def fill_metadata_after_game(metadata, engine):
    metadata["team1"] = str(engine.team1)
    metadata["team2"] = str(engine.team2)
    metadata["nticks"] = engine.nticks
    if engine.win_condition_reached:
        metadata["winner"] = str(engine.get_any_nonloser_team())
    else:
        metadata["winner"] = None
    for key, val in engine.get_metadata().items():
        metadata.setdefault(key, val)
    return metadata


def get_object_by_config(config_path, section):
    cp = configparser.ConfigParser()
    cp.read([config_path])
    module_path = cp.get(section, "module")
    classname = cp.get(section, "class")
    params = {}
    for key, val in cp.items(section):
        if key.startswith("param_"):
            params[key[6:]] = val

    cls = get_by_import_path(module_path, classname)
    inst = cls(**params)
    return inst


def get_by_import_path(module, name=None):
    if name is None:
        module, name = module.split(":")
    module = importlib.import_module(module)
    return getattr(module, name)
