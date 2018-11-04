import functools
import time
import sys
import contextlib
import signal


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
                raise exc_info
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
        'id': simulation.sim_id,
        'finished': False,
        'nticks': simulation.engine.nticks,
        **simulation.metadata
    }


def replay_descriptor_from_storage(storage, key):
    return {
        'id': key,
        'finished': True,
        **storage.load_metadata(key)
    }


class DictWithAttrAccess(dict):

    def __getattr__(self, item):
        return _dwaa_wrap_nested(self[item])

    def __setattr__(self, key, value):
        self[key] = value


def _dwaa_wrap_nested(value):
    typ = type(value)
    if typ is dict:
        value = DictWithAttrAccess(value)
    elif typ in (list, tuple, set, frozenset):
        value = typ(map(_dwaa_wrap_flat, value))
    return value


def _dwaa_wrap_flat(value):
    if type(value) is dict:
        value = DictWithAttrAccess(value)
    return value


@contextlib.contextmanager
def interrupt_atomic():
    def handler(*args, **kwargs):
        print('Will interrupt after atomic operation')
        nonlocal interrupted
        interrupted = True
    interrupted = False

    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    yield
    signal.signal(signal.SIGINT, prev_handler)
    if interrupted:
        raise KeyboardInterrupt


