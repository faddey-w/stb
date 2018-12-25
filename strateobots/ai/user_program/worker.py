import base64
import itertools
import os
from strateobots.engine import BotType


def main():

    user_code_encoded = input()
    user_code = base64.b64decode(user_code_encoded).decode('utf-8')

    persistent_state = {}

    user_api_file = os.path.join(os.path.dirname(__file__), 'user_api.py')
    with open(user_api_file) as f:
        api_code = f.read()

    namespace = {
        '__builtins__': code_builtins,
        'persistent': persistent_state,
        '__name__': '__main__',
    }
    exec(api_code, namespace)
    print('', flush=True)

    while True:
        game_state_encoded = input()
        game_state = base64.b64decode(game_state_encoded).decode('utf-8')
        game_state = eval(game_state)
        tick = game_state['tick']
        simpleobject_cls = namespace['_SimpleObject']  # type: type
        for botdata in itertools.chain(game_state['friendly_bots'], game_state['enemy_bots']):
            botdata['type_code'] = botdata['type']
            typeobj = BotType.by_code(botdata['type']).value
            botdata['type'] = simpleobject_cls(typeobj._asdict())
        game_state = repr(game_state)

        exec(input_data_code.format(game_state, persistent_state), namespace)
        exec(user_code, namespace)

        persistent_state = namespace.get('persistent')
        print('\0\0{}'.format(tick), flush=True)


input_data_code = """
data = {}
persistent = {}
bots = list(map(_SimpleObject, data['friendly_bots']))
enemies = list(map(_SimpleObject, data['enemy_bots']))
bullets = list(map(_SimpleObject, data['bullets']))
rays = list(map(_SimpleObject, data['rays']))
time = data['tick'] / 50.0
"""


code_builtins = dict(
    help=help,
    print=print,
    int=int,
    float=float,
    str=str,
    abs=abs,
    all=all,
    any=any,
    divmod=divmod,
    len=len,
    max=max,
    min=min,
    repr=repr,
    round=round,
    sorted=sorted,
    sum=sum,
    bool=bool,
    dict=dict,
    filter=filter,
    map=map,
    set=set,
    list=list,
    range=range,
    tuple=tuple,
    type=type,
    zip=zip,
    __build_class__=__build_class__,
)
code_builtins['__builtins__'] = {'__import__': __import__}
exec('from math import *', code_builtins)
del code_builtins['__builtins__']
main()
