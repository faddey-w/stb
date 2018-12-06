import argparse
import importlib
from strateobots.engine import StbEngine, BotType
from strateobots.ai.lib import bot_initializers
from strateobots.ai import simple_duel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('aipath')
    parser.add_argument('matchup')
    opts = parser.parse_args()

    typemap = {'R': BotType.Raider, 'T': BotType.Heavy, 'L': BotType.Sniper}
    t1, t2 = opts.matchup.upper().split('V')

    initializer = bot_initializers.duel_bot_initializer(typemap[t1], typemap[t2], 0.7)

    aipath = opts.aipath
    if '/' in aipath and aipath.endswith('.py'):
        aipath = aipath[:-3].replace('/', '.')
    aimod = importlib.import_module(aipath).AIModule()

    ai1 = aimod.construct_ai_function(
        StbEngine.TEAMS[0],
        aimod.list_ai_function_descriptions()[0][1]
    )
    ai2 = simple_duel.AIModule().construct_ai_function(
        StbEngine.TEAMS[1],
        'distance',
    )

    engine = StbEngine(
        ai1=ai1,
        ai2=ai2,
        initialize_bots=initializer,
        max_ticks=20,
        wait_after_win=1,
        stop_all_after_finish=True,
    )
    while not engine.is_finished:
        print('TICK #{}'.format(engine.nticks+1))
        engine.tick()


if __name__ == '__main__':
    main()
