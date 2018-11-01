import logging.config
import argparse
import itertools
from tornado import web, ioloop
from strateobots.engine import BotType
from strateobots.ai.lib.bot_initializers import random_bot_initializer, duel_bot_initializer
from strateobots.visualizer_app import config, handlers
from strateobots.replay import CachedReplayDataStorage
from strateobots.visualizer_app.controller import ServerState
from strateobots.ai import base, physics_demo, simple_duel


log = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--static-dir', required=True)
    parser.add_argument('--port', '-P', default=9999, type=int)
    parser.add_argument('--storage-dir', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    logging.config.dictConfig(config.DEBUG_LOGGING if args.debug else config.LOGGING)

    typemap = {'R': BotType.Raider, 'T': BotType.Heavy, 'L': BotType.Sniper}

    duel_matchups = [
        ('Duel {}v{}'.format(t1, t2),
         duel_bot_initializer(typemap[t1], typemap[t2], 0.7))
        for t1, t2 in itertools.combinations('RTL', 2)
    ]
    random_matchups = []
    for matchup in config.RANDOM_MATCH_SETTINGS:
        ts1, ts2 = matchup.split('v')
        ts1 = [typemap[t] for t in ts1]
        ts2 = [typemap[t] for t in ts2]
        random_matchups.append(('Random ' + matchup, random_bot_initializer(ts1, ts2)))

    default_module = base.DefaultAIModule([
        *duel_matchups,
        *random_matchups,
    ])

    storage = CachedReplayDataStorage(args.storage_dir)
    state = ServerState([
        default_module,
        physics_demo.AIModule(),
        simple_duel.AIModule(),
    ], storage)
    initargs = dict(serverstate=state, auth_handler=handlers.noop_auth_handler)
    fileserver_args = dict(path=args.static_dir, default_filename='index.html')
    app = web.Application([
        ('/api/v1/launch-params', handlers.GameLaunchParametersHandler, initargs),
        ('/api/v1/game', handlers.GameListHandler, initargs),
        ('/api/v1/game/([0-9a-f_]+)', handlers.GameViewHandler, initargs),
        ('/(.*)', web.StaticFileHandler, fileserver_args)
    ], debug=args.debug)
    app.listen(args.port)
    log.info('listening at 0.0.0.0:%s', args.port)
    ioloop.IOLoop.instance().start()
