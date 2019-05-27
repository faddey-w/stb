import logging.config
import argparse
import itertools
import os
import strateobots
from tornado import web, ioloop
from strateobots.engine import BotType
from strateobots.ai.lib.bot_initializers import (
    random_bot_initializer,
    duel_bot_initializer,
)
from strateobots.visualizer_app import config, handlers
from strateobots.replay import CachedReplayDataStorage
from strateobots.visualizer_app.controller import ServerState
from strateobots.ai import base, physics_demo, simple_duel, guided_by


log = logging.getLogger(__name__)


def main(argv=None):
    static_dir = os.path.join(strateobots.REPO_ROOT, "visualizer_frontend")

    parser = argparse.ArgumentParser()
    parser.add_argument("--static-dir", default=static_dir)
    parser.add_argument("--port", "-p", default=9999, type=int)
    parser.add_argument("--storage-dir", "-S", required=True)
    parser.add_argument("--saved-models-dir", "-M")
    parser.add_argument("--user-programs-dir", "-P")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args(argv)
    logging.config.dictConfig(config.DEBUG_LOGGING if args.debug else config.LOGGING)

    typemap = {"R": BotType.Raider, "T": BotType.Heavy, "L": BotType.Sniper}

    duel_matchups = [
        (
            "Duel {}v{}".format(t1, t2),
            duel_bot_initializer(typemap[t1], typemap[t2], 0.7),
        )
        for t1, t2 in itertools.combinations_with_replacement("RTL", 2)
    ]
    random_matchups = []
    for matchup in config.RANDOM_MATCH_SETTINGS:
        ts1, ts2 = matchup.split("v")
        ts1 = [typemap[t] for t in ts1]
        ts2 = [typemap[t] for t in ts2]
        random_matchups.append(("Random " + matchup, random_bot_initializer(ts1, ts2)))

    default_module = base.DefaultAIModule([*duel_matchups, *random_matchups])
    simple_ais = simple_duel.AIModule()
    ai_modules = [default_module, physics_demo.AIModule(), simple_ais]

    if args.saved_models_dir is not None:
        from strateobots.ai import models

        models_ais = models.AIModule(args.saved_models_dir)
        ai_modules.append(models_ais)
        ai_modules.append(guided_by.AIModule(simple_ais, models_ais))

    if args.user_programs_dir is not None:
        from strateobots.ai import user_program

        program_storage = user_program.ProgramStorage(args.user_programs_dir)
        ai_modules.append(user_program.AIModule(program_storage))
    else:
        program_storage = None

    storage = CachedReplayDataStorage(args.storage_dir)
    state = ServerState(ai_modules, storage)
    initargs = dict(
        serverstate=state,
        auth_handler=handlers.noop_auth_handler,
        program_storage=program_storage,
    )
    fileserver_args = dict(path=args.static_dir, default_filename="index.html")
    app = web.Application(
        [
            ("/api/v1/launch-params", handlers.GameLaunchParametersHandler, initargs),
            ("/api/v1/game", handlers.GameListHandler, initargs),
            ("/api/v1/game/([0-9a-zA-Z_-]+)", handlers.GameViewHandler, initargs),
            (
                "/api/v1/programs/([0-9a-zA-Z_-]+)",
                handlers.UserProgramsViewHandler,
                initargs,
            ),
            ("/(.*)", web.StaticFileHandler, fileserver_args),
        ],
        debug=args.debug,
    )
    app.listen(args.port)
    log.info("listening at 0.0.0.0:%s", args.port)
    ioloop.IOLoop.instance().start()
