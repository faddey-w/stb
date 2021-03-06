import logging.config
import argparse
import itertools
import os
from tornado import web, ioloop
from stb.models import BotType
from stb.bot_initializers import DuelInitializer, RandomInitializer, RandomSidedInitializer
from stb.visualizer_app import config, handlers
from stb.replay import CachedReplayDataStorage
from stb.visualizer_app.controller import ServerState
from stb.ai import base, physics_demo, simple_duel, guided_by, simple_team
try:
    from stb.ai import evolution
except ImportError:
    evolution = None


log = logging.getLogger(__name__)


def main(argv=None):
    static_dir = os.path.join(os.path.dirname(__file__), "frontend")

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
            DuelInitializer(typemap[t1], typemap[t2], 0.7),
        )
        for t1, t2 in itertools.combinations_with_replacement("RTL", 2)
    ]
    random_matchups = []
    for matchup in config.RANDOM_MATCH_SETTINGS:
        ts1, ts2 = matchup.split("v")
        ts1 = [typemap[t] for t in ts1]
        ts2 = [typemap[t] for t in ts2]
        random_matchups.append(("Random " + matchup, RandomInitializer(ts1, ts2)))
    sided_matchups = []
    for matchup in config.SIDED_MATCH_SETTINGS:
        ts1, ts2 = matchup.split("v")
        ts1 = [typemap[t] for t in ts1]
        ts2 = [typemap[t] for t in ts2]
        sided_matchups.append(("Sided " + matchup, RandomSidedInitializer(ts1, ts2)))
    sided_matchups.append(
        (
            "Sided 30R+L v 30R+L",
            RandomSidedInitializer(
                [BotType.Raider] * 30 + [BotType.Sniper],
                [BotType.Raider] * 30 + [BotType.Sniper],
            ),
        )
    )

    default_module = base.DefaultAIModule([*duel_matchups, *random_matchups, *sided_matchups])
    simple_ais = simple_duel.AIModule()
    ai_modules = [
        default_module,
        physics_demo.AIModule(),
        simple_ais,
        simple_team.AIModule(),
    ]
    if evolution is not None:
        ai_modules.append(evolution.AIModule())

    if args.saved_models_dir is not None:
        from stb.ai import models

        models_ais = models.AIModule(args.saved_models_dir)
        ai_modules.append(models_ais)
        ai_modules.append(guided_by.AIModule(simple_ais, models_ais))

    if args.user_programs_dir is not None:
        from stb.ai import user_program

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
