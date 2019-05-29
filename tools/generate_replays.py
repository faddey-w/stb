#! /usr/bin/env PYTHONPATH=. python3
import argparse
from strateobots.engine import StbEngine, BotType
from strateobots import replay, util
from strateobots.ai.lib.bot_initializers import random_bot_initializer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-games", "-n", type=int, default=1)
    ap.add_argument("--ai1", required=True)
    ap.add_argument("--ai2", required=True)
    ap.add_argument("--config", "-C", default="config.ini")
    opts = ap.parse_args()

    bot_init = random_bot_initializer([BotType.Raider], [BotType.Raider])

    function1 = util.get_object_by_config(opts.config, "ai." + opts.ai1)
    function2 = util.get_object_by_config(opts.config, "ai." + opts.ai2)

    storage = replay.ReplayDataStorage(opts.out)

    for game_id in range(1, opts.n_games + 1):
        engine = StbEngine(
            ai1=function1,
            ai2=function2,
            initialize_bots=bot_init,
            max_ticks=2000,
            wait_after_win_ticks=0,
            stop_all_after_finish=True,
        )
        engine.play_all()
        metadata = engine.get_metadata()
        print(f"{game_id}: {metadata}")

        storage.save_replay(f"{game_id}", metadata, engine.replay)


if __name__ == "__main__":
    main()
