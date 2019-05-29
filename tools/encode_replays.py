#! /usr/bin/env PYTHONPATH=. python3
import argparse
import os
import numpy as np
import tqdm
from strateobots import replay
from strateobots.ai.lib import data_encoding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Directory of replays storage")
    ap.add_argument("output", help="Directory to store result .npy files")
    ap.add_argument("--encoder", "-E", default="1vs1_fully_visible")
    opts = ap.parse_args()

    encoder, dimension = data_encoding.get_encoder(opts.encoder)

    storage = replay.ReplayDataStorage(opts.input)
    os.makedirs(opts.output, exist_ok=True)

    for game_id in tqdm.tqdm(storage.list_keys()):

        replay_data = storage.load_replay_data(game_id)[:-1]
        metadata = storage.load_metadata(game_id)

        if metadata["winner"] is not None:
            # last tick is after someone died
            replay_data = replay_data[:-1]

        replay_buffer = np.empty((len(replay_data), dimension))
        for i, state in enumerate(replay_data):
            replay_buffer[i] = encoder(state, metadata["team1"], metadata["team2"])
        np.save(os.path.join(opts.output, f"{game_id}.npy"), replay_buffer)


if __name__ == "__main__":
    main()
