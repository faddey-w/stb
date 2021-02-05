from pathlib import Path
from stb.replay import ReplayDataStorage
from stb.ai.replay_generator import ReplayGenerator
from stb.ai.nn.predictor.dataloader import split_state2_into_subsets
from stb.ai.datacoding import WorldStateCodes


def main():
    replay, metadata = ReplayGenerator(2, 2, 2).generate(with_metadata=True)

    replay_split1 = []
    replay_split2 = []

    for item in replay:
        state = WorldStateCodes.from_replay_item(item, with_controls=True)
        _, part1, part2 = split_state2_into_subsets((None, state))
        # breakpoint()
        part1 = part1.to_replay_item()
        part2 = part2.to_replay_item()
        part1["explosions"] = []
        part2["explosions"] = []
        replay_split1.append(part1)
        replay_split2.append(part2)

    # print(sum("controls" in item for item in replay))
    # print(sum("controls" in item for item in replay_split1))
    # print(sum("controls" in item for item in replay_split2))
    # breakpoint()

    storage = ReplayDataStorage(Path(__file__).parent / "data")
    storage.save_replay("part1", metadata, replay_split1)
    storage.save_replay("part2", metadata, replay_split2)


if __name__ == "__main__":
    main()
