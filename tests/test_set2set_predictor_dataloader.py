import random
import numpy as np
from stb.ai.replay_generator import ReplayGenerator
from stb.ai.nn.predictor.dataloader import split_state2_into_subsets
from stb.ai.datacoding import WorldStateCodes


def test_split_into_subsets_is_complementary():
    replay = ReplayGenerator(2, 2, 2, max_ticks=200).generate()
    replay_item = random.choice(replay)
    state = WorldStateCodes.from_replay_item(replay_item)
    _, part1, part2 = split_state2_into_subsets((None, state))

    map1, dists1 = _match_similar_arrays(part1.bots, state.bots)
    map2, dists2 = _match_similar_arrays(part2.bots, state.bots)

    assert len(part1.bots) + len(part2.bots) == len(state.bots)
    assert set(map1) | set(map2) == set(range(len(state.bots)))
    assert max(*dists1, *dists2) < 1e-5


def test_split_into_subsets_keeps_correspondence_with_controls():
    replay = ReplayGenerator(2, 2, 2, max_ticks=200).generate()
    replay_item = random.choice(replay)
    state = WorldStateCodes.from_replay_item(replay_item, with_controls=True)
    _, part1, part2 = split_state2_into_subsets((None, state))

    map1, _ = _match_similar_arrays(part1.bots, state.bots)
    map2, _ = _match_similar_arrays(part2.bots, state.bots)

    assert np.all(part1.controls == state.controls[map1])
    assert np.all(part2.controls == state.controls[map2])


def _match_similar_arrays(array1, array2):
    array1 = array1[:, np.newaxis, :]
    array2 = array2[np.newaxis, :, :]
    distmap = np.square(array1 - array2).sum(2)
    map_indices = np.argmin(distmap, axis=1)
    return map_indices, distmap[np.arange(distmap.shape[0]), map_indices]
