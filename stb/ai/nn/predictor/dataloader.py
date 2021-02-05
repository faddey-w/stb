import random
from torch.utils.data import DataLoader, IterableDataset
from stb.ai.datacoding import WorldStateCodes


class TransitionsStreamDataset(IterableDataset):
    def __init__(
        self,
        replay_generator_factory,
        max_transitions_per_game,
        shuffle_buffer_size,
        batch_size,
        is_train=False,
    ):
        self.replay_generator_factory = replay_generator_factory
        self.max_transitions_per_game = max_transitions_per_game
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.is_train = is_train

    def __iter__(self):
        replay_gen = self.replay_generator_factory()
        transition_lists_gen = (
            sample_transitions(replay, self.max_transitions_per_game) for replay in replay_gen
        )
        transitions_gen = flatten(transition_lists_gen)
        transitions_gen = iter_shuffled(transitions_gen, self.shuffle_buffer_size)
        if self.is_train:
            transitions_gen = map(split_state2_into_subsets, transitions_gen)
        yield from transitions_gen

    def make_dataloader(self,):
        return DataLoader(self, batch_size=self.batch_size, collate_fn=collate_transitions)


def sample_transitions(replay, max_transitions):
    total_transitions = len(replay) - 1
    n_to_sample = min(max_transitions, total_transitions)

    transitions = [
        (
            WorldStateCodes.from_replay_item(replay[i], with_controls=True),
            WorldStateCodes.from_replay_item(replay[i + 1], with_controls=False),
        )
        for i in random.sample(range(total_transitions), n_to_sample)
    ]

    return transitions


def flatten(item_batches):
    for batch in item_batches:
        for item in batch:
            yield item


def iter_shuffled(items, buffer_size):
    buffer = buffer_size
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
        else:
            index = random.randrange(buffer_size)
            yield buffer[index]
            buffer[index] = item
    random.shuffle(buffer)
    yield from buffer


def split_state2_into_subsets(transition):
    state1, state2 = transition

    state2_subset1 = WorldStateCodes()
    state2_subset2 = WorldStateCodes()

    def do_split(_, attr):
        if attr == "controls":
            return
        full_data = getattr(state2, attr)
        n = full_data.shape[0]
        all_indices = set(range(n))

        n1 = random.randint(0, n)
        indices1 = set(random.sample(all_indices, n1))
        indices2 = list(all_indices - indices1)
        indices1 = list(indices1)

        setattr(state2_subset1, attr, full_data[indices1])
        setattr(state2_subset2, attr, full_data[indices2])
        if attr == "bots" and state2.controls is not None and len(state2.controls) != 0:
            state2_subset1.controls = state2.controls[indices1]
            state2_subset2.controls = state2.controls[indices2]

    state2._map(do_split)

    return state1, state2_subset1, state2_subset2


def collate_transitions(batch):
    state1, state2 = zip(*batch)
    state1, state2 = WorldStateCodes.to_batch(state1), WorldStateCodes.to_batch(state2)
    return state1, state2
