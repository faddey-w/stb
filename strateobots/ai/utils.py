import numpy as np


def collate_sequences_with_padding(arrays, pad_value=0, sequence_axis=0, insert_batch_dim=0):
    ndim = arrays[0].ndim
    padded = []
    pad_masks = []
    seq_lens = [a.shape[sequence_axis] for a in arrays]
    max_seq = max(seq_lens)

    for a, seq_len in zip(arrays, seq_lens):
        pad_mask = np.zeros([max_seq], dtype=int)
        if seq_len != max_seq:
            pad_spec = [(0, 0)] * ndim
            pad_spec[sequence_axis] = (0, max_seq - seq_len)
            a = np.pad(a, pad_spec, constant_values=pad_value)
            pad_mask[seq_len:] = 1
        padded.append(a)
        pad_masks.append(pad_mask)
    batch = np.stack(padded, axis=insert_batch_dim)
    return batch, pad_masks
