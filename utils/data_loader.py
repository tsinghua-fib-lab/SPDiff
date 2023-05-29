# -*- coding: utf-8 -*-
"""
***

"""
import numpy as np
import sys

sys.path.append('..')
import data.data as DATA
import data.dataset as DATASET


def make_batch(train_ids, batch_size, seed, shuffle=True, drop_last=True):
    """
    return a list of batch ids for mask-based batch.
    Args:
        drop_last:
        shuffle:
        seed:
        train_ids: list of train ids
        batch_size: ~
    Output:
        batch ids, e.g., [[1,2,3], [4,5,6], ...]
    """

    num_nodes = len(train_ids)
    if shuffle:
        # rnd_state = np.random.RandomState(seed)
        permuted_idx = np.random.permutation(num_nodes)
        train_ids = train_ids[permuted_idx]

    batches = [train_ids[i * batch_size:(i + 1) * batch_size] for
               i in range(int(num_nodes / batch_size))]
    if not drop_last:
        batches.append(train_ids[(num_nodes - num_nodes % batch_size):])

    return batches


def data_loader(data, batch_size, seed, shuffle=True, drop_last=True):
    if type(data) == DATA.PointwisePedData:
        loaders = make_batch(np.array(range(len(data))), batch_size, seed)
        loaders = [data[idx] for idx in loaders]
    elif type(data) == list:
        loaders = []
        for d in data:
            steps = int(d.dataset_len / batch_size)
            loaders += [DATA.ChanneledTimeIndexedPedData.slice(d, list(range(i*batch_size, (i+1)*batch_size))) for i in range(steps)]
            if not drop_last:
                loaders += [DATA.ChanneledTimeIndexedPedData.slice(d, list(range(steps*batch_size, d.dataset_len)))]
    else:
        raise NotImplementedError

    return loaders
