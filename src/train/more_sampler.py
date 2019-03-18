import itertools
import operator
from functools import partial
import typing

import numpy as np
import torch

__all__ = [
    'SlidingWindowBatchSampler',
]


T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')


def broadcast_value2list(value: T1, iterable: typing.Iterable[T2]) \
        -> typing.Iterator[typing.Tuple[T1, T2]]:
    """
    >>> list(broadcast_value2list(4, [2, 3, 5]))
    [(4, 2), (4, 3), (4, 5)]
    """
    return map(lambda x: (value, x), iterable)


# pylint: disable=W0231
class ListSampler(torch.utils.data.Sampler):
    """
    Since list is not considered a Sampler although by duck typing it is, here
    I wrap a list into a sampler.
    """
    def __init__(self, indices: typing.Sequence[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


class SlidingWindowBatchSampler(torch.utils.data.Sampler):
    """
    Samples in a sliding window manner.
    """

    def __init__(self, indices, window_width: int,
                 shuffle: bool = False, batch_size: int = 1,
                 drop_last: bool = False):
        """
        :param indices: array-like integer indices to sample; when presented as
               a list of arrays, no sample will span across more than one array
        :param window_width: the width of the window; if ``window_width`` is
               larger than the length of ``indices`` or the length of one of
               the sublists, then that list won't be sampled
        :param shuffle: whether to shuffle sampling, but the indices order
               within a window is never shuffled
        :param batch_size: how many batches to yield upon each sampling
        :param drop_last: True to drop the remaining batches if the number of
               remaining batches is less than ``batch_size``

        Note on ``batch_size``
        ----------------------

        When ``batch_size = 2``, assuming that the two batch of indices are
        ``[1, 2, 3, 4]`` and ``[4, 5, 6, 7]``, then the yielded hyper-batch
        will be ``[1, 2, 3, 4, 4, 5, 6, 7]``.
        """
        indices = [np.array(x, dtype=np.int64) for x in list(indices)]
        if indices and not indices[0].shape:
            indices = [np.array(indices)]
        self.indices = indices  # a list of int64-arrays, or an empty list
        self.window_width = window_width
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        return sum(map(self._calc_sliding_distance, map(len, self.indices)))

    def __iter__(self):
        seglens = map(len, self.indices)
        slidedists = map(self._calc_sliding_distance, seglens)
        startindices = map(range, slidedists)
        segid_startindices = enumerate(startindices)
        segid_startindices = map(lambda x: broadcast_value2list(*x),
                                 segid_startindices)
        segid_startindices = list(itertools.chain(*segid_startindices))
        perm = (np.random.permutation if self.shuffle else np.arange)(
            len(segid_startindices))
        _gi = partial(operator.getitem, segid_startindices)
        for i in range(0, len(segid_startindices), self.batch_size):
            ind_tosample = perm[i:i + self.batch_size]
            if not (len(ind_tosample) < self.batch_size and self.drop_last):
                segid_startind_tosample = map(_gi, ind_tosample)
                sampled_batches = map(self._sample_batch_once,
                                      segid_startind_tosample)
                yield list(np.concatenate(list(sampled_batches)))

    def _calc_sliding_distance(self, length):
        return length - self.window_width + 1

    def _sample_batch_once(self, segid_startind):
        segid, startind = segid_startind
        return self.indices[segid][startind:startind + self.window_width]
