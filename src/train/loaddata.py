import os
import itertools
import operator
from functools import partial
import typing

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as trans

T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')


def broadcast_value2list(value: T1, iterable: typing.Iterable[T2]) \
        -> typing.Iterator[typing.Tuple[T1, T2]]:
    """
    >>> list(broadcast_value2list(4, [2, 3, 5]))
    [(4, 2), (4, 3), (4, 5)]
    """
    return map(lambda x: (value, x), iterable)


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
        if indices and not len(indices[0].shape):
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


class DeNormalize:
    """
    The inverse transformation of ``tochvision.transforms.Normalize``. As in
    ``tochvision.transforms.Normalize``, this operation modifies input tensor
    in place. The input tensor should be of shape (C, H, W), namely,
    (num_channels, height, width).

    :param mean: the mean used in ``Normalize``
    :param std: the std used in ``Normalize``
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float).reshape(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float).reshape(-1, 1, 1)

    def __call__(self, tensor):
        tensor.mul_(self.std)
        tensor.add_(self.mean)
        return tensor


def rearrange_temporal_batch(data_batch: torch.Tensor, T: int) -> torch.Tensor:
    """
    Rearrange a hyper-batch of frames of shape (B*T, C, H, W) into
    (B, C, T, H, W) where:

        - B: the batch size
        - C: the number of channels
        - T: the temporal batch size
        - H: the height
        - W: the width

    :param data_batch: batch tensor to convert
    :param T: the temporal batch size
    :return: converted batch
    """
    assert len(data_batch.size()) == 4
    assert data_batch.size(0) % T == 0
    B = data_batch.size(0) // T
    data_batch = data_batch.reshape(B, T, *data_batch.shape[1:])
    data_batch = data_batch.transpose(1, 2).contiguous()
    return data_batch.detach()  # so that ``is_leaf`` is True


class MovingMNIST(torch.utils.data.Dataset):
    """
    Returns single image in MovingMNIST dataset. The index
    """

    seqlen = 20

    # obtained by ./compute-nmlstats.py
    normalize = trans.Normalize(mean=(0.049270592390472503,),
                                std=(0.2002874575763297,))
    denormalize = DeNormalize(mean=(0.049270592390472503,),
                              std=(0.2002874575763297,))

    def __init__(self, transform: typing.Callable = None):
        datafile = os.path.join(os.path.normpath(
            os.environ['PYTORCH_DATA_HOME']), 'MovingMNIST',
            'mnist_test_seq.npy')
        data = np.load(datafile)  # shape: (T, N, H, W), dtype: uint8
        self.data = np.transpose(data, (1, 0, 2, 3)).reshape((-1, 64, 64))
        self.transform = transform

    def __getitem__(self, index: int):
        frame = self.data[index]
        if self.transform:
            frame = self.transform(frame)
        return frame

    @classmethod
    def get_batch_sampler(cls, video_indices: typing.Sequence[int],
                          window_width: int,
                          **kwargs) -> SlidingWindowBatchSampler:
        """
        :param video_indices: the videos to select
        :param window_width: the length of video to sample
        :param kwargs: the keyword arguments of ``SlidingWindowBatchSampler``
        :return: the batch sampler object
        """
        indices = [range(x * cls.seqlen, (1 + x) * cls.seqlen)
                   for x in video_indices]
        return SlidingWindowBatchSampler(indices, window_width, **kwargs)
