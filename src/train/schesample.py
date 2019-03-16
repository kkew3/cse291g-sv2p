import enum
from functools import partial
import typing

import numpy as np
import torch


__all__ = [
    'ScheduleAlgorithm',
    'calc_tfprob',
    'batch_sample',
    'sample_targets',
]


def calc_tfprob_linear(n_batch: int, **kwargs) -> float:
    e = float(kwargs['e'])
    k = float(kwargs['k'])
    c = float(kwargs['c'])
    if not 0.0 <= e <= 1.0:
        raise ValueError('Expecting 0<=e<=1 but got {}'.format(e))
    return np.clip(k - c * n_batch, e, 1.0)


def calc_tfprob_exponential(n_batch: int, **kwargs) -> float:
    k = float(kwargs['k'])
    if not 0.0 <= k <= 1.0:
        raise ValueError('Expecting 0<=k<=1 but got {}'.format(k))
    return np.float_power(k, n_batch)


def calc_tfprob_inverse_sigmoid(n_batch: int, **kwargs) -> float:
    k = float(kwargs['k'])
    if k < 1:
        raise ValueError('Expecting k>=1 but got {}'.format(k))
    return k / (k + np.exp(n_batch / k))


class ScheduleAlgorithm(enum.Enum):
    # Reason why to use `partial` here:
    # https://stackoverflow.com/a/40339397/7881370.
    # Not using `partial` results in
    # `AttributeError: 'function' object has no attribute 'value'` later
    LINEAR = partial(calc_tfprob_linear)
    EXP = partial(calc_tfprob_exponential)
    ISIGMOID = partial(calc_tfprob_inverse_sigmoid)


def calc_tfprob(algorithm: ScheduleAlgorithm, n_batch: int,
                **kwargs) -> float:
    """
    Compute the teacher forcing probability given the batch id.

    :param algorithm: the annealing algirthm to use
    :param n_batch: current batch id (note that it's the batch id of the
           entire training course, rather than the "local" batch id within
           current epoch)

    """
    return algorithm.value(n_batch, **kwargs)


def batch_sample(batch_size: int, seqlen: int, p: typing.Union[int, float],
                 warm_start: int, soft: bool = True) -> torch.Tensor:
    """
    Decide for a batch of video sequences of length ``seqlen`` whether or not
    to use teacher forcing. Denote the returned matrix as ``S``. If
    ``S[b, t]`` is ``0``, then teacher forcing should be used at minibatch
    ``b`` and time ``t``; otherwise, inference should be used instead.

    :param batch_size: the batch size
    :param seqlen: length of the batch of videos
    :param p: if ``soft`` is ``True``, this is the probability of using
           teacher forcing; otherwise, this is the total number of frames to
           use teacher forcing in each video sequence
    :param warm_start: the first ``warm_start`` frames in each video sequence
           will always use teacher forcing
    :param soft: see parameter ``p``
    :return: a 2D long tensor of shape ``(batch_size, seqlen)``
    """
    if soft:
        # Caution: `p` is Pr(use_teacher_forcing) but `0` corresponds to
        # using teacher forcing.
        sample = (torch.rand(batch_size, seqlen) > p).long()
        sample[:, :warm_start].zero_()
    else:
        if warm_start >= seqlen or p >= seqlen:
            teacherf = np.zeros((batch_size, seqlen), dtype=np.int64)
        else:
            teacherf = np.zeros((batch_size, seqlen - warm_start),
                                dtype=np.int64)
            teacherf[:, :seqlen - p] = 1
            for i in range(teacherf.shape[0]):
                np.random.shuffle(teacherf[i])
            teacherf = np.concatenate((np.zeros((batch_size, warm_start),
                                                dtype=np.int64),
                                       teacherf), axis=1)
        sample = torch.tensor(teacherf).long()
    return sample


def sample_targets(predictions: torch.Tensor, targets: torch.Tensor,
                   decider: torch.Tensor) -> torch.Tensor:
    """
    :param predictions: a minibatch of predictions of shape ``(B, ...)`` of
           current frames from the previous time step, which will be detached
           before used as targets
    :param targets: a minibatch of current ground truth frames of the same
           shape as ``predictions``
    :param decider: long tensor of shape ``(B,)`` deciding whether to use
           teacher forcing, such that ``0`` means using teacher forcing
           whereas ``1`` means not using it
    :return: the scheduled sampling of targets, of the same shape as
             ``predictions``
    """
    predictions = predictions.detach()
    device = predictions.device
    aug = torch.stack((targets, predictions))
    decider = (decider.reshape(1, decider.size(0), 1, 1, 1)
               * torch.ones(1, decider.size(0), *targets.shape[1:]).long())
    decider = decider.to(device)
    sampled = torch.gather(aug, 0, decider)[0]
    return sampled
