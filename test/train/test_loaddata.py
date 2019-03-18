import os
import random
import typing

import numpy as np

import train.loaddata as ld


def _test_sample_dataset(dataset, assertion: typing.Callable):
    N = len(dataset)
    assert N
    indices = list(range(N))
    random.shuffle(indices)
    for i in indices[:50]:
        v = dataset[i]
        assertion(v)


def test_sample_movingmnist():

    def _assertion(v):
        assert v.shape == (20, 64, 64)
        assert v.dtype == np.uint8

    dataset = ld.MovingMNIST()
    _test_sample_dataset(dataset, _assertion)


def test_sample_kth(tmpdir):

    def _assertion(v):
        assert len(v.shape) == 4
        assert v.shape[-1] == 3

    dataset = ld.KTH(wd=tmpdir)
    _test_sample_dataset(dataset, _assertion)
    kthdir = os.listdir(os.path.join(os.environ['PYTORCH_DATA_HOME'], 'KTH'))
    assert not [x for x in kthdir if x.endswith('.npy')]
    dataset.teardown()
    assert not [x for x in kthdir if x.endswith('.npy')]
