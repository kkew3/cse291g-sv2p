import os
import gzip
import tempfile
import shutil
import typing

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as trans

import train.more_trans as more_trans


__all__ = [
    'MovingMNIST',
    'KTH',
]


PYTORCH_DATA_HOME = os.path.normpath(os.environ['PYTORCH_DATA_HOME'])


class MovingMNIST(torch.utils.data.Dataset):
    """
    Returns single image in MovingMNIST dataset. The index
    """

    seqlen = 20

    # obtained by ./compute-nmlstats.py
    normalize = trans.Normalize(mean=(0.049270592390472503,),
                                std=(0.2002874575763297,))
    denormalize = more_trans.DeNormalize(mean=(0.049270592390472503,),
                                         std=(0.2002874575763297,))

    def __init__(self, transform: typing.Callable = None):
        datafile = os.path.join(PYTORCH_DATA_HOME,
                                'MovingMNIST',
                                'mnist_test_seq.npy')
        data = np.load(datafile)  # shape: (T, N, H, W), dtype: uint8
        self.videos = np.transpose(data, (1, 0, 2, 3))
        self.transform = more_trans.VideoTransform(transform)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int):
        v = self.videos[index]
        if self.transform:
            v = self.transform(v)
        return v


def extract_gz(filename, tofile):
    with gzip.open(filename) as infile:
        with open(tofile, 'wb') as outfile:
            shutil.copyfileobj(infile, outfile)


class KTH(torch.utils.data.Dataset):
    def __init__(self, transform=None, wd=os.getcwd()):
        """
        :param wd: the temporary working directory used as cache to expand
               the dataset; default to current working directory
        """
        self.transform = transform

        self.datadir = os.path.join(os.path.normpath(
            os.environ['PYTORCH_DATA_HOME']), 'KTH')
        with open(os.path.join(self.datadir, 'kth.lst')) as infile:
            self.videolist = list(map(str.strip, infile))
        self._tempdir = tempfile.TemporaryDirectory(dir=wd)
        self._tempdir_name = self._tempdir.name

    def __len__(self):
        return len(self.videolist)

    def __getitem__(self, index):
        filename = os.path.join(self.datadir, self.videolist[index])
        npyfile = os.path.join(self._tempdir_name, os.path.basename(filename))
        try:
            v = np.load(npyfile)
        except FileNotFoundError:
            extract_gz(filename + '.gz', npyfile)
            v = np.load(npyfile)
        if self.transform:
            v = self.transform(v)
        return v

    def __enter__(self):
        return self

    def __exit__(self, _a, _b, _c):
        self.teardown()

    def teardown(self):
        self._tempdir.cleanup()
        self._tempdir_name = None
