import io
import math
import os
import typing
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans

import train.trainlib
import train.loaddata as ld
from sv2p.ssim import DSSIM
import sv2p.cdna as cdna


def chw2hwc(img: np.ndarray) -> np.ndarray:
    assert len(img.shape) == 3, str(img.shape)
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = img.transpose((1, 2, 0))
    return img


def make_grid(n: int, col: int = 5) -> typing.Tuple[int, int]:
    if n <= col:
        return 1, n
    return math.ceil(n / col), col


def schedule_teacher_forcing(B: int, T: int, n_gt: int, warm: int):
    """
    Returns a long tensor ``T`` such that ``T[b,t]==0`` means batch ``b`` at
    time step ``t`` requires teacher forcing.

    :param B: the batch size
    :param T: the temporal batch size
    :param n_gt: number of ground truth within each temporal batch
    :param warm: warm start size
    :return: long boolean tensor of shape (B, T)
    """
    if warm >= T or n_gt >= T:
        teacherf = np.zeros((B, T), dtype=np.int64)
    else:
        teacherf = np.zeros((B, T - warm), dtype=np.int64)
        teacherf[:, :T - n_gt] = 1
        for i in range(teacherf.shape[0]):
            np.random.shuffle(teacherf[i])
        teacherf = np.concatenate((np.zeros((B, warm), dtype=np.int64),
                                   teacherf), axis=1)
    return torch.tensor(teacherf).long()


def decide_targets(gt: torch.Tensor, prev: torch.Tensor,
                   decider: torch.Tensor) -> torch.Tensor:
    """
    :param gt: of shape (B, C, H, W)
    :param prev: detached previous predictions of shape (B, C, H, W)
    :param decider: long tensor of shape (B,)
    :return: the targets to be used in training
    """
    device = gt.device
    aug = torch.stack((gt, prev))
    decider = (decider.reshape(1, decider.size(0), 1, 1, 1)
               * torch.ones(1, decider.size(0), *gt.shape[1:]).long())
    decider = decider.to(device)
    actual = torch.gather(aug, 0, decider)
    return actual[0]


class CDNATrainer(train.trainlib.BasicTrainer):
    def __init__(self, in_channels: int, cond_channels: int, n_masks: int,
                 dataset_name: str,
                 indices: typing.Sequence[typing.Sequence[int]],
                 batch_size: int, lr: float, max_epoch: int,
                 seqlen: int, criterion_name: str,
                 scheduled_sampling: str = None, warm_start: int = 2,
                 device: str = 'cpu'):
        """
        :param in_channels:
        :param cond_channels:
        :param n_masks:
        :param dataset_name: the name of the dataset, one of:
               { "MovingMNIST" }
        :param indices: partition of video indices to train and valid, and
               optionally test; the first element will be regarded as the
               trainset indices, the second the validation set indices, the
               third the testset indices. If the third element is not given,
               it will be set to the same as the second element. If the third
               element is evaluated to ``False`` (e.g. empty list), the
               "test" stage will be skipped
        :param batch_size: the batch size
        :param lr: base learning rate of Adam optimizer
        :param max_epoch: max epoch to train
        :param seqlen: length of video sequence to sample
        :param criterion_name: one of: { "L1", "L2", "DSSIM" }
        :param scheduled_sampling: if ``None``, do not use scheduled sampling,
               i.e. teacher forcing all the time. If set to 'by_epoch',
               the number of ground truth will be in a temporal batch will
               be decreased by ``self.epoch``
        :param warm_start: effective only if ``scheduled_sampling`` is not
               ``None``; the first ``warm_start`` frames within a temporal
               batch will always be teacher forcing
        :param device: where to train
        """
        super().__init__(cdna.CDNA(in_channels, cond_channels, n_masks),
                         max_epoch, device)
        if seqlen < 2:
            raise ValueError('Expecting seqlen at least 2 but got {}'
                             .format(seqlen))
        if warm_start < 1:
            raise ValueError('Expecting warm_start at least 1 but got {}'
                             .format(warm_start))
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.lr = lr
        self.max_epoch = max_epoch
        self.criterion_name = criterion_name
        self.scheduled_sampling = scheduled_sampling
        if scheduled_sampling:
            self.warm_start = warm_start
        self.device = device

        if len(indices) < 2:
            raise ValueError('Missing indices: it should at least '
                             'include "train" and "valid"')
        if dataset_name == 'MovingMNIST':
            self.dataset = ld.MovingMNIST(transform=trans.Compose([
                trans.ToTensor(),
                ld.MovingMNIST.normalize,
            ]))
            self.denormalize = ld.MovingMNIST.denormalize
            self.sam_train = ld.MovingMNIST.get_batch_sampler(
                indices[0], self.seqlen, shuffle=True,
                batch_size=self.batch_size)
            self.sam_valid = ld.MovingMNIST.get_batch_sampler(
                indices[1], self.seqlen, batch_size=self.batch_size)
            if len(indices) == 2:
                self.sam_test = ld.MovingMNIST.get_batch_sampler(
                    indices[1], self.seqlen, batch_size=self.batch_size)
                self.run_stages = 'train', 'valid', 'test'
            elif indices[2]:
                self.sam_test = ld.MovingMNIST.get_batch_sampler(
                    indices[2], self.seqlen, batch_size=self.batch_size)
                self.run_stages = 'train', 'valid', 'test'
            else:
                self.run_stages = 'train', 'valid'
        else:
            raise ValueError('Dataset "{}" not supported'
                             .format(dataset_name))

        self.stat_names = ('loss',)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = {
            'L1': nn.L1Loss(),
            'L2': nn.MSELoss(),
            'DSSIM': DSSIM(self.net.in_channels),
        }[self.criterion_name].to(self.device)

    @staticmethod
    def fired(progress):
        _, ba = progress
        return ba % 10 == 0

    def __get_loader(self, dataloader: typing.Iterable[torch.Tensor]) \
            -> typing.Iterator:
        """
        Yields (inputs, targets), each of shape
        ``(self.batch_size, 1, self.seqlen - 1, 64, 64)``
        """
        _expected_size = (1, self.seqlen - 1, 64, 64)
        for frames in dataloader:
            frames = ld.rearrange_temporal_batch(frames, self.seqlen)
            inputs, targets = frames[:, :, :-1], frames[:, :, 1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            assert inputs.shape[1:] == _expected_size, str(inputs.size())
            assert targets.shape[1:] == _expected_size, str(targets.size())
            yield inputs, targets

    def get_trainloader(self):
        loader = None
        if self.dataset_name == 'MovingMNIST':
            loader = DataLoader(self.dataset, batch_sampler=self.sam_train,
                                num_workers=2, pin_memory=True)
        yield from self.__get_loader(loader)

    def get_validloader(self):
        loader = None
        if self.dataset_name == 'MovingMNIST':
            loader = DataLoader(self.dataset, batch_sampler=self.sam_valid,
                                num_workers=2, pin_memory=True)
        yield from self.__get_loader(loader)

    def get_testloader(self):
        loader = None
        if self.dataset_name == 'MovingMNIST':
            loader = DataLoader(self.dataset, batch_sampler=self.sam_test,
                                num_workers=2, pin_memory=True)
        yield from self.__get_loader(loader)

    def train_once(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        :param inputs: of shape (self.batch_size, 1, self.seqlen-1 64, 64)
        :param targets: of shape (self.batch_size, 1, self.seqlen-1, 64, 64)
        """
        inputs_t: torch.Tensor
        predictions_t: torch.Tensor
        targets_t: torch.Tensor

        B = inputs.size(0)
        T = inputs.size(2)

        # `decider` is a long boolean tensor of shape (B,) that implies which
        # targets to use: 0) teacher forcing, 1) detached previous prediction;
        # or `None` if to use ground truth (i.e. teacher forcing) always.
        if not self.scheduled_sampling:
            decider = None
        else:
            decider = {
                'by_epoch': schedule_teacher_forcing(B, T, T - self.epoch,
                                                     self.warm_start),
            }[self.scheduled_sampling]

        hidden = None
        loss = 0.0
        for t in range(inputs.size(2)):
            inputs_t = inputs[:, :, t]
            if (not t) or (decider is None):
                # testing `bool(decider)` is not reliable as
                # `bool(torch.tensor([0]))` is evaluated to `False`
                targets_t = targets[:, :, t]
            else:
                targets_t = decide_targets(targets[:, :, t],
                                           targets_t_prev, decider[:, t])
            predictions_t, hidden, _, _ = self.net(
                inputs_t, hidden_states=hidden)
            targets_t_prev = predictions_t.detach()
            loss_t = self.criterion(predictions_t, targets_t)

            loss += loss_t
        loss /= float(inputs.size(2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (loss.detach().cpu().item(),)

    def valid_once(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs_t: torch.Tensor
        predictions_t: torch.Tensor
        targets_t: torch.Tensor

        hidden = None
        loss = 0.0

        with torch.no_grad():
            for t in range(inputs.size(2)):
                inputs_t, targets_t = inputs[:, :, t], targets[:, :, t]
                predictions_t, hidden, _, _ = self.net(
                    inputs_t, hidden_states=hidden)
                loss_t = self.criterion(predictions_t, targets_t)

                loss += loss_t
            loss /= float(inputs.size(2))

        return (loss.detach().cpu().item(),)

    def test_once(self, inputs: torch.Tensor, targets: torch.Tensor):
        hidden = None
        loss = 0.0
        predictions = []
        kerns = []
        masks = []

        with torch.no_grad():
            for t in range(inputs.size(2)):
                inputs_t, targets_t = inputs[:, :, t], targets[:, :, t]
                predictions_t, hidden, kerns_t, masks_t = self.net(
                    inputs_t, hidden_states=hidden)
                loss_t = self.criterion(predictions_t, targets_t)

                loss += loss_t
                predictions.append(predictions_t.detach().cpu())
                kerns.append(kerns_t)
                masks.append(masks_t)
            loss /= float(inputs.size(2))

        # visualization
        # preprocessing: reshape to predictions and targets to (B, C, T, H, W)
        #                reshape kerns to (B, M, T, H, W)
        #                reshape masks to (B, M+1, T, H, W)
        predictions = self.__organize_tensor_to_imgs(
            torch.stack(predictions, dim=2))
        targets = self.__organize_tensor_to_imgs(targets.detach().cpu())
        kerns = self.__organize_tensor_to_imgs(torch.stack(kerns, dim=2))
        masks = self.__organize_tensor_to_imgs(torch.stack(masks, dim=2))

        statdir_test = getattr(self, 'statdir_test')

        try:
            os.mkdir(statdir_test)
        except FileExistsError:
            pass
        tofile = os.path.join(statdir_test, 'imgs.zip')
        with zipfile.ZipFile(tofile, 'a') as outfile:
            for b in range(predictions.shape[0]):
                fid = self.batch * self.batch_size + b
                pfx = 'ep{}/f{}_'.format(self.epoch, fid)
                for t in range(predictions.shape[2]):
                    sfx = '{}.png'.format(t)
                    with io.BytesIO() as buf:
                        plt.imsave(buf, chw2hwc(targets[b, :, t]),
                                   format='png')
                        plt.close()
                        buf.seek(0)
                        outfile.writestr('{}t{}'.format(pfx, sfx), buf.read())
                    with io.BytesIO() as buf:
                        plt.imsave(buf, chw2hwc(predictions[b, :, t]),
                                   format='png')
                        plt.close()
                        buf.seek(0)
                        outfile.writestr('{}p{}'.format(pfx, sfx), buf.read())
                    with io.BytesIO() as buf:
                        nr, nc = make_grid(kerns.shape[1])
                        for m in range(kerns.shape[1]):
                            plt.subplot(nr, nc, m + 1)
                            plt.imshow(kerns[b, m, t])
                        plt.savefig(buf, format='png')
                        plt.close()
                        buf.seek(0)
                        outfile.writestr('{}k_{}'.format(pfx, sfx), buf.read())
                    for m in range(masks.shape[1]):
                        with io.BytesIO() as buf:
                            plt.imsave(buf, masks[b, m, t], format='png')
                            plt.close()
                            buf.seek(0)
                            outfile.writestr('{}m{}_{}'.format(pfx, m, sfx),
                                             buf.read())
        return (loss.detach().cpu().item(),)

    def __organize_tensor_to_imgs(self, tensor: torch.Tensor) -> np.ndarray:
        assert len(tensor.shape) >= 2, str(tensor.shape)
        prefix_sh = tensor.shape[:-2]
        tensor = tensor.reshape(np.prod(prefix_sh), *tensor.shape[-2:]).clone()
        for i in range(np.prod(prefix_sh)):
            self.denormalize(tensor[i:i + 1])
        tensor = tensor.reshape(*(prefix_sh + tensor.shape[-2:])) \
                       .clamp(min=0.0, max=1.0) \
                       .numpy()
        return tensor


class CDNAInference(train.trainlib.BasicEvaluator):
    def __init__(self, in_channels: int, cond_channels: int, n_masks: int,
                 dataset_name: str, indices: typing.Sequence[int],
                 batch_size: int, seqlen: int, criterion_name: str,
                 basedir: str, progress: typing.Tuple[int, int],
                 device: str = 'cpu'):
        super().__init__(cdna.CDNA(in_channels, cond_channels, n_masks),
                         progress, basedir, device=device)
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.n_masks = n_masks
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.criterion_name = criterion_name

        self.run_stages = ('infer',)
        self.stat_names = 'loss', 'predictions', 'targets'

        if dataset_name == 'MovingMNIST':
            self.dataset = ld.MovingMNIST(transform=trans.Compose([
                trans.ToTensor(),
                ld.MovingMNIST.normalize,
            ]))
            self.denormalize = ld.MovingMNIST.denormalize
            self.sam_infer = ld.MovingMNIST.get_batch_sampler(
                indices, self.seqlen, batch_size=self.batch_size)
        else:
            raise ValueError('Dataset "{}" not supported'
                             .format(dataset_name))

    def __get_loader(self, dataloader: typing.Iterable[torch.Tensor]) \
            -> typing.Iterator:
        """
        Yields (inputs, targets), each of shape
        ``(self.batch_size, 1, self.seqlen - 1, 64, 64)``
        """
        _expected_size = (1, self.seqlen - 1, 64, 64)
        for frames in dataloader:
            frames = ld.rearrange_temporal_batch(frames, self.seqlen)
            inputs, targets = frames[:, :, :-1], frames[:, :, 1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            assert inputs.shape[1:] == _expected_size, str(inputs.size())
            assert targets.shape[1:] == _expected_size, str(targets.size())
            yield inputs, targets

    def get_inferloader(self):
        loader = None
        if self.dataset_name == 'MovingMNIST':
            loader = DataLoader(self.dataset, batch_sampler=self.sam_infer,
                                num_workers=2, pin_memory=True)
        yield from self.__get_loader(loader)

    def infer_once(self, inputs: torch.Tensor, targets: torch.Tensor):
        pass
