import typing

import torch

__all__ = [
    'DeNormalize',
    'rearrange_temporal_batch',
    'VideoTransform',
]


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


def identitymap(x):
    return x


class VideoTransform:
    """
    Apply ``transform`` to every frame of a. The results are stacked
    together along dimension 0.
    """
    def __init__(self, transform: typing.Callable[[typing.Any], torch.Tensor]):
        self.transform = transform

    def __call__(self, video: typing.Iterable[typing.Any]):
        return torch.stack(list(map(self.transform or identitymap, video)))

    def __bool__(self):
        return bool(self.transform)
