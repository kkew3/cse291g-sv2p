import torch
import torch.nn as nn


__all__ = [
    'RotationInvarianceLoss',
]

def rot90(x: torch.Tensor):
    if len(x.shape) < 2:
        raise ValueError('Expecting at least 2D tensor but got shape {}'
                         .format(x.shape))
    return x.flip(-1).transpose(-2, -1)


class RotationInvarianceLoss(nn.Module):
    """
    Penalize rotation invariance of kernels.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    # pylint: disable=arguments-differ
    def forward(self, kerns: torch.Tensor):
        """
        :param kerns: of shape (..., KH, KW)
        """
        kh, kw = kerns.shape[-2:]
        kerns_rot = rot90(kerns)
        kerns = kerns.reshape(-1, kh, kw)
        kerns_rot = kerns_rot.reshape(-1, kh, kw)
        return self.criterion(kerns_rot, kerns)
