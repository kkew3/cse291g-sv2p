"""
This module is adapted from https://github.com/Po-Hsun-Su/pytorch-ssim.git
"""

import torch
import torch.nn as nn
import numpy as np


def make_gaussian_filter(window_size, sigma):
    if window_size % 2 == 0:
        raise ValueError('Expecting odd window_size but got {}'
                         .format(window_size))
    w = window_size // 2
    x = np.arange(window_size)
    g1d = np.exp(-np.square(x-w)/(2*sigma**2))
    g2d = np.outer(g1d, g1d)
    g2d /= np.sum(g2d)
    return torch.from_numpy(g2d).float()


class Gaussian2d(nn.Module):
    def __init__(self, in_channels: int, window_size: int, sigma: float):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, window_size,
                              padding=window_size // 2, groups=in_channels,
                              bias=False)
        self.conv.weight.requires_grad_(False)
        g2d = make_gaussian_filter(window_size, sigma)
        g2d = g2d.reshape(*((1, 1) + g2d.size())).repeat(in_channels, 1, 1, 1)
        self.conv.weight.copy_(g2d)

    # pylint: disable=arguments-differ
    def forward(self, img):
        return self.conv(img)


class SSIM(nn.Module):
    """
    The differentiable Structural Similarity index module. Can be used as an
    alternative to MSELoss between two batches of images.

    The SSIM map is computed by:

        .. math::

            \\frac{(2\\mu_1\\mu_2+C_1)(2\\sigma_{12}+C_2)}
                  {(\\mu_1^2+\\mu_2^2+C_1)(\\sigma_{11}+\\sigma_{22}+C_2)}
    """

    def __init__(self, in_channels: int,
                 window_size: int = 11, sigma: float = 1.5,
                 reduction: str = 'mean'):
        """
        :param in_channels: number of input channels
        :param window_size: Gaussian window size
        :param sigma: Gaussian standard deviation
        :param reduction: 'none', 'mean' or 'sum'
        """
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError('Expected reduction in (\'none\', \'mean\', '
                             '\'none\') but got: {}'.format(reduction))
        self.gaussian = Gaussian2d(in_channels, window_size, sigma)
        self.reduction = reduction

    # pylint: disable=arguments-differ
    def forward(self, img1, img2):
        """
        :param img1: a batch of image tensors of shape (B, C, H, W)
        :param img2: a batch of image tensors of shape (B, C, H, W)
        """
        mu1 = self.gaussian(img1)
        mu2 = self.gaussian(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        s11 = self.gaussian(img1 * img1) - mu1_sq
        s22 = self.gaussian(img2 * img2) - mu2_sq
        s12 = self.gaussian(img1 * img2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = (((2 * mu1_mu2 + C1) * (2 * s12 + C2))
                    / ((mu1_sq + mu2_sq + C1) * (s11 + s22 + C2)))
        # SSIM map flattened
        ssim_mapf = torch.mean(ssim_map.reshape(ssim_map.size(0), -1), 1)
        if self.reduction == 'sum':
            ssim_mapf = torch.sum(ssim_mapf)
        elif self.reduction == 'mean':
            ssim_mapf = torch.mean(ssim_mapf)
        return ssim_mapf


class DSSIM(nn.Module):
    """
    The dissimilarity version of SSIM, sharing the same __init__
    parameters.
    """
    def __init__(self, *args, **kwargs):
        """
        For parameters see ``help(DSSIM)`` for detail.
        """
        super().__init__()
        self.ssim = SSIM(*args, **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, img1, img2):
        return (1 - self.ssim(img1, img2)) / 2
