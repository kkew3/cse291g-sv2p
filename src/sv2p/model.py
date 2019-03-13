import typing

import torch
import torch.distributions
import torch.nn as nn


class Squeeze(nn.Module):
    """
    Merely for convenience to use in ``torch.nn.Sequential``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    # pylint: disable=arguments-differ
    def forward(self, tensor):
        return tensor.squeeze(self.dim)


class PosteriorInferenceNet(nn.Module):
    def __init__(self, tbatch: int):
        super().__init__()
        self.features = nn.Sequential(
            # the paper does not mention clearly how they condition on all
            # frames in a temporal batch; I'll just use conv3d here
            nn.Conv3d(3, 32, (tbatch, 3, 3),
                      stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            Squeeze(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, 3, stride=2, padding=1, bias=False),
        )

    # pylint: disable=arguments-differ
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        assert len(frames.shape) == 5, str(frames.shape)
        return self.features(frames)


class LatentVariableSampler:
    def __init__(self):
        self.using_prior: bool = False
        self.__prior = torch.distributions.Normal(0, 1)

    def sample(self, mu_sigma: torch.Tensor, n: int = 1) -> torch.Tensor:
        """
        If ``self.using_prior`` is True, sample from :math:`N(0, 1)`;
        otherwise, sample from :math:`N(\\mu, \\sigma^2)`.

        :param mu_sigma: the Gaussian parameter tensor of shape (B, 2, H, W)
        :param n: how many times to sample
        :return: of shape (B, n, H, W)
        """
        if self.using_prior:
            sample_shape = (mu_sigma.size(0), n,
                            mu_sigma.size(2), mu_sigma.size(3))
            z = self.__prior.sample(sample_shape)
        else:
            z = torch.distributions.Normal(
                mu_sigma[:, :1], mu_sigma[:, 1:]).sample(
                (n,))  # shape: (n, B, 1, H, W)
            z = z.squeeze(2).transpose(0, 1)
        return z
