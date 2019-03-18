"""
Implement the CDNA video prediction network.

    .. code-block:: bibtex

        Unsupervised learning for physical interaction through video
        prediction. CoRR, abs/1605.07157.

There's also an implementation in Tensorflow model zoo:
http://places.csail.mit.edu/deepscene/small-projects/TRN-pytorch-pose/model_zoo/models/video_prediction/prediction_model.py,
which, well, probably is the implementation of the aforementioned paper, as
paper was published while the author was working at Google Brain ...
"""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from sv2p.convlstm import ConvLSTMCell2d

__all__ = [
    'ConditionalUNetLSTM',
    'shifted_relu',
    'depthwise_conv2d',
    'CDNA',
]


IntOr2Tuple = typing.Union[int, typing.Tuple[int, int]]


class ConditionalUNetLSTM(nn.Module):
    input_shape = 64, 64

    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        self.scale1_conv1 = nn.Conv2d(in_channels, 32, 5, stride=2, padding=2)
        # there's slight difference with the original TensorFlow
        # implementation: the shape of learnable parameters in LayerNorm
        # of TensorFlow is (n_channels, 1, 1) whereas the shape in here
        # is (n_channels, height, width); the same applies to all
        # LayerNorm below
        self.layer_norm1 = nn.LayerNorm((32, 32, 32))
        self.state1 = self.conv_lstm(32)
        self.layer_norm2 = nn.LayerNorm((32, 32, 32))
        # skip connection 1 branches out
        self.state2 = self.conv_lstm(32)
        self.layer_norm3 = nn.LayerNorm((32, 32, 32))
        self.conv2 = self.downsample2x(32)
        self.state3 = self.conv_lstm(32, 64)
        self.layer_norm4 = nn.LayerNorm((64, 16, 16))
        # skip connection 2 branches out
        self.state4 = self.conv_lstm(64)
        self.layer_norm5 = nn.LayerNorm((64, 16, 16))
        self.conv3 = self.downsample2x(64)
        self.conv4 = self.reduce_dim(64 + cond_channels, 64)
        self.state5 = self.conv_lstm(64, 128)
        self.layer_norm6 = nn.LayerNorm((128, 8, 8))
        # branches out to CDNA transformation
        self.convt1 = self.upsample2x(128)
        self.state6 = self.conv_lstm(128, 64)
        self.layer_norm7 = nn.LayerNorm((64, 16, 16))
        # skip connection 2 merges in
        self.convt2 = self.upsample2x(64 + 64)
        self.state7 = self.conv_lstm(128, 32)
        self.layer_norm8 = nn.LayerNorm((32, 32, 32))
        # skip connection 1 merges in
        self.convt3 = self.upsample2x(32 + 32)
        self.layer_norm9 = nn.LayerNorm((64, 64, 64))
        # UNetLSTM general modules done;
        # the rest modules are DNA/CDNA/STP-specific

        gain = nn.init.calculate_gain('relu')
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight.data, gain=gain)
                nn.init.zeros_(m.bias.data)

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor = None,
                hidden_states: typing.List[torch.Tensor] = None) \
            -> typing.Tuple[torch.Tensor, torch.Tensor,
                            typing.List[torch.Tensor]]:
        """
        :param inputs: a batch of images at time t of shape
               ``(B, self.in_channels, 64, 64)``
        :param conditions: if ``self.cond_channels`` is positive, this is a
               batch of condition feature maps of shape
               ``(B, self.cond_channels, 8, 8)``
        :param hidden_states: previous hidden states, or ``None`` if
               ``inputs`` is a batch of images at time 0
        :return: ``(innermost_embedding, inputspace_feature_map,
                 new_hidden_states)``
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(inputs.size(0))
        new_hidden_states = [None for _ in hidden_states]

        enc0 = torch.relu(self.layer_norm1(self.scale1_conv1(inputs)))
        hidden1, new_hidden_states[0] = self.state1(enc0, hidden_states[0])
        hidden1 = self.layer_norm2(hidden1)
        hidden2, new_hidden_states[1] = self.state2(hidden1, hidden_states[1])
        hidden2 = self.layer_norm3(hidden2)
        enc1 = torch.relu(self.conv2(hidden2))
        hidden3, new_hidden_states[2] = self.state3(enc1, hidden_states[2])
        hidden3 = self.layer_norm4(hidden3)
        hidden4, new_hidden_states[3] = self.state4(hidden3, hidden_states[3])
        hidden4 = self.layer_norm5(hidden4)
        enc2 = torch.relu(self.conv3(hidden4))
        if self.cond_channels:
            enc2 = torch.cat((enc2, conditions), dim=1)
        enc3 = torch.relu(self.conv4(enc2))
        hidden5, new_hidden_states[4] = self.state5(enc3, hidden_states[4])
        hidden5 = self.layer_norm6(hidden5)
        # hidden5 is the the innermost embedding.
        # Don't ask me why it's not named as `encXXX` ... I'm just adhering
        # to the original TensorFlow implementation for consistency
        enc4 = torch.relu(self.convt1(hidden5))
        hidden6, new_hidden_states[5] = self.state6(enc4, hidden_states[5])
        hidden6 = self.layer_norm7(hidden6)
        hidden6 = torch.cat((hidden6, hidden3), dim=1)
        enc5 = torch.relu(self.convt2(hidden6))
        hidden7, new_hidden_states[6] = self.state7(enc5, hidden_states[6])
        hidden7 = self.layer_norm8(hidden7)
        hidden7 = torch.cat((hidden7, hidden1), dim=1)
        enc6 = torch.relu(self.layer_norm9(self.convt3(hidden7)))
        # enc6 is the output general to DNA/CDNA/STP

        return hidden5, enc6, new_hidden_states

    def init_hidden(self, batch_size: int) -> typing.List[torch.Tensor]:
        device = next(self.parameters()).device
        return [getattr(self, 'state{}'.format(i + 1))
                .init_hidden((batch_size, -1, x, x)).to(device)
                for i, x in enumerate((
                    32, 32, 16, 16, 8, 16, 32,
                ))]

    @staticmethod
    def conv_lstm(in_channels: int, out_channels: int = None):
        if not out_channels:
            out_channels = in_channels
        return ConvLSTMCell2d(in_channels, out_channels, 5, padding=2)

    @staticmethod
    def downsample2x(in_channels: int, out_channels: int = None):
        if not out_channels:
            out_channels = in_channels
        return nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    @staticmethod
    def upsample2x(in_channels: int, out_channels: int = None):
        if not out_channels:
            out_channels = in_channels
        return nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2,
                                  padding=1, output_padding=1)

    @staticmethod
    def reduce_dim(in_channels: int, out_channels: int):
        return nn.Conv2d(in_channels, out_channels, 1)


# noinspection PyTypeChecker
def shifted_relu(tensor: torch.Tensor, eps: float = 1e-12):
    """
    Compute a slightly overesitmation of ReLU by

        .. math:: \\max(0, x-\\epsilon) + \\epsilon

    :param tensor: the :math:`x`, of any shape
    :param eps: the :math:`\\epsilon`
    """
    return torch.relu(tensor - eps) + eps


def depthwise_conv2d(inputs: torch.Tensor, kernel: torch.Tensor,
                     stride: IntOr2Tuple = 1, padding: IntOr2Tuple = 0,
                     dilation: IntOr2Tuple = 1):
    """
    Apply the same set of filters to each channel of the inputs.
    For detailed help on ``stride``, ``padding`` and ``dilation`` see
    ``help(torch.nn.functional.conv2d)``.

    :param inputs: of shape (B, C, H, W)
    :param kernel: of shape (M, 1, Kh, Kw)
    :param stride:
    :param padding:
    :param dilation:
    :return: tensor of shape (B, M, C, H', W')
    """
    C = inputs.shape[1]
    M, _K = kernel.shape[0], kernel.shape[1:]
    kernel = kernel.unsqueeze(0).expand(C, -1, -1, -1, -1).reshape(C * M, *_K)
    outputs = F.conv2d(inputs, kernel,
                       stride=stride, padding=padding, dilation=dilation,
                       groups=C)
    outputs = outputs.reshape(-1, C, M, *outputs.shape[-2:]).transpose(1, 2)
    return outputs


class CDNA(nn.Module):
    """
    Different from the original implementation, I didn't implement the
    recurrent inner state prediction here.
    """

    def __init__(self, in_channels: int, cond_channels: int, n_masks: int,
                 with_generator: bool = False):
        """
        :param in_channels: image color channels
        :param cond_channels: number of channels of the condition; 0 to have
               no condition
        :param n_masks: number of masks excluding the background mask; must
               be nonnegative
        :param with_generator: whether or not to use generator to compute the
               unoccluded pixels; the ``True`` case hasn't been implemented
               yet
        """
        super().__init__()
        if in_channels <= 0 or cond_channels < 0 or n_masks < 0:
            raise ValueError('Invalid argument value in ({})'
                             .format(', '.join((
                                 '{}={}'.format(k, v)
                                 for k, v in (('in_channels', in_channels),
                                              ('cond_channels', cond_channels),
                                              ('n_masks', n_masks))))))

        self.u_lstm = ConditionalUNetLSTM(in_channels, cond_channels)
        if with_generator:
            #self.convt4 = nn.ConvTranspose2d(64, in_channels, 1)
            raise NotImplementedError()
        self.convt7 = nn.ConvTranspose2d(64, 1 + n_masks, 1)
        self.cdna_params = nn.Linear(128 * 8 * 8, 5 * 5 * n_masks)

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.n_masks = n_masks
        self.with_generator = with_generator

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, conditions: torch.Tensor = None,
                hidden_states: typing.List[torch.Tensor] = None):
        """
        :param inputs: a batch of images at time t of shape
               ``(B, self.in_channels, 64, 64)``
        :param conditions: if ``self.cond_channels`` is positive, this is a
               batch of condition feature maps of shape
               ``(B, self.cond_channels, 8, 8)``
        :param hidden_states: previous hidden states, or ``None`` if
               ``inputs`` is a batch of images at time 0
        :return: composited predictions of shape the same as ``inputs``,
                 the new hidden states, the CDNA kernels of shape
                 ``(B, n_masks, 5, 5)``, and the masks of shape
                 ``(B, n_masks+1, H, W)`` where ``masks[:, 0]`` is the
                 background mask
        """
        embeddings, rfeatures, new_hidden_states = \
            self.u_lstm(inputs, conditions=conditions,
                        hidden_states=hidden_states)
        # rfeatures -- (r)econstructed (feature) map(s)

        assert not self.with_generator
        transformed_inputs, cdna_kerns = self.transform_inputs(
            inputs, embeddings.reshape(embeddings.size(0), -1), self.n_masks)
        # transform_inputs shape: (B, M, C, H, W)
        components = torch.cat((inputs.unsqueeze(1),
                                transformed_inputs), dim=1)
        # components shape: (B, M+1, C, H, W)

        masks = F.softmax(self.convt7(rfeatures), dim=1).unsqueeze(2)
        # masks shape: (B, M+1, H, W)
        predictions = torch.sum(components * masks, dim=1)
        return predictions, new_hidden_states, cdna_kerns, masks

    def transform_inputs(self, inputs: torch.Tensor,
                         cdna_inputs: torch.Tensor, n_masks: int):
        """
        Transform a batch of frames using computed CDNA inputs.

        :param inputs: the input frames of shape
               ``(B, self.in_channels, 64, 64)``
        :param cdna_inputs: of shape ``(B, 128 * 8 * 8)``
        :param n_masks: number of object masks
        :return: the transformed frames of shape
                 ``(B, n_masks, self.in_channels, 64, 64)``, and the CDNA
                 kernels used to transform the inputs of shape
                 ``(B, n_masks, 5, 5)
        """
        batch_size = inputs.size(0)
        cdna_kerns = self.cdna_params(cdna_inputs) \
                         .reshape(-1, n_masks, 1, 5, 5)
        cdna_kerns = shifted_relu(cdna_kerns)  # so that `norm` > 0
        norm = cdna_kerns.sum(dim=3, keepdim=True) \
                         .sum(dim=4, keepdim=True)
        cdna_kerns /= norm  # now weights in each filter sum up to 1
        # cdna_kerns shape: (B, n_masks, 1, 5, 5)
        transformed = torch.cat([
            depthwise_conv2d(inputs[i:i+1], cdna_kerns[i], padding=2)
            for i in range(batch_size)], dim=0)
        return transformed, cdna_kerns.squeeze(2)
