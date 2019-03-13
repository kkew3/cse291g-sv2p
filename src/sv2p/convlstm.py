"""
Implementation of

    Shi, X., Chen, Z., Wang, H., Yeung, D., Wong, W., & Woo, W. (2015).
    Convolutional LSTM Network: A Machine Learning Approach for Precipitation
    Nowcasting. NIPS.
"""

import typing
import torch
import torch.nn as nn

__all__ = [
    'ConvLSTMCell2d',
]

IntOr2Tuple = typing.Union[int, typing.Tuple[int, int]]


class ConvLSTMCell2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: IntOr2Tuple, stride: IntOr2Tuple = 1,
                 padding: IntOr2Tuple = 0, dilation: IntOr2Tuple = 1,
                 groups: int = 1, bias: bool = True,
                 gate_cell_state: bool = False, forget_bias: float = 1.0):
        """
        The undocumented parameters have the same semantics as in
        ``help(torch.nn.Conv2d)``.

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param gate_cell_state: if not set, do not involve cell states in
               convolution at gates; cell states are involved in the original
               work of (Shi et al., 2015)
        :param forget_bias: according to (Finn et al., 2016),
               "We add forget_bias (default: 1) to the biases of the forget
               gate in order to reduce the scale of forgetting in the
               beginning of the training."; in (Shi et al., 2015) there's no
               ``forget_bias``
        """
        super().__init__()
        self.xh_convs = nn.Conv2d(in_channels + out_channels,
                                  out_channels * 4,
                                  kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        """
        the convolutions of concatenated (input, hidden states), stacked
        along dim=1 at respectively:

            - input gate (i)
            - forget gate (f)
            - output gate (o)
            - modulated cell input (c)
        """

        # Xavier initialization, according to the code of ConvLSTM author:
        # http://www.wanghao.in/code/SPARNN-release.zip, in file
        # SPARNN-release/sparnn/layers/basic/conv_lstm_layer.py;
        # and according to CDNA author:
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/initializers.py
        # https://github.com/tensorflow/models/blob/master/research/video_prediction/prediction_model.py
        nn.init.xavier_uniform_(self.xh_convs.weight.data)
        nn.init.zeros_(self.xh_convs.bias.data)

        if gate_cell_state:
            self.cell_gate_weights = nn.Parameter(
                data=torch.zeros(1, 3 * out_channels, 1, 1))

        # in case of necessary
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.gate_cell_state = gate_cell_state
        self.forget_bias = forget_bias

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor,
                hidden_states: torch.Tensor) \
            -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: the inputs of shape (B, C, H, W)
        :param hidden_states: the hidden states (hidden & cell concatenated
               along dim=1)
        :return: the hidden output and the new hidden states, where the
                 "hidden output" is in fact
                 ``new_hidden_states[:, :self.out_channels]``
        """
        hidden, cell = (hidden_states[:, :self.out_channels],
                        hidden_states[:, self.out_channels:])
        inputs_hidden = torch.cat((inputs, hidden), dim=1)
        # (i,f,o,c) before activation function
        ifoc_before = self.xh_convs(inputs_hidden)
        ifo, c = (ifoc_before[:, :-self.out_channels],
                  ifoc_before[:, -self.out_channels:])
        if self.gate_cell_state:
            ifo = ifo + cell * self.cell_gate_weights
        i, f, o = (ifo[:, :self.out_channels],
                   ifo[:, self.out_channels:2 * self.out_channels],
                   ifo[:, 2 * self.out_channels:])
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self.forget_bias)
        o = torch.sigmoid(o)
        c = torch.tanh(c)
        new_cell = f * cell + i * c
        new_hidden = o * torch.tanh(new_cell)
        new_hidden_states = torch.cat((new_hidden, new_cell), dim=1)
        return new_hidden, new_hidden_states

    def init_hidden(self, input_size: typing.Sequence[int]) -> torch.Tensor:
        """
        Returns zero-initialized hidden states and cell states stacked along
        dim=1 as per the input shape.
        """
        assert len(input_size) == 4, str(input_size)
        hidden_size = list(input_size)
        hidden_size[1] = 2 * self.out_channels
        return torch.zeros(*hidden_size)
