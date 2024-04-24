# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


def init_weights(modules_list: Union[nn.Module, List[nn.Module]]) -> None:
    if not isinstance(modules_list, List):
        modules_list = [modules_list]

    for m in modules_list:
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)


class Classifier(nn.Module):
    """Optional, pre-activation classification layer."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(input_dim, n_classes)

        init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x  # logits


class SimpleDenseNet(nn.Module):
    """
    Network with a single dense block.

    References
    ----------
    https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
    """

    def __init__(
        self,
        depth: int,
        output_dim: int,
        growth_rate: int = 32,
        initial_channels: int = 2,
        max_dilation: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()

        n_fixed_layers = 1  # embedding layer
        n_layers_per_block = depth - n_fixed_layers
        assert n_layers_per_block > 0, "`depth` is too small"

        input_channels = initial_channels

        # Single dense block
        layers = [
            DenseBlock(
                n_layers_per_block,
                input_channels,
                growth_rate,
                max_dilation=max_dilation,
                skip_bn_relu_first_layer=True,
                kernel_size=kernel_size,
            )
        ]
        input_channels += n_layers_per_block * growth_rate
        self.block_sequence = nn.Sequential(*layers)

        # Global average pooling and embedding layer
        self.bn2 = nn.BatchNorm1d(input_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_channels, output_dim)

        self.output_dim = output_dim

        # Initialize weights
        init_weights(self.modules())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_sequence(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DenseBlock(nn.Module):
    """Series of convolution blocks with dense connections."""

    def __init__(
        self,
        n_layers: int,
        input_channels: int,
        growth_rate: int,
        max_dilation: int = 64,
        skip_bn_relu_first_layer: bool = True,
        kernel_size: int = 3,
    ):
        super().__init__()

        dilation_exp_mod = int(np.log2(max_dilation)) + 1

        def dilation_at_i(i: int) -> int:
            return 2 ** (i % dilation_exp_mod)


        layers = [
            ConvBlock(
                input_channels=input_channels + i * growth_rate,
                output_channels=growth_rate,
                dilation=dilation_at_i(i),
                skip_bn_relu=i == 0 and skip_bn_relu_first_layer,
                kernel_size=kernel_size,
            ) #if i != (n_layers -1) else
              #   LstmBlock(
              #      input_channels=input_channels + i * growth_rate,
              #      output_channels=growth_rate,
              #      num_layers=5, 
              #      skip_bn_relu=False,
              #      batch_first=False)
            for i in range(n_layers)
        ]
        print(layers)
        self.block_sequence = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_sequence(x)
        return x


class ConvBlock(nn.Module):
    """BatchNorm1d + ReLU + Conv1d"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilation: int = 1,
        skip_bn_relu: bool = False,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_channels) if not skip_bn_relu else None
        self.relu = nn.ReLU(inplace=True) if not skip_bn_relu else None
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.bn is not None:
            out = self.bn(out)
        if self.relu is not None:
            out = self.relu(out)
        out = self.conv(out)
        return torch.cat([x, out], dim=1)




class LstmBlock(nn.Module):
    """Optionally BatchNorm1d + ReLU + LSTM"""
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 1,
        skip_bn_relu: bool = False,
        batch_first: bool = True
    ):
        super().__init__()
        self.skip_bn_relu = skip_bn_relu
        if not skip_bn_relu:
            self.bn = nn.BatchNorm1d(input_channels)
            self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=output_channels,
            num_layers=num_layers,
            batch_first=batch_first
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.skip_bn_relu:
            print(x.shape)
            #x = x.transpose(1, 2)  # Assuming input shape is (batch, channels, seq_len)
            x = self.bn(x)
            x = self.relu(x)
            #x = x.transpose(1, 2)  # Revert shape to (batch, seq_len, channels)
        out, (hn, cn) = self.lstm(x)
        return torch.cat([x, out], dim=1)

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        print(input_tensor.size())
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
