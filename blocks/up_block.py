from layers.subpixel import SubPixel1D
from blocks.base_block import BaseBlock
import torch
from torch import nn
from utils.constants import *


class UpBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p, use_bottleneck):
        """
        Initializes the class UpBlock that inherits its main properties from BaseBlock.
        UpBlock is the main ingredient of the decoding part of both the generator and the auto-encoder.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param kernel_sizes: kernel sizes for each scale of the multi-scale convolution (list of scalar int).
        :param channel_sizes: number of filters for each scale of the multi-scale convolution (list of scalar int).
        :param bottleneck_channels: number of filters for each of the multi-scale bottleneck convolution.
        :param p: dropout probability of channels (float in [0, 1)).
        :param use_bottleneck: boolean indicating whether to use the bottleneck channels or not.
        """
        super(UpBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck)
        self.subpixel = SubPixel1D(in_channels=sum(channel_sizes),
                                   out_channels=sum(channel_sizes) // 2,
                                   upscale_factor=UPSCALE_FACTOR)
        self.dropout = nn.Dropout(p)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x, x_shortcut=None):
        """
        :param x: input feature map.
        :param x_shortcut: short-cut feature map from the encoding part.
        :return: output feature map.
        """
        x = self.forward_base(x)
        x = self.activation(self.dropout(x))
        x = self.subpixel(x)
        if x_shortcut is None:
            return x
        return torch.cat([x, x_shortcut], dim=1)
