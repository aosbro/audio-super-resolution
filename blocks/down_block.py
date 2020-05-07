from layers.superpixel import SuperPixel1D
from blocks.base_block import BaseBlock
from utils.constants import *
from torch import nn


class DownBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck):
        """
        Initializes the class DownBlock that inherits its main properties from BaseBlock.
        DownBlock is the main ingredient of the encoding part of both the generator and the auto-encoder.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param kernel_sizes: kernel sizes for each scale of the multi-scale convolution (list of scalar int).
        :param channel_sizes: number of filters for each scale of the multi-scale convolution (list of scalar int).
        :param bottleneck_channels: number of filters for each of the multi-scale bottleneck convolution.
        :param use_bottleneck: boolean indicating whether to use the bottleneck channels or not.
        """
        super(DownBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck)
        self.superpixel = SuperPixel1D(in_channels=sum(channel_sizes),
                                       out_channels=2 * sum(channel_sizes),
                                       downscale_factor=DOWNSCALE_FACTOR)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        x = self.forward_base(x)
        x = self.superpixel(self.activation(x))
        return x
