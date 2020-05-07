from layers.subpixel import SubPixel1D
from blocks.base_block import BaseBlock
import torch
from torch import nn
from utils.constants import *


class UpBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p, use_bottleneck):
        super(UpBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck)
        self.subpixel = SubPixel1D(in_channels=sum(channel_sizes),
                                   out_channels=sum(channel_sizes) // 2,
                                   upscale_factor=UPSCALE_FACTOR)
        self.dropout = nn.Dropout(p)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x, x_shortcut=None):
        x = self.forward_base(x)
        x = self.activation(self.dropout(x))
        x = self.subpixel(x)
        if x_shortcut is None:
            return x
        return torch.cat([x, x_shortcut], dim=1)