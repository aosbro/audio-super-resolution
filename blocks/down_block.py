from layers.superpixel import *
from blocks.base_block import *
from utils.constants import *


class DownBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck):
        super(DownBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck)
        self.superpixel = SuperPixel1D(downscale_factor=DOWNSCALE_FACTOR)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x):
        x = self.forward_base(x)
        x = self.superpixel(self.activation(x))
        return x
