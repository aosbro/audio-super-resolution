from layers.subpixel import *
from blocks.base_block import *


class UpBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p):
        super(UpBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels)
        self.subpixel = SubPixel1D(upscale_factor=2)
        self.dropout = nn.Dropout(p)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x, x_shortcut):
        x = self.forward_base(x)
        x = self.activation(self.dropout(x))
        x = self.subpixel(x)
        return torch.cat([x, x_shortcut], dim=1)
