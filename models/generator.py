from blocks.down_block import *
from blocks.up_block import *


class Generator(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes, bottleneck_channels, p):
        super(Generator, self).__init__()
        # Encoder
        self.down_block_1 = DownBlock(in_channels=1, kernel_sizes=kernel_sizes, channel_sizes=channel_sizes,
                                      bottleneck_channels=bottleneck_channels)
        self.down_block_2 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_3 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_4 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_5 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        # Decoder
        self.up_block_1 = UpBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_2 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_3 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_4 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_5 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)

    def forward(self, x_l):
        # Encoder
        d1 = self.down_block_1(x_l)
        d2 = self.down_block_2(d1)
        d3 = self.down_block_3(d2)
        d4 = self.down_block_4(d3)
        d5 = self.down_block_5(d4)

        # Decoder
        u1 = self.up_block_1(d5, d4)
        u2 = self.up_block_2(u1, d3)
        u3 = self.up_block_3(u2, d2)
        u4 = self.up_block_4(u3, d1)
        return u4
