from blocks.down_block import *
from blocks.up_block import *


class Generator(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes_min, p, n_blocks):
        super(Generator, self).__init__()

        # Compute channel sizes at each level
        channel_sizes = [list(map(lambda c_size: (2 ** min(i, CHANNEL_FACTOR_MAX)) * c_size, channel_sizes_min))
                         for i in range(n_blocks)]

        # Compute bottleneck channel size at each level
        bottleneck_channels = [min(channel_size) // 4 for channel_size in channel_sizes]

        # Encoder
        self.down_block_1 = DownBlock(in_channels=1, kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[0],
                                      bottleneck_channels=bottleneck_channels[0])
        self.down_block_2 = DownBlock(in_channels=2 * sum(channel_sizes[0]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[1], bottleneck_channels=bottleneck_channels[1])
        self.down_block_3 = DownBlock(in_channels=2 * sum(channel_sizes[1]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[2], bottleneck_channels=bottleneck_channels[2])
        self.down_block_4 = DownBlock(in_channels=2 * sum(channel_sizes[2]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[3], bottleneck_channels=bottleneck_channels[3])
        self.down_block_5 = DownBlock(in_channels=2 * sum(channel_sizes[3]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[4], bottleneck_channels=bottleneck_channels[4])
        self.down_block_6 = DownBlock(in_channels=2 * sum(channel_sizes[4]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[5], bottleneck_channels=bottleneck_channels[5])
        self.down_block_7 = DownBlock(in_channels=2 * sum(channel_sizes[5]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[6], bottleneck_channels=bottleneck_channels[6])
        self.down_block_8 = DownBlock(in_channels=2 * sum(channel_sizes[6]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[7], bottleneck_channels=bottleneck_channels[7])

        # Decoder
        self.up_block_1 = UpBlock(in_channels=2 * sum(channel_sizes[7]), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes[7], bottleneck_channels=bottleneck_channels[7], p=p)
        self.up_block_2 = UpBlock(in_channels=sum(channel_sizes[7]) // 2 + 2 * sum(channel_sizes[6]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[6],
                                  bottleneck_channels=bottleneck_channels[6], p=p)
        self.up_block_3 = UpBlock(in_channels=sum(channel_sizes[6]) // 2 + 2 * sum(channel_sizes[5]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[5],
                                  bottleneck_channels=bottleneck_channels[5], p=p)
        self.up_block_4 = UpBlock(in_channels=sum(channel_sizes[5]) // 2 + 2 * sum(channel_sizes[4]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[4],
                                  bottleneck_channels=bottleneck_channels[4], p=p)
        self.up_block_5 = UpBlock(in_channels=sum(channel_sizes[4]) // 2 + 2 * sum(channel_sizes[3]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[3],
                                  bottleneck_channels=bottleneck_channels[3], p=p)
        self.up_block_6 = UpBlock(in_channels=sum(channel_sizes[3]) // 2 + 2 * sum(channel_sizes[2]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[2],
                                  bottleneck_channels=bottleneck_channels[2], p=p)
        self.up_block_7 = UpBlock(in_channels=sum(channel_sizes[2]) // 2 + 2 * sum(channel_sizes[1]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[1],
                                  bottleneck_channels=bottleneck_channels[1], p=p)
        self.up_block_8 = UpBlock(in_channels=sum(channel_sizes[1]) // 2 + 2 * sum(channel_sizes[0]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[0],
                                  bottleneck_channels=bottleneck_channels[0], p=p)

        # Output convolution
        kernel_size = 27
        padding = (kernel_size - 1) // 2
        self.output_conv = nn.Conv1d(in_channels=sum(channel_sizes[0]) // 2, out_channels=1,
                                     kernel_size=kernel_size, padding=padding)

    def forward(self, x_l):
        # Encoder
        d1 = self.down_block_1(x_l)
        d2 = self.down_block_2(d1)
        d3 = self.down_block_3(d2)
        d4 = self.down_block_4(d3)
        d5 = self.down_block_5(d4)
        d6 = self.down_block_6(d5)
        d7 = self.down_block_7(d6)
        d8 = self.down_block_8(d7)

        # Decoder
        u1 = self.up_block_1(d8, d7)
        u2 = self.up_block_2(u1, d6)
        u3 = self.up_block_3(u2, d5)
        u4 = self.up_block_4(u3, d4)
        u5 = self.up_block_5(u4, d3)
        u6 = self.up_block_6(u5, d2)
        u7 = self.up_block_7(u6, d1)
        u8 = self.up_block_8(u7, None)
        return self.output_conv(u8) + x_l
