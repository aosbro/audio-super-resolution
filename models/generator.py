from blocks.down_block import *
from blocks.up_block import *


class Generator(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes_min, p, n_blocks):
        super(Generator, self).__init__()

        # Compute channel sizes at each level
        channel_sizes = [list(map(lambda c_size: (2 ** min(i, GENERATOR_CHANNEL_FACTOR_MAX)) * c_size, channel_sizes_min))
                         for i in range(n_blocks)]

        # Compute bottleneck channel size at each level
        # bottleneck_channels = [min(channel_size) // 4 for channel_size in channel_sizes]
        bottleneck_channels = [list(map(lambda c: c // GENERATOR_BOTTLENECK_REDUCTION_FACTOR, channel_size))
                               for channel_size in channel_sizes]

        # Encoder
        self.down_block_1 = DownBlock(in_channels=1, kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[0],
                                      bottleneck_channels=bottleneck_channels[0],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_2 = DownBlock(in_channels=2 * sum(channel_sizes[0]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[1], bottleneck_channels=bottleneck_channels[1],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_3 = DownBlock(in_channels=2 * sum(channel_sizes[1]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[2], bottleneck_channels=bottleneck_channels[2],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_4 = DownBlock(in_channels=2 * sum(channel_sizes[2]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[3], bottleneck_channels=bottleneck_channels[3],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_5 = DownBlock(in_channels=2 * sum(channel_sizes[3]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[4], bottleneck_channels=bottleneck_channels[4],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_6 = DownBlock(in_channels=2 * sum(channel_sizes[4]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[5], bottleneck_channels=bottleneck_channels[5],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_7 = DownBlock(in_channels=2 * sum(channel_sizes[5]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[6], bottleneck_channels=bottleneck_channels[6],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.down_block_8 = DownBlock(in_channels=2 * sum(channel_sizes[6]), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes[7], bottleneck_channels=bottleneck_channels[7],
                                      use_bottleneck=GENERATOR_USE_BOTTLENECK)

        # Decoder
        self.up_block_1 = UpBlock(in_channels=2 * sum(channel_sizes[7]), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes[7], bottleneck_channels=bottleneck_channels[7], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_2 = UpBlock(in_channels=sum(channel_sizes[7]) // 2 + 2 * sum(channel_sizes[6]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[6],
                                  bottleneck_channels=bottleneck_channels[6], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_3 = UpBlock(in_channels=sum(channel_sizes[6]) // 2 + 2 * sum(channel_sizes[5]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[5],
                                  bottleneck_channels=bottleneck_channels[5], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_4 = UpBlock(in_channels=sum(channel_sizes[5]) // 2 + 2 * sum(channel_sizes[4]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[4],
                                  bottleneck_channels=bottleneck_channels[4], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_5 = UpBlock(in_channels=sum(channel_sizes[4]) // 2 + 2 * sum(channel_sizes[3]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[3],
                                  bottleneck_channels=bottleneck_channels[3], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_6 = UpBlock(in_channels=sum(channel_sizes[3]) // 2 + 2 * sum(channel_sizes[2]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[2],
                                  bottleneck_channels=bottleneck_channels[2], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_7 = UpBlock(in_channels=sum(channel_sizes[2]) // 2 + 2 * sum(channel_sizes[1]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[1],
                                  bottleneck_channels=bottleneck_channels[1], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)
        self.up_block_8 = UpBlock(in_channels=sum(channel_sizes[1]) // 2 + 2 * sum(channel_sizes[0]),
                                  kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[0],
                                  bottleneck_channels=bottleneck_channels[0], p=p,
                                  use_bottleneck=GENERATOR_USE_BOTTLENECK)

        # Output convolution
        kernel_size = 27
        padding = (kernel_size - 1) // 2
        self.output_conv = nn.Conv1d(in_channels=sum(channel_sizes[0]) // 2, out_channels=1,
                                     kernel_size=kernel_size, padding=padding)

        self.tanh = nn.Tanh()

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
        x = self.up_block_1(d8, d7)
        x = self.up_block_2(x, d6)
        x = self.up_block_3(x, d5)
        x = self.up_block_4(x, d4)
        x = self.up_block_5(x, d3)
        x = self.up_block_6(x, d2)
        x = self.up_block_7(x, d1)
        x = self.up_block_8(x, None)
        return self.tanh(self.output_conv(x) + x_l)
