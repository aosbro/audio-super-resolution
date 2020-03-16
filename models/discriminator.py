from blocks.discriminator_block import *
from utils.constants import *


class Discriminator(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes_min, p, n_blocks):
        super(Discriminator, self).__init__()

        # Compute channel sizes at each level
        channel_sizes = [list(map(lambda c_size: (2 ** min(i, CHANNEL_FACTOR_MAX)) * c_size, channel_sizes_min))
                         for i in range(n_blocks)]

        # Compute bottleneck channel size at each level
        bottleneck_channels = [min(channel_size) // 4 for channel_size in channel_sizes]

        # Define the first block
        self.in_block = DiscriminatorInput(in_channels=1, kernel_sizes=kernel_sizes, channel_sizes=channel_sizes[0],
                                           bottleneck_channels=bottleneck_channels[0])

        # Define the intermediate blocks
        in_channels = [2 ** min(i, CHANNEL_FACTOR_MAX + 1) * sum(CHANNEL_SIZES_MIN) for i in range(n_blocks)]

        self.mid_blocks = [DiscriminatorBlock(in_channels=in_channel,
                                              kernel_sizes=kernel_sizes,
                                              channel_sizes=channel_size,
                                              bottleneck_channels=bottleneck_channel,
                                              p=p)
                           for in_channel, channel_size, bottleneck_channel in zip(in_channels, channel_sizes, bottleneck_channels)]
        self.mid_blocks = nn.Sequential(*self.mid_blocks)

        # Define the last block
        self.out_block = DiscriminatorOutput(in_features_1=int(2 * sum(channel_sizes[-1])*WINDOW_LENGTH*2**-n_blocks),
                                             out_features_1=64, p=p)

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_blocks(x)
        return self.out_block(x)
