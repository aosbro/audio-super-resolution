from blocks.down_block import *
from blocks.up_block import *
from utils.constants import *


class AutoEncoder(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes_min, p, n_blocks):
        super(AutoEncoder, self).__init__()

        # Compute channel sizes at each level
        channel_sizes = [list(map(lambda c_size: (2 ** min(i, CHANNEL_FACTOR_MAX)) * c_size, channel_sizes_min))
                         for i in range(n_blocks)]

        # Compute bottleneck channel size at each level
        bottleneck_channels = [min(channel_size) // 4 for channel_size in channel_sizes]

        # Compute the number of input channel for the encoder
        in_channels_encoder = [1 if i == 0 else 2 * sum(channel_sizes[i-1]) for i in range(n_blocks)]

        # Encoder
        self.encoder = [DownBlock(in_channels=in_channel,
                                  kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_size,
                                  bottleneck_channels=bottleneck_channel)
                        for in_channel, channel_size, bottleneck_channel in zip(in_channels_encoder, channel_sizes,
                                                                                bottleneck_channels)]
        self.encoder = nn.Sequential(*self.encoder)

        # Compute the number of input channel for the decoder
        in_channels_decoder = [2 * sum(channel_sizes[n_blocks - i - 1]) if i == 0 else
                               sum(channel_sizes[n_blocks - i - 1]) for i in range(n_blocks)]

        # Decoder
        self.decoder = [UpBlock(in_channels=in_channel,
                                kernel_sizes=kernel_sizes,
                                channel_sizes=channel_size,
                                bottleneck_channels=bottleneck_channel,
                                p=p)
                        for in_channel, channel_size, bottleneck_channel in zip(in_channels_decoder, channel_sizes[::-1],
                                                                                bottleneck_channels[::-1])]
        self.decoder = nn.Sequential(*self.decoder)

        # Output convolution
        kernel_size = OUTPUT_KERNEL_SIZE
        padding = (kernel_size - 1) // 2
        self.output_conv = nn.Conv1d(in_channels=sum(channel_sizes[0]) // 2, out_channels=1,
                                     kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Encoder
        phi = self.encoder(x)

        # Decoder
        x = self.decoder(phi)

        # Output
        x = self.output_conv(x)
        return x, phi
