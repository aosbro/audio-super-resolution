from blocks.down_block import *
from blocks.up_block import *
from utils.constants import *


class AutoEncoder(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes, bottleneck_channels, p, n_blocks):
        super(AutoEncoder, self).__init__()
        # Encoder
        in_channels_encoder = [1 if i == 0 else 2 * sum(channel_sizes) for i in range(n_blocks)]
        self.encoder = [DownBlock(in_channels=in_channel,
                                  kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes,
                                  bottleneck_channels=bottleneck_channels)
                        for in_channel in in_channels_encoder]
        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        in_channels_decoder = [2 * sum(channel_sizes) if i == 0 else sum(channel_sizes) // 2 for i in range(n_blocks)]
        self.decoder = [UpBlock(in_channels=in_channel,
                                kernel_sizes=kernel_sizes,
                                channel_sizes=channel_sizes,
                                bottleneck_channels=bottleneck_channels,
                                p=p)
                        for in_channel in in_channels_decoder]
        self.decoder = nn.Sequential(*self.decoder)

        # Output convolution
        kernel_size = OUTPUT_KERNEL_SIZE
        padding = (kernel_size - 1) // 2
        self.output_conv = nn.Conv1d(in_channels=int((1/2) * sum(channel_sizes)), out_channels=1,
                                     kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Encoder
        phi = self.encoder(x)

        # Decoder
        x = self.decoder(phi)

        # Output
        x = self.output_conv(x)
        return x, phi
