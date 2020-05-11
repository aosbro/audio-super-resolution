from blocks.down_block import DownBlock
from blocks.up_block import UpBlock
from utils.constants import *
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes_min, p, n_blocks, return_embedding=True):
        """
        Initializes the class AutoEncoder which is the model used to extract features from audio signals. Its
        architecture is similar to the one of the generator with the difference that the encoder and decoder contain 4
         DownBlocks and 4 UpBlocks respectively. Furthermore, the auto-encoder does not embed any skip connection as it
         is desired that 
        :param kernel_sizes:
        :param channel_sizes_min:
        :param p:
        :param n_blocks:
        :param return_embedding:
        """
        super(AutoEncoder, self).__init__()

        # Compute channel sizes at each level
        channel_sizes = [list(map(lambda c_size: (2 ** min(i, AUTOENCODER_CHANNEL_FACTOR_MAX)) * c_size,
                                  channel_sizes_min))
                         for i in range(n_blocks)]

        # Compute bottleneck channel size at each level
        bottleneck_channels = [list(map(lambda c: c // AUTOENCODER_BOTTLENECK_REDUCTION_FACTOR, channel_size))
                               for channel_size in channel_sizes]

        # Compute the number of input channel for the encoder
        in_channels_encoder = [1 if i == 0 else 2 * sum(channel_sizes[i-1]) for i in range(n_blocks)]

        # Encoder
        self.encoder = [DownBlock(in_channels=in_channel,
                                  kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_size,
                                  bottleneck_channels=bottleneck_channel,
                                  use_bottleneck=AUTOENCODER_USE_BOTTLENECK)
                        for in_channel, channel_size, bottleneck_channel in zip(in_channels_encoder, channel_sizes,
                                                                                bottleneck_channels)]
        self.encoder = nn.Sequential(*self.encoder)

        # Compute the number of input channel for the decoder
        in_channels_decoder = [2 * sum(channel_sizes[n_blocks - i - 1]) if i == 0 else
                               sum(channel_sizes[n_blocks - i]) // 2 for i in range(n_blocks)]

        # Decoder
        self.decoder = [UpBlock(in_channels=in_channel,
                                kernel_sizes=kernel_sizes,
                                channel_sizes=channel_size,
                                bottleneck_channels=bottleneck_channel,
                                p=p,
                                use_bottleneck=AUTOENCODER_USE_BOTTLENECK)
                        for in_channel, channel_size, bottleneck_channel in zip(in_channels_decoder,
                                                                                channel_sizes[::-1],
                                                                                bottleneck_channels[::-1])]
        self.decoder = nn.Sequential(*self.decoder)

        # Output convolution
        kernel_size = OUTPUT_KERNEL_SIZE
        padding = (kernel_size - 1) // 2
        self.output_conv = nn.Conv1d(in_channels=sum(channel_sizes[0]) // 2, out_channels=1,
                                     kernel_size=kernel_size, padding=padding)

        # Boolean indicating if the auto-encoder should return the latent space representation
        self.return_embedding = return_embedding

    def forward(self, x):
        # Encoder
        phi = self.encoder(x)

        # Decoder
        x = self.decoder(phi)

        # Output
        x = self.output_conv(x)
        if self.return_embedding:
            return x, phi
        return x
