from blocks.discriminator_block import DiscriminatorInput, DiscriminatorBlock, DiscriminatorOutput
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, general_args):
        super(Discriminator, self).__init__()
        # Compute the input width at each level
        input_widths = [general_args.window_length // 2 ** i for i in range(general_args.discriminator_n_block)]

        # Compute channel sizes at each level
        channel_sizes = [list(map(lambda c_size: (2 ** min(i, general_args.discriminator_channel_factor_max)) * c_size,
                                  general_args.channel_sizes_min)) for i in range(general_args.discriminator_n_block)]

        # Compute bottleneck channel size at each level
        bottleneck_channels = [list(map(lambda c: max(1, c // general_args.discriminator_bottleneck_reduction_factor),
                                        channel_size)) for channel_size in channel_sizes]

        # Define the first block
        self.in_block = DiscriminatorInput(in_channels=1, channel_sizes=channel_sizes[0],
                                           bottleneck_channels=bottleneck_channels[0], general_args=general_args)

        # Define the intermediate blocks
        in_channels = [2 ** min(i, general_args.discriminator_channel_factor_max + 1) *
                       sum(general_args.channel_sizes_min) for i in range(general_args.discriminator_n_block)]

        self.mid_blocks = [DiscriminatorBlock(in_channels=in_channel,
                                              channel_sizes=channel_size,
                                              bottleneck_channels=bottleneck_channel,
                                              input_width=input_width,
                                              general_args=general_args)
                           for in_channel, channel_size, bottleneck_channel, input_width in zip(in_channels,
                                                                                                channel_sizes,
                                                                                                bottleneck_channels,
                                                                                                input_widths)]
        self.mid_blocks = nn.Sequential(*self.mid_blocks)

        # Define the last block
        self.out_block = DiscriminatorOutput(
            in_features_1=int(2 * sum(channel_sizes[-1]) * general_args.window_length * 2 ** - general_args.
                              discriminator_n_block),
            out_features_1=general_args.fc1_output_features, general_args=general_args)

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_blocks(x)
        return self.out_block(x)
