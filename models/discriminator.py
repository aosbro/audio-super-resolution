from blocks.discriminator_block import *


class Discriminator(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes, bottleneck_channels, p, n_blocks):
        super(Discriminator, self).__init__()
        input_size = 8192
        self.in_block = DiscriminatorInput(in_channels=1, kernel_sizes=kernel_sizes, channel_sizes=channel_sizes,
                                           bottleneck_channels=bottleneck_channels)
        in_channels = [sum(channel_sizes) if i == 0 else sum(channel_sizes) * 2 for i in range(n_blocks)]
        self.mid_blocks = [DiscriminatorBlock(in_channels=in_channel,
                                              kernel_sizes=kernel_sizes,
                                              channel_sizes=channel_sizes,
                                              bottleneck_channels=bottleneck_channels,
                                              p=p)
                           for in_channel in in_channels]
        self.mid_blocks = nn.Sequential(*self.mid_blocks)
        self.out_block = DiscriminatorOutput(in_features_1=int(2*sum(channel_sizes)*input_size*2**-n_blocks),
                                             out_features_1=64, p=p)

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_blocks(x)
        return self.out_block(x)

