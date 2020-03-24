from layers.superpixel import *
from blocks.base_block import *


class DiscriminatorBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p):
        super(DiscriminatorBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels)
        self.batch_normalization = nn.BatchNorm1d(sum(channel_sizes))
        self.dropout = nn.Dropout(p)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.superpixel = SuperPixel1D(downscale_factor=2)

    def forward(self, x):
        x = self.forward_base(x)
        x = self.superpixel(self.activation(self.dropout(self.batch_normalization(x))))
        return x


class DiscriminatorInput(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels):
        super(DiscriminatorInput, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.forward_base(x)
        x = self.activation(x)
        return x


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_features_1, out_features_1, p):
        super(DiscriminatorOutput, self).__init__()
        self.fc_1 = nn.Linear(in_features=in_features_1, out_features=out_features_1)
        self.dropout = nn.Dropout(p)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc_2 = nn.Linear(in_features=out_features_1, out_features=1)

    def forward(self, x):
        B, C, W = x.size()
        x = self.activation_1(self.dropout(self.fc_1(x.view(B, C * W))))
        x = self.fc_2(x)
        return x