from layers.superpixel import SuperPixel1D
from blocks.base_block import BaseBlock
from utils.constants import *
from torch import nn


class DiscriminatorBlock(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p, use_bottleneck):
        """
        Initializes the class DiscriminatorBlock that inherits its main properties from BaseBlock. DiscriminatorBlock
        is the main ingredient of the discriminator.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param kernel_sizes: kernel sizes for each scale of the multi-scale convolution (list of scalar int).
        :param channel_sizes: number of filters for each scale of the multi-scale convolution (list of scalar int).
        :param bottleneck_channels: number of filters for each of the multi-scale bottleneck convolution.
        :param p: dropout probability of channels (float in [0, 1)).
        :param use_bottleneck: boolean indicating whether to use the bottleneck channels or not.
        """
        super(DiscriminatorBlock, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels,
                                                 use_bottleneck)
        self.batch_normalization = nn.BatchNorm1d(sum(channel_sizes))
        self.dropout = nn.Dropout(p)
        self.activation = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)
        self.superpixel = SuperPixel1D(in_channels=sum(channel_sizes),
                                       out_channels=2 * sum(channel_sizes),
                                       downscale_factor=DOWNSCALE_FACTOR)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map.
        """
        x = self.forward_base(x)
        x = self.superpixel(self.activation(self.dropout(self.batch_normalization(x))))
        return x


class DiscriminatorInput(BaseBlock):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck):
        """
        Initializes the class DiscriminatorInput that implements the input layer of the discriminator. It inherits its
        main properties from BaseBlock.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param kernel_sizes: kernel sizes for each scale of the multi-scale convolution (list of scalar int).
        :param channel_sizes: number of filters for each scale of the multi-scale convolution (list of scalar int).
        :param bottleneck_channels: number of filters for each of the multi-scale bottleneck convolution.
        :param use_bottleneck: boolean indicating whether to use the bottleneck channels or not.
        """
        super(DiscriminatorInput, self).__init__(in_channels, kernel_sizes, channel_sizes, bottleneck_channels,
                                                 use_bottleneck)
        self.activation = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        x = self.forward_base(x)
        x = self.activation(x)
        return x


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_features_1, out_features_1, p):
        """
        Initializes the class DiscriminatorOutput that implements the output layer of the discriminator. It inherits its
        main properties from BaseBlock.
        :param in_features_1: number of input features of the first linear layer (scalar int).
        :param out_features_1: number of output features of the first linear layer (scalar int).
        :param p: dropout probability for the first linear layer (float in [0, 1)).
        """
        super(DiscriminatorOutput, self).__init__()
        self.fc_1 = nn.Linear(in_features=in_features_1, out_features=out_features_1)
        self.dropout = nn.Dropout(p)
        self.activation_1 = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE)
        self.fc_2 = nn.Linear(in_features=out_features_1, out_features=1)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map.
        """
        B, C, W = x.size()
        x = self.activation_1(self.dropout(self.fc_1(x.view(B, C * W))))
        x = self.fc_2(x)
        return x