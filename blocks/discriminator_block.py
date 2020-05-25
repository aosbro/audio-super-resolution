from layers.superpixel import SuperPixel1D
from blocks.base_block import BaseBlock
from torch import nn


class DiscriminatorBlock(BaseBlock):
    def __init__(self, in_channels, channel_sizes, bottleneck_channels, input_width, general_args):
        """
        Initializes the class DiscriminatorBlock that inherits its main properties from BaseBlock. DiscriminatorBlock
        is the main ingredient of the discriminator.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param channel_sizes: number of filters for each scale of the multi-scale convolution (list of scalar int).
        :param bottleneck_channels: number of filters for each of the multi-scale bottleneck convolution.
        :param input_width: width of the block's input (scalar int).
        :param general_args: argument parser that contains the arguments that are independent to the script being
        executed.
        """
        super(DiscriminatorBlock, self).__init__(in_channels, general_args.kernel_sizes, channel_sizes,
                                                 bottleneck_channels, general_args.discriminator_use_bottleneck)
        if general_args.use_layer_norm:
            self.normalization = nn.LayerNorm([sum(channel_sizes), input_width])
        else:
            self.normalization = nn.BatchNorm1d(sum(channel_sizes))
        self.dropout = nn.Dropout(general_args.dropout_probability)
        self.activation = nn.LeakyReLU(negative_slope=general_args.leaky_relu_slope)
        self.superpixel = SuperPixel1D(in_channels=sum(channel_sizes),
                                       out_channels=general_args.downscale_factor * sum(channel_sizes),
                                       downscale_factor=general_args.downscale_factor)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map.
        """
        x = self.forward_base(x)
        x = self.superpixel(self.activation(self.dropout(self.normalization(x))))
        return x


class DiscriminatorInput(BaseBlock):
    def __init__(self, in_channels, channel_sizes, bottleneck_channels, general_args):
        """
        Initializes the class DiscriminatorInput that implements the input layer of the discriminator. It inherits its
        main properties from BaseBlock.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param channel_sizes: number of filters for each scale of the multi-scale convolution (list of scalar int).
        :param bottleneck_channels: number of filters for each of the multi-scale bottleneck convolution.
        :param general_args: argument parser that contains the arguments that are independent to the script being
        executed.
        """
        super(DiscriminatorInput, self).__init__(in_channels, general_args.kernel_sizes, channel_sizes,
                                                 bottleneck_channels, general_args.discriminator_use_bottleneck)
        self.activation = nn.LeakyReLU(negative_slope=general_args.leaky_relu_slope)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        x = self.forward_base(x)
        x = self.activation(x)
        return x


class DiscriminatorOutput(nn.Module):
    def __init__(self, in_features_1, out_features_1, general_args):
        """
        Initializes the class DiscriminatorOutput that implements the output layer of the discriminator. It inherits its
        main properties from BaseBlock.
        :param in_features_1: number of input features of the first linear layer (scalar int).
        :param out_features_1: number of output features of the first linear layer (scalar int).
        :param general_args: argument parser that contains the arguments that are independent to the script being
        executed.
        """
        super(DiscriminatorOutput, self).__init__()
        self.fc_1 = nn.Linear(in_features=in_features_1, out_features=out_features_1)
        self.dropout = nn.Dropout(general_args.dropout_probability)
        self.activation_1 = nn.LeakyReLU(negative_slope=general_args.leaky_relu_slope)
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
