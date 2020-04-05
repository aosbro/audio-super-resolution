from torch import nn
import torch


class BaseBlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, use_bottleneck=False):
        """
        Initializes the class BaseBlock which is the parent class to all blocks
        :param in_channels: Number of input channel
        :param kernel_sizes: List of kernel sizes, one entry per scale
        :param channel_sizes: Number of output channel, one entry per scale
        :param bottleneck_channels: Number of output channel of the intermediate convolution used to reduce
        computational cost
        """
        super(BaseBlock, self).__init__()
        paddings = [(kernel_size - 1) // 2 for kernel_size in kernel_sizes]
        self.use_bottleneck = use_bottleneck

        # Define convolution layers
        if use_bottleneck:
            self.conv_layers_1 = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                                          out_channels=bottleneck_channel,
                                                          kernel_size=1)
                                                for bottleneck_channel in bottleneck_channels])
            self.conv_layers_2 = nn.ModuleList([nn.Conv1d(in_channels=bottleneck_channel,
                                                          out_channels=channel_size,
                                                          kernel_size=kernel_size,
                                                          padding=padding)
                                                for bottleneck_channel, kernel_size, channel_size, padding in
                                                zip(bottleneck_channels, kernel_sizes, channel_sizes, paddings)])
        else:
            self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                                        out_channels=channel_size,
                                                        kernel_size=kernel_size,
                                                        padding=padding)
                                              for kernel_size, channel_size, padding in zip(kernel_sizes, channel_sizes,
                                                                                            paddings)])

    def forward_base(self, x):
        """
        :param x: Input feature map
        :return: Stacked 
        """
        if self.use_bottleneck:
            x = [conv_layer(x) for conv_layer in self.conv_layers_1]
            x = torch.cat(list(map(lambda temp_x, conv_layer: conv_layer(temp_x), x, self.conv_layers_2)), dim=1)
            return x
        return torch.cat([conv_layer(x) for conv_layer in self.conv_layers], dim=1)
