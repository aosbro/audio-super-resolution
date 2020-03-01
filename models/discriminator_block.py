from layers.superpixel import *
import torch


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p):
        super(DiscriminatorBlock, self).__init__()
        paddings = [(kernel_size - 1) // 2 for kernel_size in kernel_sizes]
        n_scales = len(kernel_sizes)
        self.conv_layers_1 = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                                      out_channels=bottleneck_channels,
                                                      kernel_size=1)
                                            for i in range(n_scales)])
        self.conv_layers_2 = nn.ModuleList([nn.Conv1d(in_channels=bottleneck_channels,
                                                      out_channels=channel_size,
                                                      kernel_size=kernel_size,
                                                      padding=padding)
                                            for kernel_size, channel_size, padding in zip(kernel_sizes, channel_sizes,
                                                                                          paddings)])
        self.batch_normalization = nn.BatchNorm1d(sum(channel_sizes))
        self.dropout = nn.Dropout(p)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.superpixel = SuperPixel1D(downscale_factor=2)

    def forward(self, x):
        x = [conv_layer(x) for conv_layer in self.conv_layers_1]
        x = torch.cat(list(map(lambda temp_x, conv_layer: conv_layer(temp_x), x, self.conv_layers_2)), dim=1)
        x = self.superpixel(self.activation(self.dropout(self.batch_normalization(x))))
        return x
