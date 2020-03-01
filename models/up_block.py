import torch
from layers.subpixel import *


class UpBlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes, channel_sizes, bottleneck_channels, p):
        super(UpBlock, self).__init__()
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
        self.subpixel = SubPixel1D(upscale_factor=2)
        self.dropout = nn.Dropout(p)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x, x_shortcut):
        x = [conv_layer(x) for conv_layer in self.conv_layers_1]
        x = torch.cat(list(map(lambda temp_x, conv_layer: conv_layer(temp_x), x, self.conv_layers_2)), dim=1)
        x = self.activation(self.dropout(x))
        x = self.subpixel(x)
        return torch.cat([x, x_shortcut], dim=1)
