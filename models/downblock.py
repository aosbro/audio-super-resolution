import torch
from layers.superpixel import *


class DownBlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes, channel_sizes):
        super(DownBlock, self).__init__()
        paddings = [(kernel_size - 1) // 2 for kernel_size in kernel_sizes]
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=in_channels,
                                                    out_channels=channel_size,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                                          for kernel_size, channel_size, padding in zip(kernel_sizes, channel_sizes,
                                                                                        paddings)])
        self.super_pixel = SuperPixel1D(downscale_factor=2)
        self.activation = nn.PReLU(sum(channel_sizes))

    def forward(self, x):
        x = self.activation(torch.cat([conv_layer(x) for conv_layer in self.conv_layers], dim=1))
        x = self.super_pixel(x)
        return x


def main():
    in_channels = 16
    kernel_sizes = [3, 9, 27, 81]
    channel_sizes = [8, 16, 32, 64]
    down_block = DownBlock(in_channels, kernel_sizes, channel_sizes)
    [print(layer) for layer in down_block.conv_layers]
    x = torch.randn(10, 16, 8000)
    y = down_block(x)
    # [print(output.shape) for output in y]
    print(y.shape)


if __name__ == '__main__':
    main()