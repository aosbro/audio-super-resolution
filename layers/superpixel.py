from torch import nn


def pixel_unshuffle(x, downscale_factor):
    """
    Shuffles the pixels inside the tensor x to change the shape from [B, C, r * H, r * W] to [B, r^2 * C, H, W]
    where r is the upscale factor
    :param x: Original input signal with surface [r * H, r * W]
    :param downscale_factor: Factor to decrease the height and width of x
    :return: Reshaped tensor
    """
    B_in, C_in, H_in, W_in = x.size()

    C_out = C_in * downscale_factor ** 2
    H_out = int(H_in / downscale_factor)
    W_out = int(W_in / downscale_factor)

    x = x.view(B_in, C_in, H_out, downscale_factor, W_out, downscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B_in, C_out, H_out, W_out)
    return x


def pixel_unshuffle_1d(x, downscale_factor):
    B_in, C_in, W_in = x.size()

    C_out = C_in * downscale_factor
    W_out = int(W_in / downscale_factor)

    x = x.view(B_in, C_in, W_out, downscale_factor)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B_in, C_out, W_out)
    return x


class SuperPixel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downscale_factor, use_convolution=True):
        super(SuperPixel, self).__init__()
        self.downscale_factor = downscale_factor
        self.use_convolution = use_convolution
        padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv2d(in_channels=in_channels * downscale_factor ** 2,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        # Pixel un-shuffling
        x = pixel_unshuffle(x, self.downscale_factor)

        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)
        return x


class SuperPixel1D(nn.Module):
    def __init__(self, in_channels, out_channels, downscale_factor, kernel_size=9, use_convolution=True):
        super(SuperPixel1D, self).__init__()
        self.downscale_factor = downscale_factor
        self.use_convolution = use_convolution
        padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv1d(in_channels=in_channels * downscale_factor,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        # Pixel un-shuffling
        x = pixel_unshuffle_1d(x, self.downscale_factor)

        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)
        return x


