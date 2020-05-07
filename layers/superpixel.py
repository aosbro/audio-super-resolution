from torch import nn


def pixel_unshuffle(x, downscale_factor):
    """
    Shuffles the pixels inside the tensor x to change the shape from [B, C, r * H, r * W] to [B, r^2 * C, H, W]
    where r is the upscale factor.
    :param x: original input signal with surface [r * H, r * W].
    :param downscale_factor: factor to decrease the height and width of x
    :return: reshaped tensor
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
    """
    Shuffles the pixels inside the tensor x to change the shape from [B, C, r * W] to [B, r * C, W]
    where r is the upscale factor
    :param x: original input signal with surface [r * H, r * W]
    :param downscale_factor: factor to decrease the height and width of x
    :return: reshaped tensor
    """
    B_in, C_in, W_in = x.size()

    C_out = C_in * downscale_factor
    W_out = int(W_in / downscale_factor)

    x = x.view(B_in, C_in, W_out, downscale_factor)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B_in, C_out, W_out)
    return x


class SuperPixel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downscale_factor, use_convolution=False):
        """
        Initializes the class SuperPixel that implements the pixel shuffle operation that moves pixels from the channel
        dimension to the width and height dimensions followed by a convolution in the low-resolution space (LR).
        :param in_channels: number of channels of the input tensor (scalar int).
        :param out_channels: number of channels of the output tensor (scalar int).
        :param kernel_size: size of the filter of the convolution (scalar int).
        :param downscale_factor: factor by which the height and width should be decreased (scalar int).
        :param use_convolution: boolean indicating whether or not to use convolution before shuffling.
        """
        super(SuperPixel, self).__init__()
        self.downscale_factor = downscale_factor
        self.use_convolution = use_convolution
        padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv2d(in_channels=in_channels * downscale_factor ** 2,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        # Pixel un-shuffling
        x = pixel_unshuffle(x, self.downscale_factor)

        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)
        return x


class SuperPixel1D(nn.Module):
    def __init__(self, in_channels, out_channels, downscale_factor, kernel_size=9, use_convolution=False):
        """
        Initializes the class SuperPixel1D that implements the pixel shuffle operation that moves pixels from the channel
        dimension to the width dimension followed by a convolution in the low-resolution space (LR).
        :param in_channels: number of channels of the input tensor (scalar int).
        :param out_channels: number of channels of the output tensor (scalar int).
        :param kernel_size: size of the filter of the convolution (scalar int).
        :param downscale_factor: factor by which the height and width should be decreased (scalar int).
        :param use_convolution: boolean indicating whether or not to use convolution before shuffling.
        """
        super(SuperPixel1D, self).__init__()
        self.downscale_factor = downscale_factor
        self.use_convolution = use_convolution
        padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv1d(in_channels=in_channels * downscale_factor,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        # Pixel un-shuffling
        x = pixel_unshuffle_1d(x, self.downscale_factor)

        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)
        return x


