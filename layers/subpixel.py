from torch import nn


def pixel_shuffle(x, upscale_factor):
    """
    Shuffles the pixels inside the tensor x to change the shape from [B, r^2 * C, H, W] to [B, C, r * H, r * W]
    where r is the upscale factor
    :param x: Original input signal with r^2 * C channels
    :param upscale_factor: Factor to increase the height and width of x
    :return: Reshaped tensor
    """
    B_in, C_in, H_in, W_in = x.size()

    C_out = C_in // upscale_factor ** 2
    H_out = H_in * upscale_factor
    W_out = W_in * upscale_factor

    x = x.view(B_in, C_out, upscale_factor, upscale_factor, H_in, W_in)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B_in, C_out, H_out, W_out)
    return x


def pixel_shuffle_1d(x, upscale_factor):
    """
    Shuffles the pixels inside the tensor x to change the shape from [B, r * C, W] to [B, C, r * W]
    where r is the upscale factor
    :param x: Original input signal with r * C channels
    :param upscale_factor: Factor to increase the width of x
    :return: Reshaped tensor
    """
    B_in, C_in, W_in = x.size()
    C_out = C_in // upscale_factor
    W_out = W_in * upscale_factor

    x = x.view(B_in, C_out, upscale_factor, W_in)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B_in, C_out, W_out)
    return x


class SubPixel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor, use_convolution=False):
        """
        Initializes the class SubPixel that implements a convolution in the low-resolution space (LR) followed by the
        pixel shuffle operation that moves pixels from the channel dimension to the width and height dimensions.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param out_channels: number of channels of the output tensor (scalar int).
        :param kernel_size: size of the filter of the convolution (scalar int).
        :param upscale_factor: factor by which the height and width should be increased (scalar int).
        :param use_convolution: boolean indicating whether or not to use convolution before shuffling.
        """
        super(SubPixel, self).__init__()
        self.upscale_factor = upscale_factor
        self.use_convolution = use_convolution
        self.padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels * upscale_factor ** 2,
                                    kernel_size=kernel_size,
                                    padding=self.padding)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)

        # Pixel shuffling
        x = pixel_shuffle(x, self.upscale_factor)
        return x


class SubPixel1D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, kernel_size=9, use_convolution=False):
        """
        Initializes the class SubPixel1D that implements a convolution in the low-resolution space (LR) followed by the
        pixel shuffle operation that moves pixels from the channel dimension to the width dimension.
        :param in_channels: number of channels of the input tensor (scalar int).
        :param out_channels: number of channels of the output tensor (scalar int).
        :param kernel_size: size of the filter of the convolution (scalar int).
        :param upscale_factor: factor by which the width should be increased (scalar int).
        :param use_convolution: boolean indicating whether or not to use convolution before shuffling.
        """
        super(SubPixel1D, self).__init__()
        self.upscale_factor = upscale_factor
        self.use_convolution = use_convolution
        self.padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels * upscale_factor,
                                    kernel_size=kernel_size,
                                    padding=self.padding)

    def forward(self, x):
        """
        :param x: input feature map.
        :return: output feature map
        """
        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)

        # Pixel shuffling
        x = pixel_shuffle_1d(x, self.upscale_factor)
        return x
