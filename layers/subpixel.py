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
    B_in, C_in, W_in = x.size()
    C_out = C_in // upscale_factor
    W_out = W_in * upscale_factor

    x = x.view(B_in, C_out, upscale_factor, W_in)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B_in, C_out, W_out)
    return x


class SubPixel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor, use_convolution):
        super(SubPixel, self).__init__()
        self.upscale_factor = upscale_factor
        self.use_convolution = use_convolution
        self.padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels * upscale_factor ** 2,
                                    kernel_size=kernel_size,
                                    padding=self.padding)

    def forward(self, x):
        # Convolution in LR space
        if self.use_convolution:
            x = self.conv_layer(x)

        # Pixel shuffling
        x = pixel_shuffle(x, self.upscale_factor)
        return x


class SubPixel1D(nn.Module):
    def __init__(self, upscale_factor):
        super(SubPixel1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle_1d(x, self.upscale_factor)
