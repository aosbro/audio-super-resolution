from torch import nn
import torch


def pixel_unshuffle(x, downscale_factor):
    B_in, C_in, H_in, W_in = x.size()

    C_out = C_in * downscale_factor ** 2
    H_out = int(H_in / downscale_factor)
    W_out = int(W_in / downscale_factor)

    x = x.view(B_in, C_in, H_out, downscale_factor, W_out, downscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B_in, C_out, H_out, W_out)
    return x


def pixel_shuffle(x, upscale_factor):
    B_in, C_in, H_in, W_in = x.size()

    C_out = C_in // upscale_factor ** 2
    H_out = H_in * upscale_factor
    W_out = W_in * upscale_factor

    x = x.view(B_in, C_out, upscale_factor, upscale_factor, H_in, W_in)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B_in, C_out, H_out, W_out)
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


class SuperPixel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downscale_factor, use_convolution):
        super(SuperPixel, self).__init__()
        self.downscale_factor = downscale_factor
        self.use_convolution = use_convolution
        self.padding = (kernel_size - 1) // 2
        self.conv_layer = nn.Conv2d(in_channels=in_channels * downscale_factor ** 2,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=self.padding)

    def forward(self, x):
        # Pixel un-shuffling
        x = pixel_unshuffle(x, self.downscale_factor)

        # Convolution in HR space
        if self.use_convolution:
            x = self.conv_layer(x)
        return x


def main():
    x = torch.randn(10, 3, 15, 15)
    sub_pixel = SubPixel(in_channels=3,
                         out_channels=16,
                         kernel_size=5,
                         upscale_factor=4,
                         use_convolution=True)

    super_pixel = SuperPixel(in_channels=16,
                             out_channels=3,
                             kernel_size=5,
                             downscale_factor=4,
                             use_convolution=True)

    loss_func = nn.MSELoss()

    # Initialize optimizer
    optimizer = torch.optim.Adam(sub_pixel.parameters())

    optimizer.param_groups.append({'params': super_pixel.parameters()})
    # print("optimizer params", optimizer.param_groups)
    [print(p_.shape) for p in optimizer.param_groups for p_ in p['params']]

    for i in range(1000):
        optimizer.zero_grad()
        y = sub_pixel(x)
        z = super_pixel(y)
        loss = loss_func(input=z, target=x)
        if i % 50 == 0:
            print(loss.item())

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()