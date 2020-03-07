from blocks.down_block import *
from blocks.up_block import *


class Generator(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes, bottleneck_channels, p):
        super(Generator, self).__init__()
        # Encoder
        self.down_block_1 = DownBlock(in_channels=1, kernel_sizes=kernel_sizes, channel_sizes=channel_sizes,
                                      bottleneck_channels=bottleneck_channels)
        self.down_block_2 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_3 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_4 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_5 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_6 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_7 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        self.down_block_8 = DownBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                      channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels)
        # Decoder
        self.up_block_1 = UpBlock(in_channels=2 * sum(channel_sizes), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_2 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_3 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_4 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_5 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_6 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_7 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)
        self.up_block_8 = UpBlock(in_channels=int((5/2) * sum(channel_sizes)), kernel_sizes=kernel_sizes,
                                  channel_sizes=channel_sizes, bottleneck_channels=bottleneck_channels, p=p)

        # Output convolution
        kernel_size = 27
        padding = (kernel_size - 1) // 2
        self.output_conv = nn.Conv1d(in_channels=int((1/2) * sum(channel_sizes)), out_channels=1,
                                     kernel_size=kernel_size, padding=padding)

    def forward(self, x_l):
        # Encoder
        d1 = self.down_block_1(x_l)
        d2 = self.down_block_2(d1)
        d3 = self.down_block_3(d2)
        d4 = self.down_block_4(d3)
        d5 = self.down_block_5(d4)
        d6 = self.down_block_6(d5)
        d7 = self.down_block_7(d6)
        d8 = self.down_block_8(d7)

        # Decoder
        u1 = self.up_block_1(d8, d7)
        u2 = self.up_block_2(u1, d6)
        u3 = self.up_block_3(u2, d5)
        u4 = self.up_block_4(u3, d4)
        u5 = self.up_block_5(u4, d3)
        u6 = self.up_block_6(u5, d2)
        u7 = self.up_block_7(u6, d1)
        u8 = self.up_block_8(u7, None)
        return self.output_conv(u8) + x_l


# def main():
#     x_l = torch.randn(10, 1, 8192)
#     kernel_sizes = [3, 9, 27, 81]
#     channel_sizes = 4 * [16]
#     bottleneck_channels = 8
#     p = 0.2
#     G = Generator(kernel_sizes, channel_sizes, bottleneck_channels, p)
#     print(G(x_l).shape)
#
# if __name__ == '__main__':
#     main()
