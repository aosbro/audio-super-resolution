import argparse


def get_general_args():
    parser = argparse.ArgumentParser(description='Store all constants required for the models and training.')
    # Data related constants
    parser.add_argument('--window_length', default=8192, type=int, help='Number of samples per input tensor.')
    parser.add_argument('--overlap', default=0.5, type=float, help='Overlap between two contiguous windows.')
    parser.add_argument('--hanning_window_length', default=101, type=int,
                        help='Length of the hanning window used to smooth the transition after padding.')
    parser.add_argument('--num_worker', default=2, type=int, help='Number of workers used by the data loaders.')

    # General architecture related constants
    parser.add_argument('--downscale_factor', default=2, type=int,
                        help='Factor by which the width is divided at each layer of the compressing part of the '
                             'discriminator, generator and auto-encoder.')
    parser.add_argument('--upscale_factor', default=2, type=int,
                        help='Factor by which the width is multiplied at each layer of the expanding part of the '
                             'discriminator, generator and auto-encoder.')
    parser.add_argument('--leaky_relu_slope', default=0.2, type=float,
                        help='Slope of the negative part of the leaky ReLu activation.')
    parser.add_argument('--dropout_probability', default=0.2, type=float,
                        help='Dropout probability of channes during training.')
    parser.add_argument('--kernel_sizes', nargs='+', default=[3, 9, 27, 81], type=int,
                        help='All models rely on multi-scale convolutions, therefore each convolution implements '
                             'sub-convolutions with different kernel sizes and number of filter. The features maps '
                             'resulting from each of the sub-convolution are concatenated along the channel dimension '
                             'afterwards. The padding is automatically taken care of. Individual kernel sizes are '
                             'expected to be separated by a space.')
    parser.add_argument('--channel_sizes_min', nargs='+', default=[24, 24, 8, 8], type=int,
                        help='All models rely on multi-scale convolutions, therefore each convolution implements '
                             'sub-convolutions with different kernel sizes and number of filter. The features maps '
                             'resulting from each of the sub-convolution are concatenated along the channel dimension '
                             'afterwards. This argument is the size of the channels for the first convolution. The '
                             'proportions of the channel sizes are kept for deepen layers, but will be scaled together '
                             ' along the depth of the model.')

    # Auto-encoder's architecture related constants
    parser.add_argument('--autoencoder_n_block', default=4, type=int,
                        help='Number of blocks in the compressing part as well as in the expanding part of the '
                             'auto-encoder.')
    parser.add_argument('--autoencoder_channel_factor_max', default=0, type=int,
                        help='Number of DownBlocks that will have an output number of channel (C_out) twice as large '
                             'as the input number of channel (C_in). Blocks deeper that this number will have'
                             'C_out = C_in. This parameter is used to regulate the complexity of the model.')
    parser.add_argument('--autoencoder_use_bottleneck', default=True, type=bool,
                        help='Flag indicating if the convolutions should implement a scaling of the input channel '
                             'dimension (C_in) to reduce the computational complexity. This is implemented by first '
                             'applying a convolution with a relatively low number of filters (C_temp) of width 1. This '
                             'is followed by a second convolution to get the desired number of output channels'
                             '(https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14806/14311).')
    parser.add_argument('--autoencoder_bottleneck_reduction_factor', default=4, type=int,
                        help='Factor by which the number of input channels should be reduced by a first convolution '
                             'with kernels of width 1.')
    parser.add_argument('--autoencoder_output_kernel_size', default=27, type=int,
                        help='Width of the kernel for the last convolution of the autoencoder.')

    # Generator's architecture related constants
    parser.add_argument('--generator_n_block', default=8, type=int,
                        help='Number of blocks in the compressing part as well as in the expanding part of the '
                             'generator.')
    parser.add_argument('--generator_channel_factor_max', default=4, type=int,
                        help='Number of DownBlocks that will have an output number of channel (C_out) twice as large '
                             'as the input number of channel (C_in). Blocks deeper that this number will have'
                             'C_out = C_in. This parameter is used to regulate the complexity of the model.')
    parser.add_argument('--generator_use_bottleneck', default=True, type=bool,
                        help='Flag indicating if the convolutions should implement a scaling of the input channel '
                             'dimension (C_in) to reduce the computational complexity. This is implemented by first '
                             'applying a convolution with a relatively low number of filters (C_temp) of width 1. This '
                             'is followed by a second convolution to get the desired number of output channels'
                             '(https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14806/14311).')
    parser.add_argument('--generator_bottleneck_reduction_factor', default=16, type=int,
                        help='Factor by which the number of input channels should be reduced by a first convolution '
                             'with kernels of width 1.')
    parser.add_argument('--generator_output_kernel_size', default=27, type=int,
                        help='Width of the kernel for the last convolution of the generator.')

    # Discriminator's architecture related constants
    parser.add_argument('--discriminator_n_block', default=7, type=int,
                        help='Number of blocks in the compressing part of the discriminator.')
    parser.add_argument('--discriminator_channel_factor_max', default=0, type=int,
                        help='Number of DiscriminatorBlocks that will have an output number of channel (C_out) twice '
                             'as large as the input number of channel (C_in). Blocks deeper that this number will have'
                             'C_out = C_in. This parameter is used to regulate the complexity of the model.')
    parser.add_argument('--discriminator_use_bottleneck', default=True, type=bool,
                        help='Flag indicating if the convolutions should implement a scaling of the input channel '
                             'dimension (C_in) to reduce the computational complexity. This is implemented by first '
                             'applying a convolution with a relatively low number of filters (C_temp) of width 1. This '
                             'is followed by a second convolution to get the desired number of output channels'
                             '(https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14806/14311).')
    parser.add_argument('--discriminator_bottleneck_reduction_factor', default=16, type=int,
                        help='Factor by which the number of input channels should be reduced by a first convolution '
                             'with kernels of width 1.')
    parser.add_argument('--fc1_output_features', default=64, type=int,
                        help='Number of output features in first linear layer of the discriminator.')
    parser.add_argument('--use_layer_norm', default=True, type=bool,
                        help='Flag indicating whether to use layer normalization or batch normalization as the latter '
                             'is incoherent with the wgan-gp framework.')

    # General training related constants
    parser.add_argument('--train_batches_per_epoch', default=100, type=int,
                        help='Number of batches inside a training pseudo-epoch. This allows finer control over the '
                             'stopping criterion when working with large datasets.')
    parser.add_argument('--test_batches_per_epoch', default=50, type=int,
                        help='Number of batches inside a test pseudo-epoch. This allows for a faster but more'
                             ' stochastic evaluation.')
    parser.add_argument('--valid_batches_per_epoch', default=50, type=int,
                        help='Number of batches inside a validation pseudo-epoch. This allows for a faster but more'
                             ' stochastic evaluation.')
    args = parser.parse_args()
    return args

