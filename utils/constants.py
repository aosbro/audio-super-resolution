# Datasets constants
WINDOW_LENGTH = 8192
HANNING_WINDOW_LENGTH = 101
BATCH_SIZE = 10
NUM_WORKERS = 6
TRAIN_SHUFFLE = True
TEST_SHUFFLE = False
VALID_SHUFFLE = False
TRAIN_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_train_.npy'
TEST_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_test_.npy'

# Kernels constants


# Fully connected constants
FC1_OUTPUT_FEATURES = 64

# Layers and blocks constants
DOWNSCALE_FACTOR = 2
UPSCALE_FACTOR = 2
OUTPUT_KERNEL_SIZE = 27
KERNEL_SIZES = [3, 9, 27, 81]
CHANNEL_SIZES = [8, 24, 24, 8]
BOTTLENECK_CHANNELS = 8
DROPOUT_PROBABILITY = 0.2

# Models constants
N_BLOCKS = 8

# Optimizer constants
