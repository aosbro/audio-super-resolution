# Datasets constants
WINDOW_LENGTH = 8192
HANNING_WINDOW_LENGTH = 101
BATCH_SIZE = 10
NUM_WORKERS = 6

TRAIN_SHUFFLE = True
TEST_SHUFFLE = False
VALID_SHUFFLE = False

TRAIN_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_train.npy'
TEST_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_test.npy'
VALID_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_valid.npy'

TRAIN_DATAPATH_DRIVE = '/content/drive/My Drive/audio_data/music_train.npy'
TEST_DATAPATH_DRIVE = '/content/drive/My Drive/audio_data/music_test.npy'
VALID_DATAPATH_DRIVE = '/content/drive/My Drive/audio_data/music_valid.npy'

DOWNSCALE_FACTOR = 2
UPSCALE_FACTOR = 2

# Fully connected constants
FC1_OUTPUT_FEATURES = 64

# Layers and blocks constants
OUTPUT_KERNEL_SIZE = 27
KERNEL_SIZES = [3, 9, 27, 81]
CHANNEL_SIZES_MIN = [24, 24, 8, 8]
GENERATOR_CHANNEL_FACTOR_MAX = 0
AUTOENCODER_CHANNEL_FACTOR_MAX = 3
GENERATOR_BOTTLENECK_REDUCTION_FACTOR = 4
AUTOENCODER_BOTTLENECK_REDUCTION_FACOR = 1
DROPOUT_PROBABILITY = 0.2

# Activations constants
LEAKY_RELU_SLOPE = 0.2

# Models constants
GENERATOR_USE_BOTTLENECK = True

# Models constants
N_BLOCKS_GENERATOR = 8
N_BLOCKS_DISCRIMINATOR = 7
N_BLOCKS_AUTOENCODER = 4

# Optimizer constants
LEARNING_RATE = 1e-4

# Saving constants
AUTOENCODER_SAVEPATH = '../objects/autoencoder_trainer.txt'
AUTOENCODER_SAVEPATH_DRIVE = '/content/drive/My Drive/audio_objects/autoencoder_trainer.txt'

GAN_SAVEPATH = '../objects/gan_trainer.txt'
GAN_SAVEPATH_DRIVE = '/content/drive/My Drive/audio_objects/gan_trainer.txt'

GENERATOR_L2_SAVEPATH = '../objects/generator_trainer_l2.txt'
GENERATOR_L2_SAVEPATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_trainer_l2.txt'
