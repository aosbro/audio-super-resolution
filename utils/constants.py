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
N_BLOCKS_AUTOENCODER = 4

# Optimizer constants
LEARNING_RATE = 1e-3

# Saving constants
AUTOENCODER_SAVEPATH = '../objects/autoencoder_trainer.txt'
AUTOENCODER_SAVEPATH_DRIVE = '/content/drive/My Drive/audio_objects/autoencoder_trainer.txt'