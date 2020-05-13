# Data constants
WINDOW_LENGTH = 8192
HANNING_WINDOW_LENGTH = 101
BATCH_SIZE = 10
NUM_WORKERS = 6

# Maestro dataset
MAESTRO_PATH = '/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0'

# Beethoven dataset
TRAIN_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_train.npy'
TEST_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_test.npy'
VALID_DATAPATH = '/media/thomas/Samsung_T5/VITA/data/music/music_valid.npy'

TRAIN_DATAPATH_DRIVE = '/content/drive/My Drive/audio_data/music_train.npy'
TEST_DATAPATH_DRIVE = '/content/drive/My Drive/audio_data/music_test.npy'
VALID_DATAPATH_DRIVE = '/content/drive/My Drive/audio_data/music_valid.npy'

DOWNSCALE_FACTOR = 2
UPSCALE_FACTOR = 2

# Data-loaders constants
TRAIN_SHUFFLE = True
TEST_SHUFFLE = True
VALID_SHUFFLE = True

# Fully connected constants
FC1_OUTPUT_FEATURES = 64

# Layers and blocks constants
OUTPUT_KERNEL_SIZE = 27
KERNEL_SIZES = [3, 9, 27, 81]
CHANNEL_SIZES_MIN = [24, 24, 8, 8]
GENERATOR_CHANNEL_FACTOR_MAX = 3
AUTOENCODER_CHANNEL_FACTOR_MAX = 0
DISCRIMINATOR_CHANNEL_FACOTR_MAX = 3
GENERATOR_BOTTLENECK_REDUCTION_FACTOR = 4
AUTOENCODER_BOTTLENECK_REDUCTION_FACTOR = 4
DISCRIMINATOR_BOTTLENECK_RECDUCTION_FACTOR = 8
DROPOUT_PROBABILITY = 0.2

# Activations constants
LEAKY_RELU_SLOPE = 0.2

# Models constants
GENERATOR_USE_BOTTLENECK = True
AUTOENCODER_USE_BOTTLENECK = True
DISCRIMINATOR_USE_BOTTLENECK = True

# Models constants
N_BLOCKS_GENERATOR = 8
N_BLOCKS_DISCRIMINATOR = 7
N_BLOCKS_AUTOENCODER = 4

# Optimizer constants
LEARNING_RATE = 1e-4
AUTOENCODER_LEARNING_RATE = 1e-3

# Saving constants
# Auto-encoder trained with the L2 criterion in time domain
AUTOENCODER_L2T_PATH = './objects/autoencoder_trainer_l2t.tar'
AUTOENCODER_L2T_PATH_DRIVE = '/content/drive/My Drive/audio_objects/autoencoder_trainer_l2t.tar'

# Auto-encoder trained with the L2 criterion in time and frequency domain
AUTOENCODER_L2TF_PATH = './objects/autoencoder_trainer_l2tf.tar'
AUTOENCODER_L2TF_PATH_DRIVE = '/content/drive/My Drive/audio_objects/autoencoder_trainer_l2tf.tar'

GAN_PATH = './objects/gan_trainer.tar'
GAN_PATH_DRIVE = '/content/drive/My Drive/audio_objects/gan_trainer.tar'

GAN_EMBEDDING_PATH = './objects/gan_trainer_embedding.tar'
GAN_EMBEDDING_PATH_DRIVE = '/content/drive/My Drive/audio_objects/gan_trainer_embedding.tar'

GAN_EMBEDDING2_PATH = './objects/gan_trainer_embedding2.tar'
GAN_EMBEDDING2_PATH_DRIVE = '/content/drive/My Drive/audio_objects/gan_trainer_embedding2.tar'

GAN_EMBEDDING_W_PATH = './objects/gan_trainer_embedding_w.tar'
GAN_EMBEDDING_W_PATH_DRIVE = '/content/drive/My Drive/audio_objects/gan_trainer_embedding_w.tar'

GAN_EMBEDDING_ADV_PATH = './objects/gan_trainer_embedding_adv.tar'
GAN_EMBEDDING_ADV_PATH_DRIVE = '/content/drive/My Drive/audio_objects/gan_trainer_embedding_adv.tar'

# Generator trained with the L2 criterion in time domain
GENERATOR_L2T_PATH = './objects/generator_trainer_l2.tar'
GENERATOR_L2T_PATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_trainer_l2.tar'

# Generator trained with the L2 criterion in time and frequency domain
GENERATOR_L2TF_PATH = './objects/generator_trainer_l2tf.tar'
GENERATOR_L2TF_PATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_trainer_l2tf.tar'

# Generator trained with the L2 criterion in time and frequency domain
GENERATOR_L2TF_NO_WINDOW_PATH = './objects/generator_trainer_l2tf_no_window.tar'
GENERATOR_L2TF_NO_WINDOW_PATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_trainer_l2tf_no_window.tar'

GENERATOR_F_PATH = './objects/generator_f_trainer.tar'
GENERATOR_F_PATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_f_trainer.tar'

GENERATOR_NO_SKIP_PATH = './objects/generator_no_skip_trainer.tar'
GENERATOR_NO_SKIP_PATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_no_skip_trainer.tar'

# Generator trained with the L2 criterion in time and frequency (dB) domain
GENERATOR_L2TF_NO_WINDOW_DB_PATH = './objects/generator_trainer_l2tf_no_window_db.tar'
GENERATOR_L2TF_NO_WINDOW_DB_PATH_DRIVE = '/content/drive/My Drive/audio_objects/generator_trainer_l2tf_no_window_db.tar'

# ...
LAMBDA_ADVERSARIAL_MAX = 1e-3
LAMBDA_ADVERSARIAL_MIN = 1e-3
GENERATOR_CLIP_VALUE = 1000
TRAIN_BATCH_ITERATIONS = 100
TEST_BATCH_ITERATIONS = 50
VALID_BATCH_ITERATIONS = 50
ADVERSARIAL_ACTIVATION_EPOCH = 5
