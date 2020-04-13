# from torchaudio.transforms import Spectrogram, AmplitudeToDB
# import matplotlib.pyplot as plt
from datasets.datasets import DatasetBeethoven
# import numpy as np
from trainers.train_gan import get_gan_trainer
from utils.constants import *
from processing.post_processing import generate_high_resolution_sample

def main():

    # autoencoder_trainer = get_autoencoder(train_datapath=TRAIN_DATAPATH,
    #                                       test_datapath=TEST_DATAPATH,
    #                                       valid_datapath=VALID_DATAPATH,
    #                                       loadpath=AUTOENCODER_L2T_PATH,
    #                                       savepath=AUTOENCODER_L2T_PATH,
    #                                       batch_size=31)

    # generator_trainer = get_genarator_trainer(train_datapath=TRAIN_DATAPATH,
    #                                           test_datapath=TEST_DATAPATH,
    #                                           valid_datapath=VALID_DATAPATH,
    #                                           loadpath=GENERATOR_L2TF_PATH,
    #                                           savepath=GENERATOR_L2TF_PATH,
    #                                           batch_size=31)
    #
    # gan_trainer = get_gan_trainer(train_datapath=TRAIN_DATAPATH,
    #                               test_datapath=TEST_DATAPATH,
    #                               valid_datapath=VALID_DATAPATH,
    #                               loadpath=GAN_PATH,
    #                               savepath=GAN_PATH,
    #                               batch_size=31,
    #                               generator_path=None)
    #
    # generate_high_resolution_sample(gan_trainer, 11)
    dataset = DatasetBeethoven(datapath=TR)

if __name__ == '__main__':
    main()