# from torchaudio.transforms import Spectrogram, AmplitudeToDB
# import matplotlib.pyplot as plt
from datasets.datasets import DatasetBeethoven
# import numpy as np
from trainers.train_gan import get_gan_trainer
from trainers.train_generator import get_genarator_trainer
from utils.constants import *
from utils.utils import get_consecutive_samples
from utils.metrics import snr

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
    #                                           loadpath=GENERATOR_L2TF_NO_WINDOW_PATH,
    #                                           savepath=GENERATOR_L2TF_NO_WINDOW_PATH,
    #                                           batch_size=31)

    gan_trainer = get_gan_trainer(train_datapath=TRAIN_DATAPATH,
                                  test_datapath=TEST_DATAPATH,
                                  valid_datapath=VALID_DATAPATH,
                                  loadpath=GAN_EMBEDDING_PATH,
                                  savepath=GAN_EMBEDDING_PATH,
                                  batch_size=31,
                                  generator_path=None,
                                  autoencoder_path=None)

    generate_high_resolution_sample(gan_trainer, 120)
    # dataset = DatasetBeethoven(datapath=TRAIN_DATAPATH,
    #                            ratio=4,
    #                            overlap=0.5,
    #                            use_windowing=False)
    # batch_h, batch_l = get_consecutive_samples(dataset, 0)
    # print(batch_l.shape)
    # print(snr(batch_l, batch_h))



if __name__ == '__main__':
    main()