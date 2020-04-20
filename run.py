# import matplotlib.pyplot as plt
from datasets.datasets import DatasetBeethoven
# import numpy as np
from trainers.train_gan import get_gan_trainer
from trainers.train_autoencoder import get_autoencoder_trainer
from trainers.train_generator import get_genarator_trainer
from utils.constants import *
from utils.utils import plot_losses
from utils.utils import get_consecutive_samples
import torch
from utils.metrics import snr, lsd
from processing.post_processing import generate_high_resolution_sample


def main():
    # autoencoder_trainer = get_autoencoder_trainer(train_datapath=TRAIN_DATAPATH,
    #                                               test_datapath=TEST_DATAPATH,
    #                                               valid_datapath=VALID_DATAPATH,
    #                                               loadpath=AUTOENCODER_L2TF_PATH,
    #                                               savepath=AUTOENCODER_L2TF_PATH,
    #                                               batch_size=BATCH_SIZE)

    # generator_trainer = get_genarator_trainer(train_datapath=TRAIN_DATAPATH,
    #                                           test_datapath=TEST_DATAPATH,
    #                                           valid_datapath=VALID_DATAPATH,
    #                                           loadpath=GENERATOR_L2TF_PATH,
    #                                           savepath=GENERATOR_L2TF_PATH,
    #                                           batch_size=31)

    gan_trainer = get_gan_trainer(train_datapath=TRAIN_DATAPATH,
                                  test_datapath=TEST_DATAPATH,
                                  valid_datapath=VALID_DATAPATH,
                                  loadpath=GAN_EMBEDDING2_PATH,
                                  savepath=GAN_EMBEDDING2_PATH,
                                  batch_size=31,
                                  generator_path=None,
                                  autoencoder_path=None)

    gan_trainer.plot_reconstruction_frequency_domain(index=10,
                                                     model=gan_trainer.generator,
                                                     savepath='./figures/generator_specgram.png')

    # plot_losses(losses=gan_trainer.train_losses,
                # names=['time_l2', 'freq_l2', 'autoencoder_l2'],
                # is_training=True,
                # savepath='./figures/gan_train_losses2.png')

    # plot_losses(losses=gan_trainer.train_losses,
    #             names=['generator_adversarial', 'discriminator_adversarial'],
    #             is_training=True,
    #             savepath='./figures/gan_train_losses.png')

    generate_high_resolution_sample(gan_trainer, 40)

    # dataset = DatasetBeethoven(datapath=TRAIN_DATAPATH,
    #                            ratio=4,
    #                            overlap=0.5,
    #                            use_windowing=False)
    # batch_h, batch_l = get_consecutive_samples(dataset, 0)
    # batch_h_freq = torch.stft(batch_h, n_fft=256, normalized=True)

    # lsd(x=batch_h, x_ref=batch_l)


if __name__ == '__main__':
    main()
