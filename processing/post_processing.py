# # from trainers.train_autoencoder import *
# from trainers.train_generator import *
import torch
from utils.constants import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write


def overlap_and_add_samples(batch, overlap=0.5):
    """
    Re-construct a full sample from its sub-parts using the OLA algorithm
    :param overlap: proportion of the overlap between contiguous signals
    :param batch: torch tensor of shape [B, 1, WINDOW_LENGTH]
    :return:
    """
    # Compute the size of the full sample
    N, _, single_sample_size = batch.size()
    full_sample_size = int(single_sample_size * (1 + (N - 1) * (1 - overlap)))

    # Initialize the full sample
    full_sample = torch.zeros(full_sample_size)

    for window_index in range(N):
        window_start = int(window_index * (1 - overlap) * WINDOW_LENGTH)
        window_end = window_start + WINDOW_LENGTH
        full_sample[window_start: window_end] += batch[window_index].squeeze()
    return full_sample


def test_reconstruction(index, dataset):
    batch = [dataset.__getitem__(i + index * dataset.window_number) for i in range(dataset.window_number)]
    batch_h, batch_l = map(list, zip(*batch))
    batch_h, batch_l = torch.cat(batch_h), torch.cat(batch_l)
    B, W = batch_l.size()
    full_sample = overlap_and_add_samples(batch_l.view(B, 1, W))

    # Define time for x-axis
    T = full_sample.shape[0] / dataset.fs
    t = np.linspace(start=0, stop=T, num=full_sample.shape[0])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t[:dataset.data[index].shape[0]], dataset.data[index])
    axes[0].set_title('Original, high resolution', fontsize=16)
    axes[0].set_xlabel('Time [s]', fontsize=14)
    axes[0].set_ylabel('Amplitude', fontsize=14)
    axes[1].plot(t, full_sample)
    axes[1].set_title('Reconstruction, low resolution', fontsize=16)
    axes[1].set_xlabel('Time [s]', fontsize=14)
    axes[1].set_ylabel('Amplitude', fontsize=14)
    plt.show()


def generate_high_resolution_sample(trainer, index):
    dataset = trainer.test_loader.dataset
    batch = [dataset.__getitem__(i + index * dataset.window_number) for i in range(dataset.window_number)]
    batch_h, batch_l = map(list, zip(*batch))
    batch_h, batch_l = torch.cat(batch_h), torch.cat(batch_l)
    print(batch_h.mean())
    B, W = batch_l.size()
    batch_l = batch_l.view(B, 1, W)
    batch_h = batch_h.view(B, 1, W)
    if trainer.is_autoencoder:
        print(batch_l.shape)
        autoencoder = trainer.autoencoder
        fake_batch, _ = autoencoder(batch_l)
    else:
        generator = trainer.generator
        fake_batch = generator(batch_l)
    full_sample_l = overlap_and_add_samples(batch_l)
    full_sample_h = overlap_and_add_samples(batch_h)
    full_sample_fake = overlap_and_add_samples(fake_batch.detach())

    plt.plot(full_sample_fake.numpy()[2000:4000])
    plt.plot(full_sample_h.numpy()[2000:4000])
    # plt.plot(batch_h[11].squeeze().numpy())
    # plt.plot(fake_batch[11].squeeze().detach().numpy())
    plt.show()

    scaled_l = np.int16(full_sample_l.numpy() / np.max(np.abs(full_sample_l.numpy()) * 32767))
    write('../samples/test_l.wav', 16000, full_sample_l.numpy())

    scaled_h = np.int16(full_sample_h.numpy() / np.max(np.abs(full_sample_h.numpy()) * 32767))
    write('../samples/test_h.wav', 16000, full_sample_h.numpy())

    scaled_fake = np.int16(full_sample_fake.numpy() / np.max(np.abs(full_sample_fake.numpy()) * 32767))
    write('../samples/test_fake_gen.wav', 16000, full_sample_fake.numpy())

#
# def main():
#
#     # autoencoder_trainer = get_autoencoder(train_datapath=TRAIN_DATAPATH,
#     #                                       test_datapath=TEST_DATAPATH,
#     #                                       valid_datapath=VALID_DATAPATH,
#     #                                       loadpath=AUTOENCODER_L2T_PATH,
#     #                                       savepath=AUTOENCODER_L2T_PATH,
#     #                                       batch_size=31)
#
#     generator_trainer = get_genarator_trainer(train_datapath=TRAIN_DATAPATH,
#                                               test_datapath=TEST_DATAPATH,
#                                               valid_datapath=VALID_DATAPATH,
#                                               loadpath=GENERATOR_L2TF_PATH,
#                                               savepath=GENERATOR_L2TF_PATH,
#                                               batch_size=31)
#
#     generate_high_resolution_sample(generator_trainer, 11)
#
#
# if __name__ == '__main__':
#     main()
