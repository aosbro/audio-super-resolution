# # from trainers.train_autoencoder import *
from trainers.train_generator import *
import torch
from utils.constants import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write


def overlap_and_add_samples(batch, overlap=0.5, use_windowing=False):
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
        local_sample = batch[window_index].squeeze()
        if use_windowing:
            local_sample *= torch.from_numpy(np.hanning(WINDOW_LENGTH))
        full_sample[window_start: window_end] += local_sample
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
    B, W = batch_l.size()
    batch_l = batch_l.view(B, 1, W)
    batch_h = batch_h.view(B, 1, W)
    if trainer.is_autoencoder:
        autoencoder = trainer.autoencoder
        fake_batch, _ = autoencoder(batch_l)
    else:
        generator = trainer.generator
        fake_batch = generator(batch_l)
    full_sample_l = overlap_and_add_samples(batch_l, overlap=dataset.overlap, use_windowing=not dataset.use_windowing)
    full_sample_h = overlap_and_add_samples(batch_h, overlap=dataset.overlap, use_windowing=not dataset.use_windowing)
    full_sample_fake = overlap_and_add_samples(fake_batch.detach(), overlap=dataset.overlap,
                                               use_windowing=not dataset.use_windowing)

    plt.plot(full_sample_fake[4000:5000], label='fake')
    plt.plot(full_sample_h.numpy()[4000:5000], label='high')
    plt.legend()
    plt.show()

    full_sample_fake = full_sample_fake.numpy()
    # scaled_l = np.int16(full_sample_l.numpy() / np.max(np.abs(full_sample_l.numpy()) * 32767))
    write('./samples/gan_l_4.wav', 16000, full_sample_l.numpy())

    # scaled_h = np.int16(full_sample_h.numpy() / np.max(np.abs(full_sample_h.numpy()) * 32767))
    write('./samples/gan_h_4.wav', 16000, full_sample_h.numpy())

    # scaled_fake = np.int16(full_sample_fake / np.max(np.abs(full_sample_fake) * 32767))
    write('./samples/gan_fake_4.wav', 16000, full_sample_fake)

