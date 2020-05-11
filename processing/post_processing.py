from utils.constants import *
import numpy as np
import os
import torch
import shutil
from create_maestro_file import create_modified_midifile, convert_midi_to_wav, cut_track_and_stack
from scipy.io.wavfile import write, read
from utils.utils import get_generator


def overlap_and_add_samples(batch, overlap=0.5, use_windowing=False):
    """
    Re-construct a full sample from its sub-parts using the OLA algorithm
    :param overlap: proportion of the overlap between contiguous signals
    :param batch: input signal previously split in overlapping windows torch tensor of shape [B, 1, WINDOW_LENGTH]
    :return: reconstructed sample (torch tensor)
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


def generate_single_track(original_midi_filepath, temporary_directory_path, transformations, generator_path, device):
    """
    Generates a part of single track using a pre-trained generator starting from the original .midi file. The
    transformation to apply to get the (input, target) pair of signals are specified in the transformations dictionary.
    Ideally the transformation should be the same as the one used to train the generator. The temporary directory is
    used to store the temporary .wav and .midi files. Caution: Its content will be removed before creating the new
    files. The generated track will be saved inside the temporary directory.
    :param original_midi_filepath: location of the original .midi file (string).
    :param temporary_directory_path: location of the temporary directory (string).
    :param transformations: dictionary of transformations to apply to the 'input' and 'target' tracks (dictionary).
    :param generator_path: location of the generator trainer to restore pre-trained weights (string).
    :param device: either 'cpu' or 'cuda' depending on hardware availability
    :return:None
    """
    # Load the pre-trained generator
    generator = get_generator(loadpath=generator_path, device=device)

    # Create a temporary directory
    if os.path.exists(temporary_directory_path):
        shutil.rmtree(temporary_directory_path)
    os.mkdir(temporary_directory_path)

    # Generate the pair of .midi files
    create_modified_midifile(midi_filepath=original_midi_filepath,
                             midi_savepath=os.path.join(temporary_directory_path, 'input.midi'),
                             **transformations['input'])
    create_modified_midifile(midi_filepath=original_midi_filepath,
                             midi_savepath=os.path.join(temporary_directory_path, 'target.midi'),
                             **transformations['target'])

    # Generate the pair of .wav files
    convert_midi_to_wav(midi_filepath=os.path.join(temporary_directory_path, 'input.midi'),
                        wav_savepath=os.path.join(temporary_directory_path, 'input.wav'))
    _, full_sample_input = read(os.path.join(temporary_directory_path, 'input.wav'))

    convert_midi_to_wav(midi_filepath=os.path.join(temporary_directory_path, 'target.midi'),
                        wav_savepath=os.path.join(temporary_directory_path, 'target.wav'))
    _, full_sample_target = read(os.path.join(temporary_directory_path, 'target.wav'))

    # Split the input file
    input_tensor, fs = cut_track_and_stack(track_path=os.path.join(temporary_directory_path, 'input.wav'))
    input_tensor = torch.from_numpy(input_tensor).float()[:160]

    # # Generate the output
    generated_tensor = torch.zeros_like(input_tensor)
    batch_size = 10
    with torch.no_grad():
        for batch in range(0, input_tensor.shape[0], batch_size):
            batch_generated = generator(input_tensor.narrow(0, batch, batch_size))
            generated_tensor[batch: batch + batch_generated.shape[0]] = batch_generated

    full_sample_generated = overlap_and_add_samples(batch=generated_tensor, overlap=0.5, use_windowing=True)
    write(os.path.join(temporary_directory_path, 'generated.wav'), fs, full_sample_generated.numpy())
    write(os.path.join(temporary_directory_path, 'input.wav'), fs, full_sample_input[:, 0])
    write(os.path.join(temporary_directory_path, 'target.wav'), fs, full_sample_target[:, 0])

# def test_reconstruction(index, dataset):
#     batch = [dataset.__getitem__(i + index * dataset.window_number) for i in range(dataset.window_number)]
#     batch_h, batch_l = map(list, zip(*batch))
#     batch_h, batch_l = torch.cat(batch_h), torch.cat(batch_l)
#     B, W = batch_l.size()
#     full_sample = overlap_and_add_samples(batch_l.view(B, 1, W))
#
#     # Define time for x-axis
#     T = full_sample.shape[0] / dataset.fs
#     t = np.linspace(start=0, stop=T, num=full_sample.shape[0])
#
#     # Plot
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].plot(t[:dataset.data[index].shape[0]], dataset.data[index])
#     axes[0].set_title('Original, high resolution', fontsize=16)
#     axes[0].set_xlabel('Time [s]', fontsize=14)
#     axes[0].set_ylabel('Amplitude', fontsize=14)
#     axes[1].plot(t, full_sample)
#     axes[1].set_title('Reconstruction, low resolution', fontsize=16)
#     axes[1].set_xlabel('Time [s]', fontsize=14)
#     axes[1].set_ylabel('Amplitude', fontsize=14)
#     plt.show()
#
#
# def generate_high_resolution_sample(trainer, index):
#     """
#     Generates a single track of the Beethoven dataset using a pre-trained generator.
#     :param trainer:
#     :param index:
#     :return:
#     """
#     dataset = trainer.test_loader.dataset
#     batch = [dataset.__getitem__(i + index * dataset.window_number) for i in range(dataset.window_number)]
#     batch_h, batch_l = map(list, zip(*batch))
#     batch_h, batch_l = torch.cat(batch_h), torch.cat(batch_l)
#     B, W = batch_l.size()
#     batch_l = batch_l.view(B, 1, W)
#     batch_h = batch_h.view(B, 1, W)
#     if trainer.is_autoencoder:
#         autoencoder = trainer.autoencoder
#         fake_batch, _ = autoencoder(batch_l)
#     else:
#         generator = trainer.generator
#         fake_batch = generator(batch_l)
#     full_sample_l = overlap_and_add_samples(batch_l, overlap=dataset.overlap, use_windowing=not dataset.use_windowing)
#     full_sample_h = overlap_and_add_samples(batch_h, overlap=dataset.overlap, use_windowing=not dataset.use_windowing)
#     full_sample_fake = overlap_and_add_samples(fake_batch.detach(), overlap=dataset.overlap,
#                                                use_windowing=not dataset.use_windowing)
#
#     plt.plot(full_sample_fake[4000:5000], label='fake')
#     plt.plot(full_sample_h.numpy()[4000:5000], label='high')
#     plt.legend()
#     plt.show()
#
#     full_sample_fake = full_sample_fake.numpy()
#     write('./samples/gan_l_4.wav', 16000, full_sample_l.numpy())
#     write('./samples/gan_h_4.wav', 16000, full_sample_h.numpy())
#     write('./samples/gan_fake_4.wav', 16000, full_sample_fake)

