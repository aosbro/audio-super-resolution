from processing.pre_processing import convert_midi_to_wav, cut_track_and_stack
from create_maestro_file import create_modified_midifile
from scipy.io.wavfile import write, read
from utils.utils import get_generator
import numpy as np
import shutil
import torch
import os


def overlap_and_add_samples(batch, general_args, use_windowing=True):
    """
    Re-construct a full sample from its sub-parts using the OLA algorithm.
    :param batch: input signal previously split in overlapping windows torch tensor of shape [B, 1, WINDOW_LENGTH].
    :return: reconstructed sample (torch tensor).
    """
    # Compute the size of the full sample
    N, _, single_sample_size = batch.size()
    full_sample_size = int(single_sample_size * (1 + (N - 1) * (1 - general_args.overlap)))

    # Initialize the full sample
    full_sample = torch.zeros(full_sample_size)

    for window_index in range(N):
        window_start = int(window_index * (1 - general_args.overlap) * general_args.window_length)
        window_end = window_start + general_args.window_length
        local_sample = batch[window_index].squeeze()
        if use_windowing:
            local_sample *= torch.from_numpy(np.hanning(general_args.window_length))
        full_sample[window_start: window_end] += local_sample
    return full_sample


def generate_single_track(original_midi_filepath, temporary_directory_path, transformations, generator_path,
                          device, general_args, track_args):
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
    :param general_args: argument parser that contains the arguments that are independent to the script being executed.
    :param track_args: argument parser that contains the arguments related to the track generation.
    :return:None
    """
    # Load the pre-trained generator
    generator = get_generator(loadpath=generator_path, device=device, general_args=general_args)

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
    input_tensor = torch.from_numpy(input_tensor).float()[:track_args.n_samples]

    # # Generate the output
    generated_tensor = torch.zeros_like(input_tensor)
    batch_size = 10
    with torch.no_grad():
        for batch in range(0, input_tensor.shape[0], batch_size):
            batch_generated = generator(input_tensor.narrow(0, batch, batch_size))
            generated_tensor[batch: batch + batch_generated.shape[0]] = batch_generated

    full_sample_generated = overlap_and_add_samples(batch=generated_tensor, general_args=general_args)
    write(os.path.join(temporary_directory_path, 'generated.wav'), fs, full_sample_generated.numpy())
    print(full_sample_input.shape)
    write(os.path.join(temporary_directory_path, 'input.wav'), fs, full_sample_input[:, 0])
    write(os.path.join(temporary_directory_path, 'target.wav'), fs, full_sample_target[:, 0])



