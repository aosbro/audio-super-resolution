import os
import torch
import shutil
from create_maestro_file import create_modified_midifile, convert_midi_to_wav, cut_track_and_stack
from processing.post_processing import overlap_and_add_samples
from scipy.io.wavfile import write
from utils.utils import get_generator
import argparse


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
    convert_midi_to_wav(midi_filepath=os.path.join(temporary_directory_path, 'target.midi'),
                        wav_savepath=os.path.join(temporary_directory_path, 'target.wav'))

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


def main():
    midi_filepath = '/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0/2017/MIDI-Unprocessed_083_PIANO083_MID--AUDIO' \
                    '-split_07-09-17_Piano-e_2_-06_wav--3.midi'
    temporary_directory_path = 'data/temp'
    generator_path = 'objects/generator_trainer_time2.tar'
    transformations = {'input': {'instrument': 4, 'velocity': None, 'control': None, 'control_value': None},
                       'target': {'instrument': 0, 'velocity': None, 'control': None, 'control_value': None}}

    generate_single_track(midi_filepath, temporary_directory_path, transformations, generator_path, 'cpu')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Train a deep clustering autoencoder for a given number of epochs')
    # parser.add_argument('epochs', metavar='epochs', type=int, help='Number of epochs to train the encoder_model on')
    # parser.add_argument('data_path', metavar='data_path', type=str, help='Path to data directory')
    # parser.add_argument('filepath', metavar='filepath', type=str, help='Path to hdf5 file')
    # parser.add_argument('models_path', metavar='models_path', type=str, help='Path to models')
    # parser.add_argument('dec_name', metavar='dec_name', type=str, help='Name of the DEC encoder_model')
    # args = parser.parse_args()
    #
    # main(args)
    main()
