from models.generator import Generator
from utils.constants import *
import os
import torch
import shutil
from create_maestro_file import create_modified_midifile, convert_midi_to_wav, cut_track_and_stack
from processing.post_processing import overlap_and_add_samples
from scipy.io.wavfile import write


def get_generator(loadpath, device):
    """
    Returns a pre-trained generator.
    :param loadpath: location of the generator trainer (string).
    :param device: either 'cpu' or 'cuda' depending on hardware availability (string).
    :return: pre-trained generator (nn.Module).
    """
    # Instantiate a new generator with identical architecture
    generator = Generator(kernel_sizes=KERNEL_SIZES,
                          channel_sizes_min=CHANNEL_SIZES_MIN,
                          p=DROPOUT_PROBABILITY,
                          n_blocks=N_BLOCKS_GENERATOR).to(device)

    # Restore pre-trained weights
    checkpoint = torch.load(loadpath, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    return generator.eval()


def main():
    midi_filepath = '/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0/2017/MIDI-Unprocessed_083_PIANO083_MID--AUDIO' \
                    '-split_07-09-17_Piano-e_2_-06_wav--3.midi'
    temporary_directory_path = 'data/temp'

    transformations = {'input': {'instrument': 4, 'velocity': None, 'control': None, 'control_value': None},
                       'target': {'instrument': 0, 'velocity': None, 'control': None, 'control_value': None}}

    # Load the pre-trained generator
    generator = get_generator(loadpath='objects/generator_trainer_freq.tar', device='cpu')

    # Create a temporary directory
    if os.path.exists(temporary_directory_path):
        shutil.rmtree(temporary_directory_path)
    os.mkdir(temporary_directory_path)

    # Generate the pair of .midi files
    create_modified_midifile(midi_filepath=midi_filepath,
                             midi_savepath=os.path.join(temporary_directory_path, 'input.midi'),
                             **transformations['input'])
    create_modified_midifile(midi_filepath=midi_filepath,
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


if __name__ == '__main__':
    main()
