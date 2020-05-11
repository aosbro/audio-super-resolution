import os
import torch
import shutil
from create_maestro_file import create_modified_midifile, convert_midi_to_wav, cut_track_and_stack
from processing.post_processing import overlap_and_add_samples
from scipy.io.wavfile import write
from utils.utils import get_generator
import argparse
from processing.post_processing import generate_single_track

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
