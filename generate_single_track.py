import argparse
from processing.post_processing import generate_single_track
import torch
import os


def main():
    # midi_filepath = '/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0/2017/MIDI-Unprocessed_083_PIANO083_MID--AUDIO' \
    #                 '-split_07-09-17_Piano-e_2_-06_wav--3.midi'
    # temporary_directory_path = 'data/temp'
    # generator_path = 'objects/generator_trainer_time2.tar'
    # transformations = {'input': {'instrument': 4, 'velocity': None, 'control': None, 'control_value': None},
    #                    'target': {'instrument': 0, 'velocity': None, 'control': None, 'control_value': None}}

    parser = argparse.ArgumentParser(description='Generates a track from an input .midi file.')
    parser.add_argument('--original_midi', type=str, help='Original .midi file to base the input and target on.')
    parser.add_argument('--temp_dir', default='data/temp', type=str, help='Path to temporary directory.')
    parser.add_argument('--generator_path', type=str, help='Path to a generator trainer to load pre-trained weights.')
    parser.add_argument('--input_instrument', default=4, type=int, help='Input instrument, default is electric piano.')
    parser.add_argument('--input_velocity', default=None, type=int, help='Velocity corresponds to the volume at which a'
                                                                         ' given key is played. When set to a value in '
                                                                         '[0, 127] this information is lost.')
    parser.add_argument('--input_control', default=None, type=int, help='Control corresponds to effects (sustain/pedal)'
                                                                        'applied on the signal. The only control in the'
                                                                        ' Maestro dataset is the pedal which is encoded'
                                                                        ' on value 64. A specific control can be set to'
                                                                        ' a new control value using control and '
                                                                        'control_value arguments.')
    parser.add_argument('--input_control_value', default=None, type=int, help='The control value is typically set to '
                                                                              'zero to remove a specific effect '
                                                                              'selected with the control argument.')
    parser.add_argument('--target_instrument', default=0, type=int, help='Input instrument, default is classic piano.')
    parser.add_argument('--target_velocity', default=None, type=int, help='Velocity corresponds to the volume at which '
                                                                          'a given key is played. When set to a value '
                                                                          'in[0, 127] this information is lost.')
    parser.add_argument('--target_control', default=None, type=int, help='Control corresponds to effects '
                                                                         '(sustain/pedal) applied on the signal. The '
                                                                         'only control in the Maestro dataset is the '
                                                                         'pedal which is encoded on value 64. A '
                                                                         'specific control can be set to a new control '
                                                                         'value using control and  arguments.')
    parser.add_argument('--target_control_value', default=None, type=int, help='The control value is typically set to '
                                                                               'zero to remove a specific effect '
                                                                               'selected with the control argument.')
    args = parser.parse_args()

    # Prepare the transformations in adequate format
    transformations = {'input': {'instrument': args.input_instrument, 'velocity': args.input_velocity,
                                 'control': args.input_control, 'control_value': args.input_control_value},
                       'target': {'instrument': args.target_instrument, 'velocity': args.target_velocity,
                                  'control': args.target_control, 'control_value': args.target_control_value}}

    # Set the device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    generate_single_track(original_midi_filepath='/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0/2017/MIDI'
                                                 '-Unprocessed_083_PIANO083_MID--AUDIO-split_07-09-17_Piano-e_2_'
                                                 '-06_wav--3.midi',
                          temporary_directory_path=args.temp_dir,
                          transformations=transformations,
                          generator_path='objects/gan_trainer.tar',
                          device='cpu')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
