import argparse


def parse_and_strore_contants():
    parser = argparse.ArgumentParser(description='Store all constants required for the models and training.')
    parser.add_argument('--window_length', default=8192, type=int, help='Number of samples per input tensor.')
    parser.add_argument('--overlap', default=0.5, type=float, help='Overlap between two contiguous windows.')
    parser.add_argument('--hanning_window_length', default=101, type=int, help='Length of the hanning window used to '
                                                                               'smooth the transition after padding.')
    parser.add_argument('--num_worker', default=2, type=int, help='Number of workers used by the data loaders.')
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