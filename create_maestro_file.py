from processing.pre_processing import sample_dataset, create_modified_midifile, create_hdf5_file, create_npy_files
from utils.utils import prepare_transformations
import argparse
import shutil
import os


def get_dataset_creation_args():
    """
    Parses the arguments related to the creation of the dataset if provided by the user, otherwise uses default
    values.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Creates the files on which training can occur from the Maestro dataset. Can either create a single'
                    '.hdf5 file that contains the complete data for all phases (train, test and validation) or three '
                    'separate .npy files (one for each phase).')
    parser.add_argument('--use_npy', default=True, type=bool,
                        help='Flag indicating if the data is stored as multiple .npy files or a single .hdf5 file. This'
                             'data format is not recommended as it required the data to fit entirely in RAM.')
    parser.add_argument('--hdf5_savepath', default='data/maestro'
                                                   '.hdf5', type=str,
                        help='Location of the .hdf5 file to create if this data format is selected')
    parser.add_argument('--train_npy_filepath', default='data/train.npy', type=str,
                        help='Location of the train .npy file if this data format is selected.')
    parser.add_argument('--test_npy_filepath', default='data/test.npy', type=str,
                        help='Location of the test .npy file if this data format is selected.')
    parser.add_argument('--valid_npy_filepath', default='data/valid.npy', type=str,
                        help='Location of the valid .npy file if this data format is selected.')
    parser.add_argument('--data_root', default='/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0', type=str,
                        help='Root directory of the Maestro dataset.')
    parser.add_argument('--temporary_directory', default='data/temp', type=str,
                        help='Location of a temporary directory to store temporary files. If is does not exists it will'
                             'be created. If it already exists its content will be erased. After the creation of the '
                             'dataset the temporary folder and its content will be deleted.')
    parser.add_argument('--remove_temporary_directory', default=True, type=bool,
                        help='Flag indicating if the temporary directory must be deleted after the dataset creation.')
    parser.add_argument('--used_tracks_file', default='data/used_tracks.txt', type=str,
                        help='Location of a text file to store the names of the tracks that have been used.')
    parser.add_argument('--n_train', default=5, type=int,
                        help='Number of tracks to base the train dataset on. The average length of a track is over 10 '
                             'minutes and can go up to 30 minutes. The tracks are sampled at 44.1 kHz or 48 kHz this '
                             'information is stored as a meta-data in each track.')
    parser.add_argument('--n_test', default=1, type=int,
                        help='Number of tracks to base the test dataset on. The average length of a track is over 10 '
                             'minutes and can go up to 30 minutes. The tracks are sampled at 44.1 kHz or 48 kHz this '
                             'information is stored as a meta-data in each track.')
    parser.add_argument('--n_valid', default=1, type=int,
                        help='Number of tracks to base the validation dataset on. The average length of a track is over'
                             ' 10 minutes and can go up to 30 minutes. The tracks are sampled at 44.1 kHz or 48 kHz '
                             'this information is stored as a meta-data in each track.')
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
    return args


def create_maestro_dataset(dataset_args):
    """
    Parses the data_root folder to collect all .midi files. From all the files a random selection is done to get the
    specified numbers for each phase ('train', 'test', 'valid'). The .midi files are then modified according to the
    transformations dictionary. From the new .midi files, the .wav files are generated by calling Timidity++. Finally,
    the .wav files are loaded as numpy arrays and cut in overlapping windows of size window_length=8192. These numpy
    arrays are stored in a hdf5 file.
    """
    # Randomly select the required number of tracks for each phase
    midifiles = sample_dataset(dataset_path=dataset_args.data_root,
                               n_train=dataset_args.n_train,
                               n_test=dataset_args.n_test,
                               n_valid=dataset_args.n_valid)

    # Save the name of used tracks
    with open(dataset_args.used_tracks_file, 'w') as f:
        print(midifiles, file=f)

    # Create a dictionary to store files location
    file_dict = {phase: {status: [] for status in ['input', 'target']} for phase in ['train', 'test', 'valid']}

    # Get the transformations to apply to input and target signals
    transformations = prepare_transformations(dataset_args)

    # Create a directory to store the temporary files (.midi and .wav)
    if os.path.exists(dataset_args.temporary_directory):
        shutil.rmtree(dataset_args.temporary_directory)
    os.mkdir(dataset_args.temporary_directory)

    # Create the new .midi files
    for phase in ['train', 'test', 'valid']:
        # Create sub-directories for each phase
        phase_directory = os.path.join(dataset_args.temporary_directory, phase)
        os.mkdir(phase_directory)
        for status in ['input', 'target']:
            # Create sub-directories for each status
            status_directory = os.path.join(phase_directory, status)
            os.mkdir(status_directory)

            # Process the .midi files and save as .wav
            output_midifiles = [os.path.join(status_directory, os.path.split(midifile)[-1])
                                for midifile in midifiles[phase]]

            # Add files location to dict
            file_dict[phase][status] = output_midifiles

            [create_modified_midifile(input_midifile, output_midifile, **transformations[status])
             for input_midifile, output_midifile in zip(midifiles[phase], output_midifiles)]

    # Loop over all selected files and add them to the dataset
    if dataset_args.use_npy:
        file_savepath = {'train': dataset_args.train_npy_filepath,
                         'test': dataset_args.test_npy_filepath,
                         'valid': dataset_args.valid_npy_filepath}
        create_npy_files(file_dict, dataset_args.temporary_directory, savepath=file_savepath)
    else:
        create_hdf5_file(file_dict, dataset_args.temporary_directory, hdf5_path=dataset_args.hdf5_savepath)

    # Remove temporary files
    if dataset_args.remove_temporary_directory:
        shutil.rmtree(dataset_args.temporary_directory)


if __name__ == '__main__':
    # Get the parameters related to the dataset creation
    dataset_args = get_dataset_creation_args()

    # Create the dataset
    create_maestro_dataset(dataset_args)
