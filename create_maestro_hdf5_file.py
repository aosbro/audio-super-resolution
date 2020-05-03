import os
import numpy as np
import h5py
from scipy.io import wavfile
from mido import MidiFile, MidiTrack
from subprocess import call
import shutil

def sample_dataset(dataset_path, n_train, n_test, n_valid):
    """
    Selects randomly from the complete dataset a specified number of train, test and valid samples
    :param dataset_path: path to root directory containing the sub-directories where the .wav files are
    :param n_train: number of train samples to select
    :param n_test: number of test samples to select
    :param n_valid: number of valid samples to select
    :return: dictionary indexed by 'train', 'test' and 'valid' tp access a list of paths to selected files
    """
    wavfiles = []
    # Collect all .wav files recursively
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith('.wav'):
                wavfiles.append(os.path.join(root, name))

    n_total = n_train + n_test + n_valid
    selected_wavfiles = np.random.choice(wavfiles, size=n_total, replace=False)
    wavfiles_dict = {'train': selected_wavfiles[:n_train],
                     'test': selected_wavfiles[n_train: n_train + n_test],
                     'valid': selected_wavfiles[n_train + n_test:]}
    return wavfiles_dict


def compute_window_number(track_length, window_length=8192, overlap=0.5):
    """
    Computes the number of overlapping window for a specific track
    :param track_length: total number of samples in the track (scalar int)
    :param window_length: number of samples per window (scalar int)
    :param overlap: ratio of overlapping samples for consecutive samples (scalar int in [0, 1))
    :return: number of windows in the track
    """
    num = track_length - window_length
    den = window_length * (1 - overlap)
    return int(num // den + 2)


def cut_track_and_stack(track_path, window_length=8192, overlap=0.5):
    """
    Cuts a given track in overlapping windows and stacks them along a new axis
    :param track_path: path to .wav track to apply the function on
    :param window_length: number of samples per window (scalar int)
    :param overlap: ratio of overlapping samples for consecutive samples (scalar int in [0, 1))
    :return: processed track as a numpy array with dimension [window_number, 1, window_length]
    """
    # Load a single track
    _, track = wavfile.read(track_path)

    # Get rid of identical second channel
    track = (track[:, 0] / np.iinfo(np.int16).max).astype('float32')

    # Get number of windows and prepare empty array
    window_number = compute_window_number(track_length=track.shape[0])
    cut_track = np.zeros((window_number, window_length))

    # Cut the tracks in smaller windows
    for i in range(window_number):
        window_start = int(i * (1 - overlap) * window_length)
        window = track[window_start: window_start + window_length]

        # Check if last window needs padding
        if window.shape[0] != window_length:
            padding = window_length - window.shape[0]
            window = np.concatenate([window, np.zeros(padding)])
        cut_track[i] = window
    return cut_track.reshape((window_number, 1, window_length))


def create_hdf5_file(data_root, hdf5_path, window_length, n_train, n_test, n_valid):
    """
    Parses the root folder to find all .wav files and randomly select disjoint sets for the 'train', 'test'
    and 'valid' phases.
    :param data_root: path to root folder containing all data files and sub-folders (string)
    :param hdf5_path: path to location where to create the .h5 file (string)
    :param window_length: number of samples per window (scalar int)
    :param n_train: number of train tracks to select
    :param n_test: number of test tracks to select
    :param n_valid: number of valid tracks to select
    :return: None
    """
    wavfiles = sample_dataset(data_root, n_train=n_train, n_test=n_test, n_valid=n_valid)
    with h5py.File(hdf5_path, 'w') as hdf:
        # Create the groups inside the files
        for phase in ['train', 'test', 'valid']:
            hdf.create_group(name=phase)
            for i, file in enumerate(wavfiles[phase]):
                # Get a stacked track
                data = cut_track_and_stack(file, window_length=window_length)

                # Create the datasets for each group
                if i == 0:
                    hdf[phase].create_hdf5_file(name='original', data=data, maxshape=(None, 1, window_length))
                else:
                    # Resize and append dataset
                    hdf[phase]['original'].resize((hdf[phase]['original'].shape[0] + data.shape[0]), axis=0)
                    hdf[phase]['original'][-data.shape[0]:] = data


def modify_file(midi_filepath, midi_savepath, instrument=None, velocity=None, control=False, control_value=None):
    """
    Modifies a .midi file according to the given parameters. The new .midi file is saved in a user specified location.
    Information regarding the codes for instruments and control is available here:
    http://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html
    All information is coded on 8-bit therefore the values must lie in [0 - 127]
    :param midi_filepath: path to the input .midi file (string)
    :param midi_savepath: path to the outout .midi file (string)
    :param instrument: code of the new instrument (scalar int)
    :param velocity: value of the velocity to set for all notes (scalar int)
    :param control: boolean indicating if control should be modified (only pedal control with code 64 is used)
    :param control_value: new value of the control (set to 0 to remove the control) (scalar int)
    :return: None
    """
    # Load the original midi file
    mid_old = MidiFile(midi_filepath)

    # Create a new midi file
    mid_new = MidiFile()

    # Iterate over messages and apply transformation
    for i, track_old in enumerate(mid_old.tracks):
        track_new = MidiTrack()
        for j, msg_old in enumerate(track_old):
            # Append new message to new track
            msg_new = msg_old.copy()
            if instrument and msg_old.type == 'program_change':
                msg_new.program = instrument
            if velocity and msg_old.type == 'note_on':
                # Do not modify messages with velocity 0 as they correspond to 'note_off' (stops the note from playing)
                if msg_old.velocity != 0:
                    msg_new.velocity = velocity
            if control and msg_old.type == 'control_change':
                msg_new.value = control_value
            track_new.append(msg_new)

        # Append the new track to new file
        mid_new.tracks.append(track_new)
    mid_new.save(midi_savepath)


def convert_midi_to_wav(midi_filepath, wav_savepath, f_s):
    command = 'timidity {} -s {}Hz -Ow -o {}'.format(midi_filepath, wav_savepath, f_s)
    call(command.split())


def create_npy_file(data_root, temporary_directory_path, file_path, window_length, n_train, n_test, n_valid):
    # Randomly select the required number of tracks for each phase
    wavfiles = sample_dataset(data_root, n_train=n_train, n_test=n_test, n_valid=n_valid)

    # Create a directory to store the temporary files (.midi and .wav)
    if os.path.exists(temporary_directory_path):
        os.rmdir(temporary_directory_path)
        os.mkdir(temporary_directory_path)

    # Create the new .midi files
    input_midifiles = [wavfile.rsplit('.', 1)[0] + '.midi' for wavfile in wavfiles['train']]
    output_midifiles = [os.path.join(temporary_directory_path, os.path.split(wavfile)[-1].rsplit('.', 1)[0] + '.midi')
                 for wavfile in wavfiles['train']]
    [modify_file(input_midifile, output_midifile, velocity=64, control=True, control_value=0)
     for input_midifile, output_midifile in zip(input_midifiles, output_midifiles)]

if __name__ == '__main__':
    create_npy_file(data_root='/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0',
                    temporary_directory_path='data/maestro',
                    file_path=None,
                    window_length=None,
                    n_train=3,
                    n_test=1,
                    n_valid=1)
    # create_hdf5_file(data_root='/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0',
    #                  hdf5_path='/media/thomas/Samsung_T5/VITA/data/maestro_data.h5')
