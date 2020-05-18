from scipy.signal import decimate, resample
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
from scipy.io import wavfile
from mido import MidiFile, MidiTrack
from subprocess import call


def downsample(x, downscale_factor):
    """
    Down-samples the original signal by a factor upscale_factor after applying an anti-aliasing filter
    :param x: Original signal (numpy array)
    :param downscale_factor: Integer down-sampling factor
    :return: Down-sampled signal (numpy array)
    """
    x_ds = decimate(x, downscale_factor, axis=0, zero_phase=True)
    return x_ds.astype(x.dtype)


def upsample(x, upscale_factor):
    """
    Up-samples the original signal by a factor downscale_factor
    :param x: Previously down-sampled signal (numpy array)
    :param upscale_factor: Integer down-sampling factor
    :return: Up-sampled signal (numpy array)
    """
    x_us = resample(x, len(x) * upscale_factor, axis=0)
    return x_us.astype(x.dtype)


def compute_spectrogram(x):
    """
    Computes the spectrogram of the x signal
    :param x: Time domain signal (numpy array)
    :return: Spectrogram (numpy array)
    """
    X = librosa.stft(x)
    X_db = librosa.amplitude_to_db(abs(X))
    return X_db


def plot_spectrograms(x_h, x_l, fs):
    """
    Displays the high and low quality signals' spectrogram
    :param x_h: Original signal (numpy array)
    :param x_l: Previously down-sampled signal (numpy array)
    :param fs: Sampling frequency
    :return: None
    """
    # Getting the high quality spectrogram
    X_h = librosa.stft(x_h)
    X_h_db = librosa.amplitude_to_db(abs(X_h))
    X_h_db -= np.max(X_h_db)

    # Getting the low quality spectrogram
    X_l = librosa.stft(x_l)
    X_l_db = librosa.amplitude_to_db(abs(X_l))
    X_l_db -= np.max(X_l_db)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    librosa.display.specshow(X_h_db, sr=fs, x_axis='time', y_axis='hz', ax=axes[0])
    axes[0].set_title('High quality, spectrogram', fontsize=16)
    librosa.display.specshow(X_l_db, sr=fs, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Low quality, spectrogram', fontsize=16)
    plt.show()


def sample_dataset(dataset_path, n_train, n_test, n_valid):
    """
    Selects randomly from the complete dataset a specified number of train, test and valid samples.
    :param dataset_path: path to root directory containing the sub-directories where the .wav files are (string).
    :param n_train: number of train samples to select (scalar int).
    :param n_test: number of test samples to select (scalar int).
    :param n_valid: number of valid samples to select (scalar int).
    :return: dictionary indexed by 'train', 'test' and 'valid' to access a list of paths to selected files
    """
    midifiles = []
    # Collect all .wav files recursively
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith('.midi'):
                midifiles.append(os.path.join(root, name))

    n_total = n_train + n_test + n_valid
    selected_wavfiles = np.random.choice(midifiles, size=n_total, replace=False)
    wavfiles_dict = {'train': selected_wavfiles[:n_train],
                     'test': selected_wavfiles[n_train: n_train + n_test],
                     'valid': selected_wavfiles[n_train + n_test:]}
    return wavfiles_dict


def compute_window_number(track_length, window_length=8192, overlap=0.5):
    """
    Computes the number of overlapping window for a specific track.
    :param track_length: total number of samples in the track (scalar int).
    :param window_length: number of samples per window (scalar int).
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
    :return: processed track as a numpy array with dimension [window_number, 1, window_length], sampling frequency
    """
    # Load a single track
    fs, track = wavfile.read(track_path)

    # Select left channel (do not use the right channel (track[:, 1]) as Timidity++ introduces distorsions in it)
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
    return cut_track.reshape((window_number, 1, window_length)), fs


def create_hdf5_file(file_dict, temporary_directory_path, hdf5_path, window_length=8192):
    """
    Parses the root folder to find all .wav files and randomly select disjoint sets for the 'train', 'test'
    and 'valid' phases.
    :param hdf5_path: path to location where to create the .h5 file (string)
    :param window_length: number of samples per window (scalar int)
    :param n_train: number of train tracks to select
    :param n_test: number of test tracks to select
    :param n_valid: number of valid tracks to select (scalar int).
    :return: None
    """
    with h5py.File(hdf5_path, 'w') as hdf:
        # Create the groups inside the files
        for phase in ['train', 'test', 'valid']:
            hdf.create_group(name=phase)
            phase_directory = os.path.join(temporary_directory_path, phase)
            status_directories = {status: os.path.join(phase_directory, status) for status in ['input', 'target']}
            for i, (input_midifile, target_midifile) in enumerate(zip(file_dict[phase]['input'],
                                                                      file_dict[phase]['target'])):
                # Create the new wavfiles
                input_wav_savepath = os.path.join(status_directories['input'],
                                                     os.path.split(input_midifile)[-1].rsplit('.', 1)[0] + '.wav')
                target_wav_savepath = os.path.join(status_directories['target'],
                                                     os.path.split(target_midifile)[-1].rsplit('.', 1)[0] + '.wav')
                convert_midi_to_wav(input_midifile, input_wav_savepath)
                convert_midi_to_wav(target_midifile, target_wav_savepath)

                # Get the data as a numpy array with shape [window_number, 1, window_length]
                input_data, _ = cut_track_and_stack(input_wav_savepath, window_length=window_length)
                target_data, _ = cut_track_and_stack(target_wav_savepath, window_length=window_length)

                print(input_data.shape, target_data.shape)

                # Create the datasets for each group
                if i == 0:
                    hdf[phase].create_dataset(name='input', data=input_data, maxshape=(None, 1, window_length),
                                              chunks=(32, 1, window_length))
                    hdf[phase].create_dataset(name='target', data=target_data, maxshape=(None, 1, window_length),
                                              chunks=(32, 1, window_length))
                else:
                    # Resize and append dataset
                    hdf[phase]['input'].resize((hdf[phase]['input'].shape[0] + input_data.shape[0]), axis=0)
                    hdf[phase]['input'][-input_data.shape[0]:] = input_data
                    hdf[phase]['target'].resize((hdf[phase]['target'].shape[0] + target_data.shape[0]), axis=0)
                    hdf[phase]['target'][-target_data.shape[0]:] = target_data


def create_npy_files(file_dict, temporary_directory_path, savepath, window_length=8192):
    # Iterate over the phases
    for phase in ['train', 'test', 'valid']:
        phase_directory = os.path.join(temporary_directory_path, phase)
        status_directories = {status: os.path.join(phase_directory, status) for status in ['input', 'target']}
        # Iterate all selected files
        for i, (input_midifile, target_midifile) in enumerate(zip(file_dict[phase]['input'],
                                                                  file_dict[phase]['target'])):
            # Create the new wavfiles
            input_wav_savepath = os.path.join(status_directories['input'],
                                                 os.path.split(input_midifile)[-1].rsplit('.', 1)[0] + '.wav')
            target_wav_savepath = os.path.join(status_directories['target'],
                                                 os.path.split(target_midifile)[-1].rsplit('.', 1)[0] + '.wav')
            convert_midi_to_wav(input_midifile, input_wav_savepath)
            convert_midi_to_wav(target_midifile, target_wav_savepath)

            # Get the data as a numpy array with shape [window_number, 1, window_length]
            input_data, _ = cut_track_and_stack(input_wav_savepath, window_length=window_length)
            target_data, _ = cut_track_and_stack(target_wav_savepath, window_length=window_length)

            # Create the datasets for each group
            window_number = min(input_data.shape[0], target_data.shape[0])
            if i == 0:
                phase_data = np.zeros((window_number, 2, window_length))

                # Store the data in the new data in the phase array
                phase_data[:, 0, :] = np.squeeze(input_data[:window_number])
                phase_data[:, 1, :] = np.squeeze(target_data[:window_number])
            else:
                temp_data = np.concatenate([input_data[:window_number], target_data[:window_number]], axis=1)
                phase_data = np.concatenate([phase_data, temp_data], axis=0)

        # Save array to disk
        np.save(savepath[phase], phase_data)


def create_modified_midifile(midi_filepath, midi_savepath, instrument=None, velocity=None, control=False,
                             control_value=None):
    """
    Modifies a .midi file according to the given parameters. The new .midi file is saved in a user specified location.
    Information regarding the codes for instruments and control is available here:
        http://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html
    All information is coded on 8-bit therefore the values must lie in [0 - 127].
    :param midi_filepath: path to the input .midi file (string).
    :param midi_savepath: path to the outout .midi file (string).
    :param instrument: code of the new instrument (scalar int).
    :param velocity: value of the velocity to set for all notes (scalar int).
    :param control: boolean indicating if control should be modified (pedal control is coded 64) (boolean).
    :param control_value: new value of the control (set to 0 to remove the control) (scalar int).
    :return: None.
    """
    # Load the original midi file
    mid_old = MidiFile(midi_filepath)

    # Create a new midi file
    mid_new = MidiFile()

    # Iterate over messages and apply transformation
    for track_old in mid_old.tracks:
        track_new = MidiTrack()
        for msg_old in track_old:

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


def convert_midi_to_wav(midi_filepath, wav_savepath, fs=44100):
    """
    Converts a .midi file to a .wav file using Timidity++. The .wav file is encoded on 16 bits integers, the amplitude
    is in [-2**15, 2**15 - 1]. The .wav is temporary written to disk.
    :param midi_filepath: location where the .midi file is stored (string).
    :param wav_savepath: location where to save the new .wav file (string).
    :param fs: sampling frequency in Hz to reconstruct the .wav file (scalar int).
    :return: None
    """
    command = 'timidity {} -s {} -Ow -o {}'.format(midi_filepath, fs, wav_savepath)
    call(command.split())


