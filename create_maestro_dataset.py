import os
import numpy as np
import h5py
from scipy.io import wavfile


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
    Computes the number of overlapping windows in a single track
    :return:
    """
    num = track_length - window_length
    den = window_length * (1 - overlap)
    return int(num // den + 2)


def cut_track_and_stack(track_path, use_transform, window_length=8192, overlap=0.5):
    # Load a single track
    _, track = wavfile.read(track_path)

    # Get rid of identical second channel
    track = (track[:, 0] / np.iinfo(np.int16).max).astype('float32')

    # Apply noise on the track if needed
    if use_transform:
        pass

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


def create_dataset(data_root, hdf5_path):
    wavfiles = sample_dataset(data_root, n_train=10, n_test=2, n_valid=2)
    window_length = 8192
    with h5py.File(hdf5_path, 'w') as hdf:
        # Create the groups inside the files
        for phase in ['train', 'test', 'valid']:
            hdf.create_group(name=phase)
            for i, file in enumerate(wavfiles[phase]):
                # Get a stacked track
                data = cut_track_and_stack(file, use_transform=False, window_length=8192)

                # Create the datasets for each group
                if i == 0:
                    hdf[phase].create_dataset(name='original', data=data, maxshape=(None, 1, window_length))
                else:
                    # Resize and append dataset
                    hdf[phase]['original'].resize((hdf[phase]['original'].shape[0] + data.shape[0]), axis=0)
                    hdf[phase]['original'][-data.shape[0]:] = data


    print(data.shape)


if __name__ == '__main__':
    create_dataset(data_root='/media/thomas/Samsung_T5/VITA/data/maestro-v1.0.0',
                   hdf5_path='/media/thomas/Samsung_T5/VITA/data/maestro_data.h5')
