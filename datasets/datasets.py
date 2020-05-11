from utils.constants import *
from torch.utils import data
from processing.pre_processing import upsample, downsample
import numpy as np
import math
import torch
from scipy.signal import butter, filtfilt
import h5py


class DatasetBeethoven(data.Dataset):
    def __init__(self, datapath, ratio=4, overlap=0.5, use_windowing=False):
        """
        Initializes the class DatasetBeethoven that stores the original high quality data (target) and applies the
        transformation to get the low quality data (input) on the fly. The raw data is stored as [n_tracks,
        track_length]. The track length is fixed and is equal to 128000. As the track length is too large to be fed
        directly in the models it is further split in overlapping windows, this split is done on the fly. The
        transformation applied on the original signal is a down-sampling in the time domain by 'ratio'. This dataset
        only stores the data for a single phase, therefore one should instantiate such a class for train,
        test and validation individually.
        :param datapath: path to raw .npy file (string).
        :param ratio: down-sampling ratio (scalar int).
        :param overlap: overlap ratio with adjacent windows (float in [0, 1)).
        :param use_windowing: boolean indicating if a Hanning window must be applied on the input tensors (boolean).
        """
        self.data = np.load(datapath)
        self.ratio = ratio
        self.overlap = overlap
        self.window_length = WINDOW_LENGTH
        self.window_number = self.compute_window_number()
        self.hanning_length = HANNING_WINDOW_LENGTH
        self.fs = 16000
        self.use_windowing = use_windowing

    def compute_window_number(self):
        """
        Computes the number of overlapping windows in a single track.
        :return: the number of windows in the track (scalar int).
        """
        num = self.data.shape[1] - self.window_length
        den = self.window_length * (1 - self.overlap)
        return math.ceil(num / den) + 1

    def __len__(self):
        """
        Returns the total number of samples in the dataset: n_tracks * (windows per track).
        :return: number of samples (scalar int).
        """
        return self.data.shape[0] * self.window_number

    def butter_lowpass_filter(self, x_target, cutoff_frequency, order):
        """
        Applies a butterworth low-pass filter to the high resolution signal.
        :param x_target: high resolution signal as a numpy array.
        :param cutoff_frequency: desired max frequency of the filtered signal (scalar float).
        :param order: shapness of the filter (scalar int).
        :return: filtered signal (numpy array).
        """
        nyquist_frequency = self.fs / 2
        normalised_cutoff_frequency = cutoff_frequency / nyquist_frequency

        # Get the filter coefficients
        coefficients = butter(N=order, Wn=normalised_cutoff_frequency, btype='lowpass', analog=False)
        x_input = filtfilt(b=coefficients[0], a=coefficients[1], x=x_target)
        return x_input.astype(np.float32)

    def pad_signal(self, x):
        """
        Adds zero-padding at the end of the last window.
        :param x: Signal with length smaller than the window length (numpy array).
        :return: Padded signal (numpy array).
        """
        # Apply a half Hanning window before padding to avoid aliasing
        half_hanning = np.hanning(self.hanning_length)[self.hanning_length // 2:]
        x[- self.hanning_length // 2:] = x[- self.hanning_length // 2:] * half_hanning

        padding = self.window_length - x.shape[0]
        return np.concatenate([x, np.zeros(padding)])

    def __getitem__(self, index):
        """
        Loads a single pair (x_target, x_input) of length 8192 sampled at 16 kHz for x_target
        :param index: index of the sample to load (scalar int).
        :return: corresponding pair of signals (tuple of torch tensor with shape [1, window_length]).
        """
        # Get the row of the sample
        signal_index = int(index // self.window_number)

        # Load the row
        signal = self.data[signal_index]

        # Get the position inside the row
        window_index = index % self.window_number
        window_start = int(window_index * (1 - self.overlap) * self.window_length)

        # Load the high quality signal containing WINDOW_LENGTH samples
        x_target = signal[window_start: window_start + self.window_length]

        # Add padding for last window
        if x_target.shape != self.window_length:
            x_target = self.pad_signal(x_target)

        # Apply hanning window over the whole window to avoid aliasing and for reconstruction
        if self.use_windowing:
            x_target *= np.hanning(WINDOW_LENGTH)

        x_input = upsample(downsample(x_target, self.ratio), self.ratio)
        return torch.from_numpy(np.expand_dims(x_input, axis=0)).float(), \
               torch.from_numpy(np.expand_dims(x_target, axis=0)).float()


class DatasetMaestroHDF(data.Dataset):
    def __init__(self, hdf5_filepath, phase, batch_size, use_cache, cache_size=30):
        """
        Initializes the class DatasetMaestroHDF that stores the data in a .hdf5 file that contains the complete data for
        all phases (train, test, validation). It contains the input data as well as the target data to reduce the amount
        of computation done in fly. The samples are first split w.r.t. the phase (train, test, validation) and then
        w.r.t. the status (input, target). A pair of (input, target) samples is accessed with same index. For a given
        phase a pair is accessed as: (hdf[phase]['input'][index], hdf[phase]['target'][index]).
        The .hdf5 file is stored on disk and only the queried samples are loaded in RAM. To increase retrieval speed
        a small cache in RAM  is implemented. When using the cache, one should note the following observations:
            - The speed will only improve if the data is not shuffled.
            - The cache size must be adapted to the computer used.
            - The number of workers of the data loader must be adapted to the computer used and the cache size.
            - The cache size must be a multiple of the chunk size that was used to create the dataset.

        :param hdf5_filepath: location of the .hdf5 file (string).
        :param phase: current phase in 'train', 'test', 'validation' (string).
        :param batch_size: size of a single batch (scalar int).
        :param use_cache: boolean indicating if the cache should be used or not (boolean).
        :param cache_size: size of the cache in number of batches (scalar int).
        """
        self.hdf5_filepath = hdf5_filepath
        self.phase = phase
        self.batch_size = batch_size

        # Initialize cache to store in RAM
        self.use_cache = use_cache
        if self.use_cache:
            self.cache = {'input': None, 'target': None}
            self.cache_size = cache_size * batch_size
            self.cache_min_index = None
            self.cache_max_index = None
            self.load_chunk_to_cache(0)

    def __len__(self):
        """
        Returns the total length of the dataset
        :return: length of the dataset (scalar int)
        """
        with h5py.File(self.hdf5_filepath, 'r') as hdf:
            length = hdf[self.phase]['input'].shape[0]
        return length

    def is_in_cache(self, index):
        """
        Checks if the queried data is in cache.
        :param index: index of the sample to load (scalar int)
        :return: boolean indicating if the data is available in cache (boolean).
        """
        return index in set(range(self.cache_min_index, self.cache_max_index))

    def load_chunk_to_cache(self, index):
        """
        Loads a chunk of data in cache from disk. The chunk of data is the block of size self.size_cache and contains
        the samples following the current index. This is only efficient if data is not shuffled.
        :param index: index of a single sample that is currently being queried (scalar int).
        :return: None.
        """
        with h5py.File(self.hdf5_filepath, 'r') as hdf:
            self.cache_min_index = index
            self.cache_max_index = min(len(self), index + self.cache_size)
            self.cache['input'] = hdf[self.phase]['input'][self.cache_min_index: self.cache_max_index]
            self.cache['target'] = hdf[self.phase]['target'][self.cache_min_index: self.cache_max_index]

    def __getitem__(self, index):
        """
        Loads a single pair (x_input, x_target).
        :param index: index of the sample to load (scalar int).
        :return: corresponding pair of signals (tuple of torch tensor with shape [1, window_length]).
        """
        if self.use_cache:
            if not self.is_in_cache(index):
                self.load_chunk_to_cache(index)
            x_input = self.cache['input'][index - self.cache_min_index]
            x_target = self.cache['target'][index - self.cache_min_index]
        else:
            with h5py.File(self.hdf5_filepath, 'r') as hdf:
                x_input = hdf[self.phase]['input'][index]
                x_target = hdf[self.phase]['target'][index]
        return torch.from_numpy(x_input).float(), torch.from_numpy(x_target).float()


class DatasetMaestroNPY(data.Dataset):
    def __init__(self, datapath):
        """
        Initializes the class DatasetMaestroNPY that is based on a
        :param datapath:
        """
        self.data = np.load(datapath)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x_input, x_target = self.data[index, 0, :][None], self.data[index, 1, :][None]
        return torch.from_numpy(x_input).float(), torch.from_numpy(x_target).float()


# def main():
#     dataset = DatasetMaestroNPY('../data/train.npy')
#     test = dataset.__getitem__(3000)
#     print(test[0].shape, test[1].shape)
#
#     plt.plot(test[0].squeeze().cpu().numpy())
#     plt.plot(test[1].squeeze().cpu().numpy())
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
