from utils.constants import *
from torch.utils import data
from processing.pre_processing import upsample, downsample
import numpy as np
import math
import torch
from scipy.signal import butter, filtfilt
import h5py
from pysndfx import AudioEffectsChain
import time
import rtmidi


class DatasetBeethoven(data.Dataset):
    def __init__(self, datapath, ratio=4, overlap=0.5, use_windowing=False):
        """
        Initializes the class DatasetBeethoven
        :param datapath: path to raw .npy file
        :param ratio: down-sampling ratio
        :param overlap: overlap ratio with adjacent windows
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
        Computes the number of overlapping windows in a single track
        :return: the number of windows in the track
        """
        num = self.data.shape[1] - self.window_length
        den = self.window_length * (1 - self.overlap)
        return math.ceil(num / den) + 1

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        :return: number of samples
        """
        return self.data.shape[0] * self.window_number

    def butter_lowpass_filter(self, x_h, cutoff_frequency, order):
        """
        Applies a butterworth low-pass filter to the high resolution signal
        :param x_h: high resolution signal as a numpy array
        :param cutoff_frequency: desired max frequency of the filtered signal
        :param order: shapness of the filter
        :return: filtered signal as a numpy array
        """
        nyquist_frequency = self.fs / 2
        normalised_cutoff_frequency = cutoff_frequency / nyquist_frequency

        # Get the filter coefficients
        coefficients = butter(N=order, Wn=normalised_cutoff_frequency, btype='lowpass', analog=False)
        x_l = filtfilt(b=coefficients[0], a=coefficients[1], x=x_h)
        return x_l.astype(np.float32)

    def pad_signal(self, x):
        """
        Adds zero-padding at the end of the last window
        :param x: Signal with length smaller than the window length
        :return: Padded signal
        """
        # Apply hanning window before padding to avoid aliasing
        half_hanning = np.hanning(self.hanning_length)[self.hanning_length // 2:]
        x[- self.hanning_length // 2:] = x[- self.hanning_length // 2:] * half_hanning

        padding = self.window_length - x.shape[0]
        return np.concatenate([x, np.zeros(padding)])

    def __getitem__(self, index):
        """
        Loads a single pair (x_h, x_l) of length 8192 sampled at 16 kHz for x_l
        :param index: index of the sample to load
        :return: corresponding image
        """
        # Get the row of the sample
        signal_index = int(index // self.window_number)

        # Load the row
        signal = self.data[signal_index]

        # Get the position inside the row
        window_index = index % self.window_number
        window_start = int(window_index * (1 - self.overlap) * self.window_length)

        # Load the high quality signal containing WINDOW_LENGTH samples
        x_h = signal[window_start: window_start + self.window_length]

        # Add padding for last window
        if x_h.shape != self.window_length:
            x_h = self.pad_signal(x_h)

        # Apply hanning window over the whole window to avoid aliasing and for reconstruction
        if self.use_windowing:
            x_h *= np.hanning(WINDOW_LENGTH)

        x_l = upsample(downsample(x_h, self.ratio), self.ratio)
        return torch.from_numpy(np.expand_dims(x_h, axis=0)).float(), \
               torch.from_numpy(np.expand_dims(x_l, axis=0)).float()


class DatasetMaestro(data.Dataset):
    def __init__(self, hdf5_filepath, phase, noise_params, batch_size):
        self.hdf5_filepath = hdf5_filepath
        self.phase = phase
        self.noise_params = noise_params
        self.batch_size = batch_size

        # Initialize cache to store in RAM
        self.cache = None
        self.cache_size = 20 * batch_size
        self.cache_min_index = None
        self.load_chunk_to_cache(0)

    def __len__(self):
        with h5py.File(self.hdf5_filepath, 'r') as hdf:
            length = hdf[self.phase]['original'].shape[0]
        return length

    def is_in_cache(self, index):
        return index in set(range(self.cache_min_index, self.cache_min_index + self.cache_size))

    def load_chunk_to_cache(self, index):
        print('cache miss')
        with h5py.File(self.hdf5_filepath, 'r') as hdf:
            self.cache = hdf[self.phase]['original'][index: index + self.cache_size]
            self.cache_min_index = index

    def __getitem__(self, index):
        if not self.is_in_cache(index):
            self.load_chunk_to_cache(index)
        x = self.cache[index % self.cache_size]
        # with h5py.File(self.hdf5_filepath, 'r') as hdf:
        #     x = hdf[self.phase]['original'][index]
        return x, self.simulate_noisy_recording(x)

    def simulate_noisy_recording(self, x):
        fx = (
            AudioEffectsChain().reverb(
                reverberance=self.noise_params['reverberance'],
                hf_damping=self.noise_params['hf_damping'],
                room_scale=self.noise_params['room_scale'],
                pre_delay=self.noise_params['pre_delay'],
                wet_gain=self.noise_params['wet_gain']
            ).reverb(
                reverberance=self.noise_params['reverberance'],
                hf_damping=self.noise_params['hf_damping'],
                room_scale=self.noise_params['room_scale'],
                pre_delay=self.noise_params['pre_delay'],
                wet_gain=self.noise_params['wet_gain']
            ).reverb(
                reverberance=self.noise_params['reverberance'],
                hf_damping=self.noise_params['hf_damping'],
                room_scale=self.noise_params['room_scale'],
                pre_delay=self.noise_params['pre_delay'],
                wet_gain=self.noise_params['wet_gain']
            ).gain(
                db=self.noise_params['saturation_gain']
            ).gain(
                db=-self.noise_params['saturation_gain']
            ).highpass(
                frequency=self.noise_params['high_pass_cutoff']
            ).lowpass(
                frequency=self.noise_params['low_pass_cutoff']
            ).normalize()
        )
        return fx(np.squeeze(x)).reshape((1, -1))


def main():
    noise_params = {
        'reverberance': 40,
        'hf_damping': 20,
        'room_scale': 40,
        'pre_delay': 20,
        'wet_gain': -1,
        'saturation_gain': 17,
        'high_pass_cutoff': 200,
        'low_pass_cutoff': 10000
    }
    dataset = DatasetMaestro(hdf5_filepath='/media/thomas/Samsung_T5/VITA/data/maestro_data.h5',
                             phase='valid',
                             noise_params=noise_params,
                             batch_size=64)
    params = {'batch_size': 64,
              'shuffle': False,
              'num_workers': 6}

    loader = data.DataLoader(dataset, **params)

    start_time = time.time()
    loader_iter = iter(loader)
    for i in range(50):
        print(i)
        local_batch = next(loader_iter)
    end_time = time.time()
    print('done', end_time - start_time)


if __name__ == '__main__':
    main()