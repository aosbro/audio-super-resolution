from utils.constants import *
from torch.utils import data
from processing.pre_processing import upsample, downsample
import numpy as np
import math
import torch
from scipy.signal import butter, filtfilt


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
    def __init__(self, phase, file):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass