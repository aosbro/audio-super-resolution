from torch.utils import data
import numpy as np
from preprocessing.preprocessing import *
import matplotlib.pyplot as plt
import librosa.display
import math
import torch
from torch import nn


class DatasetBeethoven(data.Dataset):
    def __init__(self, datapath, fs=16000, ratio=8, overlap=0.5):
        """
        Initializes the class DatasetBeethoven
        :param datapath: path to raw .npy file
        :param fs: original sampling frequency
        :param ratio: downsampling ratio
        :param overlap: overlap ratio with adjacent windows
        """
        self.data = np.load(datapath)
        self.fs = fs
        self.ratio = ratio
        self.overlap = overlap
        self.window_length = 8192
        self.window_number = self.compute_window_number()
        self.hanning_length = 101

    def compute_window_number(self):
        """
        Computes the number of overlapping windows in a audio signal
        :return:
        """
        num = self.data.shape[1] - self.window_length
        den = self.window_length * (1 - self.overlap)
        return math.ceil(num / den) + 1

    def __len__(self):
        """
        Returns the number of samples in the dataset
        :return: number of samples
        """
        return self.data.shape[0] * self.window_number

    def pad_signal(self, x):
        """
        Adds zero-padding at the end of the last window
        :param x: Signal with length smaller than the window length
        :return: Padded signal
        """
        # Apply hanning window to avoid alliasing
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
        signal_index = int(index // self.window_number)
        window_index = index % self.window_number
        signal = self.data[signal_index]
        window_start = int(window_index * (1 - self.overlap) * self.window_length)
        x_h = signal[window_start: window_start + self.window_length]

        # Add padding for last window
        if x_h.shape != self.window_length:
            x_h = self.pad_signal(x_h)

        x_l = upsample(downsample(x_h, self.ratio), self.ratio)
        return torch.from_numpy(np.expand_dims(x_h, axis=0)), torch.from_numpy(np.expand_dims(x_l, axis=0))



# def main():
#     datapath = '/media/thomas/Samsung_T5/VITA/data/music/music_train_.npy'
#     fs = 16000
#     ratio = 8
#     overlap = 0.5
#     dataset = DatasetBeethoven(datapath, fs, ratio, overlap)
#     print(dataset.__len__())
#     print(dataset.window_number)
#
#     x_h, x_l = dataset.__getitem__(5)
#         X_h_db = compute_spectrogram(x_h)
#         X_l_db = compute_spectrogram(x_l)
#
#     # Plotting
#     fig, axes = plt.subplots(1, 2, figsize=(14, 7))
#     librosa.display.specshow(X_h_db, sr=fs, x_axis='time', y_axis='hz', ax=axes[0])
#     axes[0].set_title('High quality, spectrogram', fontsize=16)
#     librosa.display.specshow(X_l_db, sr=fs, x_axis='time', y_axis='hz', ax=axes[1])
#     axes[1].set_title('Low quality, spectrogram', fontsize=16)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
