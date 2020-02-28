from torch.utils import data
import numpy as np
from preprocessing.preprocessing import *
import matplotlib.pyplot as plt
import librosa.display
import torch
from torch import nn


class DatasetBeethoven(data.Dataset):
    def __init__(self, datapath, fs, ratio, overlap):
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
        self.window_length = 8000
        self.window_number = self.compute_window_number()

    def compute_window_number(self):
        """
        Computes the number of overlapping windows in a audio signal
        :return:
        """
        num = self.data.shape[1] - self.window_length
        den = self.window_length * (1 - self.overlap)
        return num / den + 1

    def __len__(self):
        """
        Returns the number of samples in the dataset
        :return: number of samples
        """
        return self.data.shape[0] * self.window_number

    def __getitem__(self, index):
        """
        Loads a single pair (x_h, x_l) of length 8000 sampled at 16 kHz for x_l
        :param index: index of the sample to load
        :return: corresponding image
        """
        signal_index = int(index // self.window_number)
        window_index = index % self.window_number
        signal = self.data[signal_index]
        window_start = int(window_index * (1 - self.overlap) * self.window_length)
        x_h = signal[window_start: window_start + self.window_length]
        x_l = upsample(downsample(x_h, self.ratio), self.ratio)
        X_h_db = compute_spectrogram(x_h)
        X_l_db = compute_spectrogram(x_l)
        return x_h, x_l, X_h_db, X_l_db


def main():
    datapath = '/media/thomas/Samsung_T5/VITA/data/music/music_train.npy'
    fs = 16000
    ratio = 8
    overlap = 0.5
    dataset = DatasetBeethoven(datapath, fs, ratio, overlap)
    print(dataset.__len__())
    print(dataset.window_number)

    x_h, x_l, X_h_db, X_l_db = dataset.__getitem__(18)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    librosa.display.specshow(X_h_db, sr=fs, x_axis='time', y_axis='hz', ax=axes[0])
    axes[0].set_title('High quality, spectrogram', fontsize=16)
    librosa.display.specshow(X_l_db, sr=fs, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Low quality, spectrogram', fontsize=16)
    plt.show()


    # pixel_shuffle = torch.nn.PixelShuffle(2)
    # gt = torch.randn(10, 4, 8, 8)
    # gt.requires_grad_(True)
    #
    # print(gt.requires_grad)
    # x = torch.randn(10, 16, 4, 4)
    # output = pixel_shuffle.forward(x)
    # loss_funtction = nn.MSELoss()
    # loss = loss_funtction(gt, output)
    # loss.backward()
    # test = gt.con
    # print(output.shape)


if __name__ == '__main__':
    main()
