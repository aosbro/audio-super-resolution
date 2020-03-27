from scipy.signal import decimate, resample
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


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
