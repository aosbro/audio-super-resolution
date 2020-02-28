from scipy.signal import decimate, resample
import librosa


def downsample(x, rate):
    """
    Down-samples the original signal by a factor rate after applying an anti-aliasing filter
    :param x: Original signal (numpy array)
    :param rate: Integer down-sampling factor
    :return: Down-sampled signal (numpy array)
    """
    x_ds = decimate(x, rate, axis=0, zero_phase=True)
    return x_ds.astype(x.dtype)


def upsample(x, rate):
    """
    Up-samples the original signal by a factor rate
    :param x: Previously down-sampled signal (numpy array)
    :param rate: Integer down-sampling factor
    :return: Up-sampled signal (numpy array)
    """
    x_us = resample(x, len(x)*rate, axis=0)
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
