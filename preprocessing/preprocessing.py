from scipy.signal import decimate, resample
import librosa


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
