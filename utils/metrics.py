from torchaudio.transforms import Spectrogram
import torch


def snr(x, x_ref):
    """
    Computes the signal-to-noise ratio of a batch of signals
    :param x: approximate reconstruction of the signals x_ref as a torch tensor of dimension [B, W]
    :param x_ref: reference signals as a torch tensor of dimension [B, W]
    :return: snr ratio as a torch tensor
    """
    num = torch.norm(x_ref, p=2, dim=1).pow(2)
    den = torch.norm(x - x_ref, p=2, dim=1).pow(2)
    return 10 * (num / den).log10()


def lsd(x, x_ref):
    """
    Computes the LSD metric of a batch of signals
    :param x: approximate reconstruction of the signals x_ref as a torch tensor of dimension [B, W]
    :param x_ref: reference signals as a torch tensor of dimension [B, W]
    :return: lsd metric as a torch tensor
    """
    # Get the STFT of the signals
    spectrogram = Spectrogram(n_fft=512, hop_length=128, power=2)
    X, X_ref = spectrogram(x), spectrogram(x_ref)
    return (X / X_ref).log10().pow(2).mean(dim=1).pow(0.5).mean(dim=1)




