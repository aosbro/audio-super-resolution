import torch
from torchaudio.transforms import Spectrogram


def snr(x, x_ref):
    """
    Computes the average signal-to-noise ratio of a batch of signals
    :param x: approximate reconstruction of the signals x_ref as a torch tensor of dimension [B, W]
    :param x_ref: reference signals as a torch tensor of dimension [B, W]
    :return: snr ratio as a torch tensor
    """
    num = torch.norm(x_ref, p=2, dim=1)
    den = torch.norm(x - x_ref, p=2, dim=1)
    return torch.mean(20 * torch.log10_(num / den))


def lsd(x, x_ref):
    """
    Computes the average LSD metric of a batch of signals
    :param x: approximate reconstruction of the signals x_ref as a torch tensor of dimension [B, W]
    :param x_ref: reference signals as a torch tensor of dimension [B, W]
    :return: lsd metric as a torch tensor
    """
    # Get the STFT of the signals
    spectrogram = Spectrogram(n_fft=400, power=2)
    X, X_ref = spectrogram(x), spectrogram(x_ref)
    return torch.mean(torch.mean((X / X_ref).log10_().pow_(2), dim=-1).pow_(0.5))




