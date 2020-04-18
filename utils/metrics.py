import torch


def snr(x, r_ref):
    """
    Computes the signal-to-noise ratio of the the signal
    :param x: approximate reconstruction of the signal x_ref as a torch tensor
    :param r_ref: reference signal as a torch tensor
    :return: snr ratio as a torch tensor
    """
    num = torch.norm