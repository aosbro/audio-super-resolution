import torch


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

def lsd():
    pass




