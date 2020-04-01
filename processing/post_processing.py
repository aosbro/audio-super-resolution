import torch
from utils.constants import *


def overlap_and_add_samples(batch, overlap=0.5):
    """
    Re-construct a full sample from its sub-parts using the OLA algorithm
    :param overlap: proportion of the overlap between contiguous signals
    :param batch: torch tensor of shape [B, 1, WINDOW_LENGTH]
    :return:
    """
    # Compute the size of the full sample
    N, _, single_sample_size = batch.size()
    full_sample_size = int(single_sample_size * (1 + (N - 1) * (1 - overlap)))

    # Initialize the full sample
    full_sample = torch.empty(full_sample_size)

    for window_index in range(N):
        window_stat = int(window_index * (1 - overlap) * WINDOW_LENGTH)
        full_sample[window_stat: window_stat + WINDOW_LENGTH] += batch[window_index].squeeze()
    return full_sample
