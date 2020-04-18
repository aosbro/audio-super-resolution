from datasets.datasets import DatasetBeethoven
from utils.constants import *
import torch
from torch.utils import data


def get_the_data_loaders(train_datapath, test_datapath, valid_datapath, batch_size):
    train_dataset = DatasetBeethoven(train_datapath)
    test_dataset = DatasetBeethoven(test_datapath)
    valid_dataset = DatasetBeethoven(valid_datapath)

    # Create the generators
    train_params = {'batch_size': batch_size,
                    'shuffle': TRAIN_SHUFFLE,
                    'num_workers': NUM_WORKERS}
    test_params = {'batch_size': batch_size,
                   'shuffle': TEST_SHUFFLE,
                   'num_workers': NUM_WORKERS}
    valid_params = {'batch_size': batch_size,
                    'shuffle': VALID_SHUFFLE,
                    'num_workers': NUM_WORKERS}

    train_loader = data.DataLoader(train_dataset, **train_params)
    test_loader = data.DataLoader(test_dataset, **test_params)
    valid_loader = data.DataLoader(valid_dataset, **valid_params)
    return train_loader, test_loader, valid_loader


def get_consecutive_samples(dataset, index):
    """
    Samples a batch of consecutive samples from the data
    :param dataset: a torch Dataset object that contains the raw data
    :param index: index of the track that contains the samples
    :return: two batches of high and low resolutions samples as torch tensors
    """
    batch = [dataset.__getitem__(i + index * dataset.window_number) for i in range(dataset.window_number)]
    batch_h, batch_l = map(list, zip(*batch))
    batch_h, batch_l = torch.cat(batch_h), torch.cat(batch_l)
    return batch_h, batch_l
