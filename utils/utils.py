from datasets.datasets import DatasetBeethoven, DatasetMaestro
from utils.constants import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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

    train_loader = DataLoader(train_dataset, **train_params)
    test_loader = DataLoader(test_dataset, **test_params)
    valid_loader = DataLoader(valid_dataset, **valid_params)
    return train_loader, test_loader, valid_loader


def get_the_maestro_data_loaders(datapath, datasets_parameters, loaders_parameters):
    """
    Prepares the loaders for each phase ('train', 'test', 'valid') according to the parameters given in the dictionary
    loaders_parameters[phase]. The datapath is unique as the .hdf5 file contains the dataset for each phase.
    :param datapath: location of .hdf5 file (string).
    :param loaders_parameters: dictionary of parameters whose first keys are the phases (dictionary).
    :return: one data loader for each phase (torch DataLoader)
    """
    datasets = {phase: DatasetMaestro(datapath, phase, **datasets_parameters[phase]) for phase in ['train', 'test', 'valid']}
    data_loaders = [DataLoader(dataset, **loaders_parameters[phase]) for phase, dataset in datasets.items()]

    # train_loader = DataLoader(train_dataset, **loaders_parameters['train'])
    # test_loader = DataLoader(test_dataset, **loaders_parameters['test'])
    # valid_loader = data.DataLoader(valid_dataset, **loaders_parameters['valid'])
    # return train_loader, test_loader, valid_loader
    return tuple(data_loaders)


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


def plot_losses(losses, names, is_training, savepath=None):
    """
    Display the different losses accumulated throughout train and test phases
    :param losses: dictionary containing the different losses
    :param names: keys of the dictionary to select the desired losses
    :param is_training: boolean indicating the desired phase (train/test) needed for proper labels
    :param savepath: string path indicating where to save the plot
    :return: None
    """
    title = ('Train losses' if is_training else 'Test losses')
    xlabel = ('Gradient update' if is_training else 'Epoch')
    for name in names:
        if isinstance(losses[name], dict):
            for key, value in losses[name].items():
                plt.semilogy(value, label=key)
        else:
            plt.semilogy(losses[name], label=name)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Losses', fontsize=14)
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    plt.show()
