from datasets.datasets import DatasetBeethoven, DatasetMaestroHDF, DatasetMaestroNPY
from utils.constants import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.generator import Generator


def get_the_data_loaders(train_datapath, test_datapath, valid_datapath, batch_size):
    """
    Prepares the loaders for each phase (train, test, valid). The data for each phase is separated in different
    files whose location is given in parameters. Each file only contain the target signals as the input signals are
    generated on the fly.
    :param train_datapath: location of the .npy file for the train phase (string).
    :param test_datapath: location of the .npy file for the test phase (string).
    :param valid_datapath: location of the .npy file for the valid phase (string).
    :param batch_size: size of the batches, identical for all phases (scalar int).
    :return: one data loader for each phae (torch DataLoader)
    """
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


def get_the_maestro_data_loaders_hdf(datapath, datasets_parameters, loaders_parameters):
    """
    Prepares the loaders for each phase ('train', 'test', 'valid') according to the parameters given in the dictionary
    loaders_parameters[phase]. The datapath is unique as the .hdf5 file contains the dataset for each phase.
    :param datapath: location of .hdf5 file (string).
    :param loaders_parameters: dictionary of parameters whose first keys are the phases (dictionary).
    :return: one data loader for each phase (torch DataLoader)
    """
    datasets = {phase: DatasetMaestroHDF(datapath, phase, **datasets_parameters[phase]) for phase in ['train', 'test',
                                                                                                      'valid']}
    data_loaders = [DataLoader(dataset, **loaders_parameters[phase]) for phase, dataset in datasets.items()]
    return tuple(data_loaders)


def get_the_maestro_data_loaders_npy(datapath, loaders_parameters):
    """
    Prepares the loaders for each phase ('train', 'test', 'valid') according to the parameters given in the dictionary
    loaders_parameters[phase]. The data paths are contained in a dictionary, there is a single file for each phase
    :param datapath: dictionary containing the locations for each phase.
    :param loaders_parameters: dictionary of parameters whose first keys are the phases (dictionary).
    :return: one data loader for each phase (torch DataLoader)
    """
    datasets = {phase: DatasetMaestroNPY(datapath[phase]) for phase in ['train', 'test', 'valid']}
    data_loaders = [DataLoader(dataset, **loaders_parameters[phase]) for phase, dataset in datasets.items()]
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


def get_generator(loadpath, device):
    """
    Returns a pre-trained generator.
    :param loadpath: location of the generator trainer (string).
    :param device: either 'cpu' or 'cuda' depending on hardware availability (string).
    :return: pre-trained generator (nn.Module).
    """
    # Instantiate a new generator with identical architecture
    generator = Generator(kernel_sizes=KERNEL_SIZES,
                          channel_sizes_min=CHANNEL_SIZES_MIN,
                          p=DROPOUT_PROBABILITY,
                          n_blocks=N_BLOCKS_GENERATOR).to(device)

    # Restore pre-trained weights
    checkpoint = torch.load(loadpath, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    return generator.eval()
