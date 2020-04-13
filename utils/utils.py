from datasets.datasets import DatasetBeethoven
from utils.constants import *
from torch.utils import data


# def load_class(loadpath):
#     if torch.cuda.is_available():
#         return torch.load(loadpath)
#     return torch.load(loadpath, map_location=torch.device('cpu'))


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