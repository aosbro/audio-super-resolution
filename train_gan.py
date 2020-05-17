from trainers.gan_trainer import GanTrainer
from utils.utils import get_the_maestro_data_loaders_hdf, get_the_maestro_data_loaders_npy
from utils.constants import *
import os


def get_gan_trainer(datapath, loadpath, savepath, datasets_parameters, loaders_parameters, generator_path=None,
                    autoencoder_path=None, use_hdf5=None):
    # Get the data loader for each phase
    if use_hdf5:
        train_loader, test_loader, valid_loader = get_the_maestro_data_loaders_hdf(datapath, datasets_parameters,
                                                                                   loaders_parameters)
    else:
        train_loader, test_loader, valid_loader = get_the_maestro_data_loaders_npy(datapath, loaders_parameters)

    gan_trainer = GanTrainer(train_loader=train_loader,
                             test_loader=test_loader,
                             valid_loader=valid_loader,
                             lr=LEARNING_RATE,
                             loadpath=loadpath,
                             savepath=savepath,
                             generator_path=generator_path,
                             autoencoder_path=autoencoder_path)

    return gan_trainer


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    datapath = {phase: os.path.join('data', phase + '.npy') for phase in ['train', 'test', 'valid']}
    datasets_parameters = {phase: {'batch_size': 64, 'use_cache': True} for phase in ['train', 'test', 'valid']}
    loaders_parameters = {phase: {'batch_size': 64, 'shuffle': False, 'num_workers': 2}
                          for phase in ['train', 'test', 'valid']}

    gan_trainer = get_gan_trainer(datapath=datapath,
                                  loadpath='',
                                  savepath='',
                                  datasets_parameters=datasets_parameters,
                                  loaders_parameters=loaders_parameters,
                                  use_hdf5=False)
    gan_trainer.train(epochs=1)

