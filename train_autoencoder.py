from models.autoencoder import AutoEncoder
from utils.utils import get_the_maestro_data_loaders_hdf, get_the_maestro_data_loaders_npy
from utils.constants import *
import os
from trainers.base_trainer import Trainer
import torch
from torch.optim import lr_scheduler
from torch import nn
import numpy as np
from itertools import cycle


class AutoEncoderTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, lr, loadpath, savepath):
        super(AutoEncoderTrainer, self).__init__(train_loader, test_loader, valid_loader, loadpath, savepath)

        # Model
        self.autoencoder = AutoEncoder(kernel_sizes=KERNEL_SIZES,
                                       channel_sizes_min=CHANNEL_SIZES_MIN,
                                       p=DROPOUT_PROBABILITY,
                                       n_blocks=N_BLOCKS_AUTOENCODER).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.autoencoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5, gamma=0.5)

        # Load saved states
        if os.path.exists(self.loadpath):
            self.load()

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

        # Boolean to differentiate generator from auto-encoder
        self.is_autoencoder = True

        # Iterators to cycle over the datasets
        self.train_loader_iter = cycle(iter(self.train_loader))
        self.test_loader_iter = cycle(iter(self.test_loader))

    def train(self, epochs):
        for epoch in range(epochs):
            self.autoencoder.train()
            for i in range(TRAIN_BATCH_ITERATIONS):
                local_batch = next(self.train_loader_iter)
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                fake_batch, _ = self.autoencoder.forward(local_batch)

                # Get the spectrogram
                specgram_local_batch = self.spectrogram(local_batch)
                specgram_fake_batch = self.spectrogram(fake_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(fake_batch, local_batch)
                freq_l2_loss = self.frequency_criterion(specgram_fake_batch, specgram_local_batch)
                self.train_losses['time_l2'].append(time_l2_loss.item())
                self.train_losses['freq_l2'].append(freq_l2_loss.item())
                loss = time_l2_loss + freq_l2_loss

                # Print message
                if not (i % 10):
                    message = 'Batch {}: \n' \
                              '\t Time: {} \n' \
                              '\t Frequency: {} \n' .format(i, time_l2_loss.item(), freq_l2_loss.item())
                    print(message)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Increment epoch counter
            self.epoch += 1
            self.scheduler.step()

            with torch.no_grad():
                self.eval()

            # Save the trainer state
            if self.need_saving:
                self.save()

    def eval(self):
        with torch.no_grad():
            self.autoencoder.eval()
            batch_losses = {'time_l2': [], 'freq_l2': []}
            # for i, local_batch in enumerate(self.test_loader):
            for i in range(TEST_BATCH_ITERATIONS):
                # Transfer to GPU
                local_batch = next(self.test_loader_iter)
                local_batch = torch.cat(local_batch).to(self.device)

                # Forward pass
                fake_batch, _ = self.autoencoder.forward(local_batch)

                # Get the spectrogram
                specgram_local_batch = self.spectrogram(local_batch)
                specgram_fake_batch = self.spectrogram(fake_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(fake_batch, local_batch)
                freq_l2_loss = self.frequency_criterion(specgram_fake_batch, specgram_local_batch)
                batch_losses['time_l2'].append(time_l2_loss.item())
                batch_losses['freq_l2'].append(freq_l2_loss.item())

            # Store test losses
            self.test_losses['time_l2'].append(np.mean(batch_losses['time_l2']))
            self.test_losses['freq_l2'].append(np.mean(batch_losses['freq_l2']))

            # Display test loss
            message = 'Epoch {}: \n' \
                      '\t Time: {} \n' \
                      '\t Frequency: {} \n'.format(self.epoch,
                                                   np.mean(np.mean(batch_losses['time_l2'])),
                                                   np.mean(np.mean(batch_losses['freq_l2'])))
            print(message)

            # Check if the loss is decreasing
            self.check_improvement()

    def save(self):
        """
        Saves the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        torch.save({
            'epoch': self.epoch,
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'valid_losses': self.valid_losses
        }, self.savepath)

    def load(self):
        """
        Loads the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        checkpoint = torch.load(self.loadpath, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.valid_losses = checkpoint['valid_losses']


def train_autoencoder(datapath, loadpath, savepath, epochs, datasets_parameters, loaders_parameters, use_hdf5):
    # Get the data loader for each phase
    if use_hdf5:
        train_loader, test_loader, valid_loader = get_the_maestro_data_loaders_hdf(datapath, datasets_parameters,
                                                                                   loaders_parameters)
    else:
        train_loader, test_loader, valid_loader = get_the_maestro_data_loaders_npy(datapath, loaders_parameters)

    # Load the train class which will automatically resume previous state from 'loadpath'
    autoencoder_trainer = AutoEncoderTrainer(train_loader=train_loader,
                                             test_loader=test_loader,
                                             valid_loader=valid_loader,
                                             lr=AUTOENCODER_LEARNING_RATE,
                                             loadpath=loadpath,
                                             savepath=savepath)

    # Start training
    autoencoder_trainer.train(epochs=epochs)
    return autoencoder_trainer


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    datapath = {phase: os.path.join('data', phase + '.npy') for phase in ['train', 'test', 'valid']}
    datasets_parameters = {phase: {'batch_size': 64, 'use_cache': True} for phase in ['train', 'test', 'valid']}
    loaders_parameters = {phase: {'batch_size': 64, 'shuffle': False, 'num_workers': 2}
                          for phase in ['train', 'test', 'valid']}

    train_autoencoder(datapath=datapath,
                      loadpath='',
                      savepath=None,
                      epochs=1,
                      datasets_parameters=datasets_parameters,
                      loaders_parameters=loaders_parameters,
                      use_hdf5=False)
