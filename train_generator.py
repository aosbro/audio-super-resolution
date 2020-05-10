from trainers.base_trainer import Trainer
from utils.utils import get_the_maestro_data_loaders_hdf, get_the_maestro_data_loaders_npy
from models.generator import Generator
from utils.constants import *
from torch.optim import lr_scheduler
import numpy as np
import torch
from torch import nn
import os


class GeneratorTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, lr, loadpath, savepath):
        super(GeneratorTrainer, self).__init__(train_loader, test_loader, valid_loader, loadpath, savepath)

        # Model
        self.generator = Generator(kernel_sizes=KERNEL_SIZES,
                                   channel_sizes_min=CHANNEL_SIZES_MIN,
                                   p=DROPOUT_PROBABILITY,
                                   n_blocks=N_BLOCKS_GENERATOR).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=15, gamma=0.5)

        # Load saved states
        if os.path.exists(self.loadpath):
            self.load()

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

    def train(self, epochs):
        """
        Trains the model for a specified number of epochs on the train dataset
        :param epochs: Number of iterations over the complete dataset to perform
        :return: None
        """
        for epoch in range(epochs):
            self.generator.train()
            for i in range(TRAIN_BATCH_ITERATIONS):
                # Get the next batch
                local_batch = next(self.train_loader_iter)
                # Transfer to GPU
                x_h_batch, x_l_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

                # Reset all gradients in the graph
                self.optimizer.zero_grad()

                # Generates a fake batch
                fake_batch = self.generator(x_l_batch)

                # Get the spectrogram
                specgram_h_batch = self.spectrogram(x_h_batch)
                specgram_fake_batch = self.spectrogram(fake_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(fake_batch, x_h_batch)
                freq_l2_loss = self.frequency_criterion(specgram_fake_batch, specgram_h_batch)
                self.train_losses['time_l2'].append(time_l2_loss.item())
                self.train_losses['freq_l2'].append(freq_l2_loss.item())
                loss = time_l2_loss + freq_l2_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Print message
                if not (i % 10):
                    message = 'Batch {}: \n' \
                              '\t Time: {} \n' \
                              '\t Frequency: {} \n' .format(i, time_l2_loss.item(), freq_l2_loss.item())
                    print(message)

            # Increment epoch counter
            self.epoch += 1
            self.scheduler.step()

            with torch.no_grad():
                self.eval()

            # Save the trainer state
            if self.need_saving:
                self.save()

    def eval(self):
        """
        Evaluates the model on the validation dataset
        :param epoch: Current epoch, used to print status information
        :return: None
        """
        self.generator.eval()
        batch_losses = {'time_l2': [], 'freq_l2': []}
        for i in range(VALID_BATCH_ITERATIONS):
            # Get the next batch
            local_batch = next(self.valid_loader_iter)
            # Transfer to GPU
            x_h_batch, x_l_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

            # Generates a fake batch
            fake_batch = self.generator(x_l_batch)

            # Get the spectrogram
            specgram_h_batch = self.spectrogram(x_h_batch)
            specgram_fake_batch = self.spectrogram(fake_batch)

            # Compute and store the loss
            time_l2_loss = self.time_criterion(fake_batch, x_h_batch)
            freq_l2_loss = self.frequency_criterion(specgram_fake_batch, specgram_h_batch)
            batch_losses['time_l2'].append(time_l2_loss.item())
            batch_losses['freq_l2'].append(freq_l2_loss.item())

        # Store validation losses
        self.valid_losses['time_l2'].append(np.mean(batch_losses['time_l2']))
        self.valid_losses['freq_l2'].append(np.mean(batch_losses['freq_l2']))

        # Display valid loss
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
            'generator_state_dict': self.generator.state_dict(),
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
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.valid_losses = checkpoint['valid_losses']


def train_generator(datapath, loadpath, savepath, datasets_parameters, loaders_parameters, use_hdf5):
    # Get the data loader for each phase
    if use_hdf5:
        train_loader, test_loader, valid_loader = get_the_maestro_data_loaders_hdf(datapath, datasets_parameters,
                                                                                   loaders_parameters)
    else:
        train_loader, test_loader, valid_loader = get_the_maestro_data_loaders_npy(datapath, loaders_parameters)

    # Load the train class which will automatically resume previous state from 'loadpath'
    generator_trainer = GeneratorTrainer(train_loader=train_loader,
                                         test_loader=test_loader,
                                         valid_loader=valid_loader,
                                         lr=LEARNING_RATE,
                                         loadpath=loadpath,
                                         savepath=savepath)

    return generator_trainer


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    datapath = {phase: os.path.join('data', phase + '.npy') for phase in ['train', 'test', 'valid']}
    datasets_parameters = {phase: {'batch_size': 64, 'use_cache': True} for phase in ['train', 'test', 'valid']}
    loaders_parameters = {phase: {'batch_size': 64, 'shuffle': False, 'num_workers': 2}
                          for phase in ['train', 'test', 'valid']}

    generator_trainer = train_generator(datapath=datapath,
                                        loadpath='',
                                        savepath='',
                                        datasets_parameters=datasets_parameters,
                                        loaders_parameters=loaders_parameters,
                                        use_hdf5=False)
    generator_trainer.train(epochs=1)


