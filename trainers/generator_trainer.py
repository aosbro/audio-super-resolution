from trainers.base_trainer import Trainer
from models.generator import Generator
from torch.optim import lr_scheduler
import numpy as np
import torch
from torch import nn
import os


class GeneratorTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, lr, loadpath, savepath, general_args, trainer_args):
        super(GeneratorTrainer, self).__init__(train_loader, test_loader, valid_loader, loadpath, savepath,
                                               general_args)
        # Model
        self.generator = Generator(general_args).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=30, gamma=0.5)

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
            for i in range(self.train_batches_per_epoch):
                # Get the next batch
                data_batch = next(self.train_loader_iter)
                # Transfer to GPU
                input_batch, target_batch = data_batch[0].to(self.device), data_batch[1].to(self.device)

                # Reset all gradients in the graph
                self.optimizer.zero_grad()

                # Generates a fake batch
                generated_batch = self.generator(input_batch)

                # Get the spectrogram
                specgram_target_batch = self.spectrogram(target_batch)
                specgram_generated_batch = self.spectrogram(generated_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(generated_batch, target_batch)
                freq_l2_loss = self.frequency_criterion(specgram_generated_batch, specgram_target_batch)
                self.train_losses['time_l2'].append(time_l2_loss.item())
                self.train_losses['freq_l2'].append(freq_l2_loss.item())
                loss = time_l2_loss #+ freq_l2_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Print message
            message = 'Train, epoch {}: \n' \
                      '\t Time: {} \n' \
                      '\t Frequency: {} \n'.format(
                self.epoch, np.mean(self.train_losses['time_l2'][-self.train_batches_per_epoch:]),
                np.mean(self.train_losses['freq_l2'][-self.train_batches_per_epoch:]))
            print(message)

            with torch.no_grad():
                self.eval()

            # Save the trainer state
            if self.need_saving:
                self.save()

            # Increment epoch counter
            self.epoch += 1
            self.scheduler.step()

    def eval(self):
        """
        Evaluates the model on the validation dataset
        :param epoch: Current epoch, used to print status information
        :return: None
        """
        self.generator.eval()
        batch_losses = {'time_l2': [], 'freq_l2': []}
        for i in range(self.valid_batches_per_epoch):
            # Get the next batch
            local_batch = next(self.valid_loader_iter)
            # Transfer to GPU
            input_batch, target_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

            # Generates a fake batch
            generated_batch = self.generator(input_batch)

            # Get the spectrogram
            specgram_target_batch = self.spectrogram(target_batch)
            specgram_fake_batch = self.spectrogram(generated_batch)

            # Compute and store the loss
            time_l2_loss = self.time_criterion(generated_batch, target_batch)
            freq_l2_loss = self.frequency_criterion(specgram_fake_batch, specgram_target_batch)
            batch_losses['time_l2'].append(time_l2_loss.item())
            batch_losses['freq_l2'].append(freq_l2_loss.item())

        # Store validation losses
        self.valid_losses['time_l2'].append(np.mean(batch_losses['time_l2']))
        self.valid_losses['freq_l2'].append(np.mean(batch_losses['freq_l2']))

        # Display valid loss
        message = 'Validation, epoch {}: \n' \
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
