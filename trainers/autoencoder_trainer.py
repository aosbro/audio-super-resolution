from trainers.base_trainer import Trainer
import torch
from torch.optim import lr_scheduler
from torch import nn
import numpy as np
from models.autoencoder import AutoEncoder
import os


class AutoEncoderTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, general_args, trainer_args):
        super(AutoEncoderTrainer, self).__init__(train_loader, test_loader, valid_loader, general_args)
        # Paths
        self.loadpath = trainer_args.loadpath
        self.savepath = trainer_args.savepath

        # Model
        self.autoencoder = AutoEncoder(general_args=general_args).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.autoencoder.parameters(), lr=trainer_args.lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=trainer_args.scheduler_step,
                                             gamma=trainer_args.scheduler_gamma)

        # Load saved states
        if os.path.exists(trainer_args.loadpath):
            self.load()

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

        # Boolean to differentiate generator from auto-encoder
        self.is_autoencoder = True

    def train(self, epochs):
        for epoch in range(epochs):
            self.autoencoder.train()
            for i in range(self.train_batches_per_epoch):
                local_batch = next(self.train_loader_iter)
                # Concatenate the input and target signals along first dimension and transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                generated_batch, _ = self.autoencoder.forward(local_batch)

                # Get the spectrogram
                specgram_local_batch = self.spectrogram(local_batch)
                specgram_generated_batch = self.spectrogram(generated_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(generated_batch, local_batch)
                freq_l2_loss = self.frequency_criterion(specgram_generated_batch, specgram_local_batch)
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

            with torch.no_grad():
                self.eval()

            # Save the trainer state
            if self.need_saving:
                self.save()

            # Increment epoch counter
            self.epoch += 1
            self.scheduler.step()

    def eval(self):
        with torch.no_grad():
            self.autoencoder.eval()
            batch_losses = {'time_l2': [], 'freq_l2': []}
            for i in range(self.valid_batches_per_epoch):
                # Transfer to GPU
                data_batch = next(self.valid_loader_iter)
                data_batch = torch.cat(data_batch).to(self.device)

                # Forward pass
                generated_batch, _ = self.autoencoder.forward(data_batch)

                # Get the spectrogram
                specgram_batch = self.spectrogram(data_batch)
                specgram_generated_batch = self.spectrogram(generated_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(generated_batch, data_batch)
                freq_l2_loss = self.frequency_criterion(specgram_generated_batch, specgram_batch)
                batch_losses['time_l2'].append(time_l2_loss.item())
                batch_losses['freq_l2'].append(freq_l2_loss.item())

            # Store the validation losses
            self.valid_losses['time_l2'].append(np.mean(batch_losses['time_l2']))
            self.valid_losses['freq_l2'].append(np.mean(batch_losses['freq_l2']))

            # Display the validation loss
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
