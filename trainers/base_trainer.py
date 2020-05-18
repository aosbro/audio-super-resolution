import abc
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from itertools import cycle


class Trainer(abc.ABC):
    def __init__(self, train_loader, test_loader, valid_loader, general_args):
        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Data generators
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader

        # Iterators to cycle over the datasets
        self.train_loader_iter = cycle(iter(self.train_loader))
        self.valid_loader_iter = cycle(iter(self.valid_loader))
        self.test_loader_iter = cycle(iter(self.test_loader))

        # Epoch counter
        self.epoch = 0

        # Stored losses
        self.train_losses = {
            'time_l2': [],
            'freq_l2': [],
            'autoencoder_l2': [],
            'generator_adversarial': [],
            'discriminator_adversarial': {
                 'real': [],
                 'fake': []
             }
        }
        self.test_losses = {
            'time_l2': [],
            'freq_l2': [],
            'autoencoder_l2': [],
            'generator_adversarial': [],
            'discriminator_adversarial': {
                'real': [],
                'fake': []
            }
        }
        self.valid_losses = {
            'time_l2': [],
            'freq_l2': [],
            'autoencoder_l2': [],
            'generator_adversarial': [],
            'discriminator_adversarial': {
                'real': [],
                'fake': []
            }
        }

        # Time to frequency converter
        self.spectrogram = Spectrogram(normalized=True, n_fft=512, hop_length=128).to(self.device)
        self.amplitude_to_db = AmplitudeToDB()

        # Boolean indicting if auto-encoder or generator
        self.is_autoencoder = False

        # Boolean indicating if the model needs to be saved
        self.need_saving = True

        # Set the pseudo-epochs
        self.train_batches_per_epoch = general_args.train_batches_per_epoch
        self.test_batches_per_epoch = general_args.test_batches_per_epoch
        self.valid_batches_per_epoch = general_args.valid_batches_per_epoch

    def generate_single_validation_batch(self, model):
        """
        Loads a batch
        :param model: pre-trained model used to generate the fake samples
        :return: low resolution, high resolution and fake sample as torch tensor with dimension [B, 1, W]
        """
        model.eval()
        with torch.no_grad():
            data_batch = next(self.valid_loader_iter)
            input_batch, target_batch = data_batch[0].to(self.device), data_batch[1].to(self.device)
            if self.is_autoencoder:
                generated_batch = model(torch.cat([input_batch, target_batch]))[0]
            else:
                generated_batch = model(input_batch)
        return input_batch, target_batch, generated_batch

    def check_improvement(self):
        self.need_saving = np.less_equal(self.valid_losses['time_l2'][-1], min(self.valid_losses['time_l2'])) or \
                           np.less_equal(self.valid_losses['freq_l2'][-1], min(self.valid_losses['freq_l2']))

    def plot_reconstruction_time_domain(self, index, model):
        """
        Plots real samples against fake sample in time domain
        :param model:
        :param index: index of the batch in the validation generator to use
        :return: None
        """
        batch_size = self.valid_loader.batch_size
        index = index % batch_size

        # Get a pair of high quality and fake samples batches
        input_batch, target_batch, generated_batch = self.generate_single_validation_batch(model=model)

        # Plot differently for generator than for the autoencoder
        if self.is_autoencoder:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            # Plot the input of the auto-encoder (input-like)
            axes[0][0].plot(input_batch[index].cpu().detach().numpy().squeeze())
            axes[0][0].set_title('Input (input-like)', fontsize=16)

            # Plot the output of the auto-encoder (from an input-like signal)
            axes[0][1].plot(generated_batch[index].cpu().detach().numpy().squeeze())
            axes[0][1].set_title('Generated (from input-like)', fontsize=16)

            # Plot the input of the auto-encoder (target-like)
            axes[1][0].plot(target_batch[index].cpu().detach().numpy().squeeze())
            axes[1][0].set_title('Input (target-like)', fontsize=16)

            # Plot the output of the auto-encoder (from a target-like signal)
            axes[1][1].plot(generated_batch[index + batch_size].cpu().detach().numpy().squeeze())
            axes[1][1].set_title('Generated from (target-like)', fontsize=16)
            plt.show()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            # Plot the input signal
            axes[0].plot(input_batch[index].cpu().detach().numpy().squeeze())
            axes[0].set_title('Input', fontsize=16)

            # Plot the target signal
            axes[1].plot(target_batch[index].cpu().detach().numpy().squeeze())
            axes[1].set_title('Target', fontsize=16)

            # Plot the generated signal
            axes[2].plot(generated_batch[index].cpu().detach().numpy().squeeze())
            axes[2].set_title('Generated', fontsize=16)
            plt.show()

    def plot_reconstruction_frequency_domain(self, index, model, savepath=None):
        """
        Plots real samples against fake sample in frequency domain
        :param model: model used to generate a fake batch (auto-encoder or generator)
        :param index:
        :param savepath
        :return:
        """
        batch_size = self.valid_loader.batch_size
        index = index % batch_size

        # Get high resolution, low resolution and fake batches
        input_batch, target_batch, generated_batch = self.generate_single_validation_batch(model=model)

        # Get the power spectrogram in decibels
        specgram_input_db = self.amplitude_to_db(self.spectrogram(input_batch))
        specgram_target_db = self.amplitude_to_db(self.spectrogram(target_batch))
        specgram_generated_db = self.amplitude_to_db(self.spectrogram(generated_batch))

        # Define the extent
        extent = [0, specgram_input_db.shape[-1], 0, specgram_input_db.shape[-2]]

        # Plot
        if self.is_autoencoder:
            fig, axes = plt.subplots(2, 2, figsize=(12, 15))

            # Plot the input of the auto-encoder (input-like)
            axes[0][0].imshow(np.flip(specgram_input_db[index, 0].cpu().numpy(), axis=0), extent=extent)
            axes[0][0].set_title('Input (input-like)', fontsize=16)
            axes[0][0].set_xlabel('Window index', fontsize=14)
            axes[0][0].set_ylabel('Frequency index', fontsize=14)

            # Plot the output of the auto-encoder (from an input-like signal)
            axes[0][1].imshow(np.flip(specgram_generated_db[index, 0].cpu().numpy(), axis=0), extent=extent)
            axes[0][1].set_title('Generated (from input-like)', fontsize=16)
            axes[0][1].set_xlabel('Window index', fontsize=14)
            axes[0][1].set_ylabel('Frequency index', fontsize=14)

            # Plot the input of the auto-encoder (target-like)
            axes[1][0].imshow(np.flip(specgram_target_db[index, 0].cpu().numpy(), axis=0), extent=extent)
            axes[1][0].set_title('Input (target-like)', fontsize=16)
            axes[1][0].set_xlabel('Window index', fontsize=14)
            axes[1][0].set_ylabel('Frequency index', fontsize=14)

            # Plot the output of the auto-encoder (from a target-like signal)
            axes[1][1].imshow(np.flip(specgram_generated_db[index + batch_size, 0].cpu().numpy(), axis=0), extent=extent)
            axes[1][1].set_title('Generated (from target-like)', fontsize=16)
            axes[1][1].set_xlabel('Window index', fontsize=14)
            axes[1][1].set_ylabel('Frequency index', fontsize=14)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(11, 7))
            # Plot the input signal
            axes[0].imshow(np.flip(specgram_input_db[index, 0].cpu().numpy(), axis=0), extent=extent)
            axes[0].set_title('Input', fontsize=16)
            axes[0].set_xlabel('Window index', fontsize=14)
            axes[0].set_ylabel('Frequency index', fontsize=14)

            # Plot the target signal
            axes[1].imshow(np.flip(specgram_target_db[index, 0].cpu().numpy(), axis=0), extent=extent)
            axes[1].set_title('Target', fontsize=16)
            axes[1].set_xlabel('Window index', fontsize=14)
            axes[1].set_ylabel('Frequency index', fontsize=14)

            # Plot the generated signal
            axes[2].imshow(np.flip(specgram_generated_db[index, 0].cpu().numpy(), axis=0), extent=extent)
            axes[2].set_title('Generated', fontsize=16)
            axes[2].set_xlabel('Window index', fontsize=14)
            axes[2].set_ylabel('Frequency index', fontsize=14)

        # Save plot if needed
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def plot_l2_losses(self):
        """
        Plot the train and validation losses of type L2. The train losses are stored at a batch resolution it must
        therefore be aggregated before plotting.
        :return: None
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for key, value in self.train_losses.items():
            if key.endswith('l2') and value:
                epoch_loss = np.reshape(np.array(value), newshape=[-1, self.train_batches_per_epoch]).mean(axis=-1)
                axes[0].plot(epoch_loss, label=key)
                axes[0].set_title('Train losses, L2', fontsize=16)
                axes[0].set_xlabel('Pseudo-epochs ({} batches)'.format(self.train_batches_per_epoch), fontsize=14)
                axes[0].legend()
        for key, value in self.valid_losses.items():
            if key.endswith('l2') and value:
                axes[1].plot(value, label=key)
                axes[1].set_title('Validation losses, L2', fontsize=16)
                axes[1].set_xlabel('Pseudo-epochs ({} batches)'.format(self.valid_batches_per_epoch), fontsize=14)
                axes[1].legend()
        plt.show()

    @abc.abstractmethod
    def train(self, epochs):
        """
        Trains the model for a specified number of epochs on the train dataset
        :param epochs: Number of iterations over the complete dataset to perform
        :return: None
        """

    @abc.abstractmethod
    def eval(self):
        """
        Evaluates the model on the validation dataset
        :return: None
        """

    @abc.abstractmethod
    def save(self):
        """
        Saves the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """

    @abc.abstractmethod
    def load(self):
        """
        Loads the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """