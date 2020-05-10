import abc
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from itertools import cycle


class Trainer(abc.ABC):
    def __init__(self, train_loader, test_loader, valid_loader, loadpath, savepath):
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

        # Paths
        self.loadpath = loadpath
        self.savepath = savepath

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

        # Boolean indicting if autoencoder of generator
        self.is_autoencoder = False

        # Boolean indicating if the model needs to be saved
        self.need_saving = True

    def generate_single_validation_batch(self, model):
        """
        Loads a batch
        :param model: pre-trained model used to generate the fake samples
        :return: low resolution, high resolution and fake sample as torch tensor with dimension [B, 1, W]
        """
        model.eval()
        with torch.no_grad():
            local_batch = next(self.valid_loader_iter)
            x_h_batch, x_l_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)
            fake_batch = model(x_l_batch)
        if self.is_autoencoder:
            # Avoid returning the latent space
            return x_h_batch, x_l_batch, fake_batch[0]
        return x_h_batch, x_l_batch, fake_batch

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
        x_h_batch, x_l_batch, fake_batch = self.generate_single_validation_batch(model=model)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot(x_h_batch[index].cpu().detach().numpy().squeeze())
        axes[0].set_title('Real high quality sample', fontsize=16)
        axes[1].plot(fake_batch[index].cpu().detach().numpy().squeeze())
        axes[1].set_title('Fake high quality sample', fontsize=16)
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
        x_h_batch, x_l_batch, fake_batch = self.generate_single_validation_batch(model=model)

        # Get the power spectrogram in decibels
        specgram_h_db = self.amplitude_to_db(self.spectrogram(x_h_batch))
        specgram_l_db = self.amplitude_to_db(self.spectrogram(x_l_batch))
        specgram_fake_db = self.amplitude_to_db(self.spectrogram(fake_batch))

        # Define the extent
        f_min = 0
        f_max = specgram_h_db.shape[-2]
        k_min = 0
        k_max = specgram_h_db.shape[-1]

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(11, 7))
        axes[0].imshow(np.flip(specgram_h_db[index, 0].cpu().numpy(), axis=0), extent=[k_min, k_max, f_min, f_max])
        axes[0].set_title('High resolution', fontsize=16)
        axes[0].set_xlabel('Window index', fontsize=14)
        axes[0].set_ylabel('Frequency index', fontsize=14)
        axes[1].imshow(np.flip(specgram_l_db[index, 0].cpu().numpy(), axis=0), extent=[k_min, k_max, f_min, f_max])
        axes[1].set_title('Low resolution', fontsize=16)
        axes[1].set_xlabel('Window index', fontsize=14)
        axes[1].set_ylabel('Frequency index', fontsize=14)
        axes[2].imshow(np.flip(specgram_fake_db[index, 0].cpu().numpy(), axis=0), extent=[k_min, k_max, f_min, f_max])
        axes[2].set_title('Fake', fontsize=16)
        axes[2].set_xlabel('Window index', fontsize=14)
        axes[2].set_ylabel('Frequency index', fontsize=14)

        # Save plot if needed
        if savepath:
            plt.savefig(savepath)
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