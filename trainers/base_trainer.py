import abc
import torch
from processing.pre_processing import *
from torchaudio.transforms import Spectrogram, AmplitudeToDB


class Trainer(abc.ABC):
    def __init__(self, train_loader, test_loader, valid_loader, loadpath, savepath):
        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Data generators
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader

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

        self.is_autoencoder = False

    def generate_single_test_batch(self, model):
        model.eval()
        with torch.no_grad():
            local_batch = next(iter(self.test_loader))
            x_h_batch, x_l_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)
            fake_batch = model(x_l_batch)
        if self.is_autoencoder:
            return x_h_batch, fake_batch[0]
        return x_h_batch, fake_batch

    def plot_reconstruction_time_domain(self, index, model):
        """
        Plots real samples against fake sample in time domain
        :param model:
        :param index: index of the batch in the test generator to use
        :return: None
        """
        batch_size = self.test_loader.batch_size
        index = index % batch_size

        # Get a pair of high quality and fake samples batches
        x_h_batch, fake_batch = self.generate_single_test_batch(model=model)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].plot(x_h_batch[index].cpu().detach().numpy().squeeze())
        axes[0].set_title('Real high quality sample', fontsize=16)
        axes[1].plot(fake_batch[index].cpu().detach().numpy().squeeze())
        axes[1].set_title('Fake high quality sample', fontsize=16)
        plt.show()

    def plot_reconstruction_frequency_domain(self, index, model):
        """
        Plots real samples against fake sample in frequency domain
        :param model: model used to generate a fake batch (auto-encoder or generator)
        :param index:
        :return:
        """
        batch_size = self.test_loader.batch_size
        index = index % batch_size

        # Get a pair of low quality and fake samples batches
        x_h_batch, fake_batch = self.generate_single_test_batch(model=model)

        print(fake_batch[1].shape)

        specgram_h_db = self.amplitude_to_db(self.spectrogram(x_h_batch))
        specgram_fake_db = self.amplitude_to_db(self.spectrogram(fake_batch))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(np.flip(specgram_h_db[index, 0].cpu().numpy(), axis=0))
        axes[1].imshow(np.flip(specgram_fake_db[index, 0].cpu().numpy(), axis=0))
        plt.show()

    @abc.abstractmethod
    def train(self, epochs):
        """
        Trains the model for a specified number of epochs on the train dataset
        :param epochs: Number of iterations over the complete dataset to perform
        :return: None
        """

    @abc.abstractmethod
    def eval(self, epoch):
        """
        Evaluates the model on the test dataset
        :param epoch: Current epoch, used to print status information
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