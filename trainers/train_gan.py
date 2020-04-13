import os
from utils.utils import get_the_data_loaders
from utils.constants import *
from models.generator import Generator
from models.discriminator import Discriminator
from trainers.base_trainer import Trainer
from torch.optim import lr_scheduler
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import normalize
import torch
from torchaudio.transforms import Spectrogram


class GanTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, lr, loadpath, savepath, generator_path=None):
        super(GanTrainer, self).__init__(train_loader, test_loader, valid_loader, loadpath, savepath)

        # Models
        self.generator = Generator(kernel_sizes=KERNEL_SIZES,
                                   channel_sizes_min=CHANNEL_SIZES_MIN,
                                   p=DROPOUT_PROBABILITY,
                                   n_blocks=N_BLOCKS_GENERATOR).to(self.device)
        if generator_path:
            self.load_pretrained_generator(generator_path)

        self.discriminator = Discriminator(kernel_sizes=KERNEL_SIZES,
                                           channel_sizes_min=CHANNEL_SIZES_MIN,
                                           p=DROPOUT_PROBABILITY,
                                           n_blocks=N_BLOCKS_DISCRIMINATOR).to(self.device)

        # Optimizers and schedulers
        self.generator_optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=lr)
        self.generator_scheduler = lr_scheduler.StepLR(optimizer=self.generator_optimizer, step_size=1000, gamma=0.3)
        self.discriminator_scheduler = lr_scheduler.StepLR(optimizer=self.discriminator_optimizer, step_size=1000,
                                                           gamma=0.3)

        # Load saved states
        if os.path.exists(self.loadpath):
            self.load()

        # Loss function and stored losses
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.generator_time_criterion = nn.MSELoss()
        self.generator_frequency_criterion = nn.MSELoss()

        # Define labels
        self.real_label = 1
        self.fake_label = 0

        # Loss scaling factors
        self.lambda_adv = LAMBDA_ADVERSARIAL

        # Spectrogram converter
        self.spectrogram = Spectrogram(normalized=True).to(self.device)

    def load_pretrained_generator(self, generator_path):
        checkpoint = torch.load(generator_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])

    def train(self, epochs):
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            for i, local_batch in enumerate(self.train_loader):
                # Transfer to GPU
                x_h_batch, x_l_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)
                batch_size = x_h_batch.shape[0]

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train the discriminator with real data
                self.discriminator_optimizer.zero_grad()
                label = torch.full((batch_size,), self.real_label, device=self.device)
                output = self.discriminator(x_h_batch)

                # Compute and store the discriminator loss on real data
                loss_discriminator_real = self.adversarial_criterion(torch.squeeze(output), label)
                self.train_losses['discriminator_adversarial']['real'].append(loss_discriminator_real.item())
                loss_discriminator_real.backward()

                # Train the discriminator with fake data
                fake_batch = self.generator(x_l_batch)
                label.fill_(self.fake_label)
                output = self.discriminator(fake_batch.detach())

                # Compute and store the discriminator loss on fake data
                loss_discriminator_fake = self.adversarial_criterion(torch.squeeze(output), label)
                self.train_losses['discriminator_adversarial']['fake'].append(loss_discriminator_fake.item())
                loss_discriminator_fake.backward()

                # Update the discriminator weights
                self.discriminator_optimizer.step()

                ############################
                # Update G network: maximize log(D(G(z)))
                ###########################
                self.generator_optimizer.zero_grad()

                # Get the spectrogram
                specgram_h_batch = normalize(self.amplitude_to_db(self.spectrogram(x_h_batch)))
                specgram_fake_batch = normalize(self.amplitude_to_db(self.spectrogram(fake_batch)))

                # Fake labels are real for the generator cost
                label.fill_(self.real_label)
                output = self.discriminator(fake_batch)

                # Compute the generator loss on fake data
                loss_generator_adversarial = self.adversarial_criterion(torch.squeeze(output), label)
                self.train_losses['generator_adversarial'].append(loss_generator_adversarial.item())
                loss_generator_time = self.generator_time_criterion(fake_batch, x_h_batch)
                self.train_losses['time_l2'].append(loss_generator_time.item())
                loss_generator_frequency = self.generator_frequency_criterion(specgram_fake_batch, specgram_h_batch)
                self.train_losses['freq_l2'].append(loss_generator_frequency)

                loss_generator = self.lambda_adv * loss_generator_adversarial + loss_generator_time + \
                                 loss_generator_frequency

                # Back-propagate and update the generator weights
                loss_generator.backward()
                clip_grad_norm_(parameters=self.generator.parameters(), max_norm=GENERATOR_CLIP_VALUE)
                self.generator_optimizer.step()

                # Print message
                if not (i % 10):
                    message = 'Batch {}: \n' \
                              '\t Genarator: \n' \
                              '\t\t Time: {} \n' \
                              '\t\t Frequency: {} \n' \
                              '\t\t Adversarial: {} \n' \
                              '\t Discriminator: \n' \
                              '\t\t Real {} \n' \
                              '\t\t Fake {} \n'.format(i, loss_generator_time.item(), loss_generator_frequency.item(),
                                                       loss_generator_adversarial, loss_discriminator_real,
                                                       loss_discriminator_fake)
                    print(message)

            # Increment epoch counter
            self.epoch += 1

            # Save the trainer state
            self.save()

    def eval(self, epoch):
        pass

    def save(self):
        """
        Saves the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        torch.save({
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'generator_scheduler_state_dict': self.generator_scheduler.state_dict(),
            'discriminator_scheduler_state_dict': self.discriminator_scheduler.state_dict(),
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
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.generator_scheduler.load_state_dict(checkpoint['generator_scheduler_state_dict'])
        self.discriminator_scheduler.load_state_dict(checkpoint['discriminator_scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.valid_losses = checkpoint['valid_losses']


def get_gan_trainer(train_datapath, test_datapath, valid_datapath, loadpath, savepath, batch_size, generator_path):
    # Create the datasets
    train_loader, test_loader, valid_loader = get_the_data_loaders(train_datapath, test_datapath, valid_datapath,
                                                                   batch_size)

    gan_trainer = GanTrainer(train_loader=train_loader,
                             test_loader=test_loader,
                             valid_loader=valid_loader,
                             lr=LEARNING_RATE,
                             loadpath=loadpath,
                             savepath=savepath,
                             generator_path=generator_path)
    return gan_trainer


def train_gan(train_datapath, test_datapath, valid_datapath, loadpath, savepath, epochs, batch_size, generator_path):
    gan_trainer = get_gan_trainer(train_datapath=train_datapath,
                                  test_datapath=test_datapath,
                                  valid_datapath=valid_datapath,
                                  loadpath=loadpath,
                                  savepath=savepath,
                                  batch_size=batch_size,
                                  generator_path=generator_path)

    # Start training
    gan_trainer.train(epochs=epochs)
    return gan_trainer


# if __name__ == '__main__':
#     gan_trainer = train_gan(train_datapath=TRAIN_DATAPATH,
#                             test_datapath=TEST_DATAPATH,
#                             valid_datapath=VALID_DATAPATH,
#                             loadpath=GAN_PATH,
#                             savepath=GAN_PATH,
#                             epochs=1,
#                             batch_size=16,
#                             generator_path=GENERATOR_L2TF_PATH)

