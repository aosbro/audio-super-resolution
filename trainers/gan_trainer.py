from models.generator import Generator
from models.autoencoder import AutoEncoder
from models.discriminator import Discriminator
from trainers.base_trainer import Trainer
from torch.optim import lr_scheduler
from torch import nn
import torch
from torchaudio.transforms import Spectrogram
import numpy as np
import os


class GanTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, general_args, trainer_args):
        super(GanTrainer, self).__init__(train_loader, test_loader, valid_loader, general_args)
        # Paths
        self.loadpath = trainer_args.loadpath
        self.savepath = trainer_args.savepath

        # Load the auto-encoder
        self.use_autoencoder = False
        if trainer_args.autoencoder_path and os.path.exists(trainer_args.autoencoder_path):
            self.use_autoencoder = True
            self.autoencoder = AutoEncoder(general_args=general_args).to(self.device)
            self.load_pretrained_autoencoder(trainer_args.autoencoder_path)
            self.autoencoder.eval()

        # Load the generator
        self.generator = Generator(general_args=general_args).to(self.device)
        if trainer_args.generator_path and os.path.exists(trainer_args.generator_path):
            self.load_pretrained_generator(trainer_args.generator_path)

        self.discriminator = Discriminator(general_args=general_args).to(self.device)

        # Optimizers and schedulers
        self.generator_optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=trainer_args.generator_lr)
        self.discriminator_optimizer = torch.optim.Adam(params=self.discriminator.parameters(),
                                                        lr=trainer_args.discriminator_lr)
        self.generator_scheduler = lr_scheduler.StepLR(optimizer=self.generator_optimizer,
                                                       step_size=trainer_args.generator_scheduler_step,
                                                       gamma=trainer_args.generator_scheduler_gamma)
        self.discriminator_scheduler = lr_scheduler.StepLR(optimizer=self.discriminator_optimizer,
                                                           step_size=trainer_args.discriminator_scheduler_step,
                                                           gamma=trainer_args.discriminator_scheduler_gamma)

        # Load saved states
        if os.path.exists(self.loadpath):
            self.load()

        # Loss function and stored losses
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.generator_time_criterion = nn.MSELoss()
        self.generator_frequency_criterion = nn.MSELoss()
        self.generator_autoencoder_criterion = nn.MSELoss()

        # Define labels
        self.real_label = 1
        self.generated_label = 0

        # Loss scaling factors
        self.lambda_adv = trainer_args.lambda_adversarial

        # Spectrogram converter
        self.spectrogram = Spectrogram(normalized=True).to(self.device)

        # Boolean indicating if the model needs to be saved
        self.need_saving = True

        # Boolean if the generator receives the feedback from the discriminator
        self.use_adversarial = True

    def load_pretrained_generator(self, generator_path):
        """
        Loads a pre-trained generator. Can be used to stabilize the training.
        :param generator_path: location of the pre-trained generator (string).
        :return: None
        """
        checkpoint = torch.load(generator_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])

    def load_pretrained_autoencoder(self, autoencoder_path):
        """
        Loads a pre-trained auto-encoder. Can be used to infer
        :param autoencoder_path: location of the pre-trained auto-encoder (string).
        :return: None
        """
        checkpoint = torch.load(autoencoder_path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])

    def train(self, epochs):
        """
        Trains the GAN for a given number of pseudo-epochs.
        :param epochs: Number of time to iterate over a part of the dataset (int).
        :return: None
        """
        for epoch in range(epochs):
            for i in range(self.train_batches_per_epoch):
                self.generator.train()
                self.discriminator.train()
                # Transfer to GPU
                local_batch = next(self.train_loader_iter)
                input_batch, target_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)
                batch_size = input_batch.shape[0]

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train the discriminator with real data
                self.discriminator_optimizer.zero_grad()
                label = torch.full((batch_size,), self.real_label, device=self.device)
                output = self.discriminator(target_batch)

                # Compute and store the discriminator loss on real data
                loss_discriminator_real = self.adversarial_criterion(output, torch.unsqueeze(label, dim=1))
                self.train_losses['discriminator_adversarial']['real'].append(loss_discriminator_real.item())
                loss_discriminator_real.backward()

                # Train the discriminator with fake data
                generated_batch = self.generator(input_batch)
                label.fill_(self.generated_label)
                output = self.discriminator(generated_batch.detach())

                # Compute and store the discriminator loss on fake data
                loss_discriminator_generated = self.adversarial_criterion(output, torch.unsqueeze(label, dim=1))
                self.train_losses['discriminator_adversarial']['fake'].append(loss_discriminator_generated.item())
                loss_discriminator_generated.backward()

                # Update the discriminator weights
                self.discriminator_optimizer.step()

                ############################
                # Update G network: maximize log(D(G(z)))
                ###########################
                self.generator_optimizer.zero_grad()

                # Get the spectrogram
                specgram_target_batch = self.spectrogram(target_batch)
                specgram_fake_batch = self.spectrogram(generated_batch)

                # Fake labels are real for the generator cost
                label.fill_(self.real_label)
                output = self.discriminator(generated_batch)

                # Compute the generator loss on fake data
                # Get the adversarial loss
                loss_generator_adversarial = torch.zeros(size=[1], device=self.device)
                if self.use_adversarial:
                    loss_generator_adversarial = self.adversarial_criterion(output, torch.unsqueeze(label, dim=1))
                self.train_losses['generator_adversarial'].append(loss_generator_adversarial.item())

                # Get the L2 loss in time domain
                loss_generator_time = self.generator_time_criterion(generated_batch, target_batch)
                self.train_losses['time_l2'].append(loss_generator_time.item())

                # Get the L2 loss in frequency domain
                loss_generator_frequency = self.generator_frequency_criterion(specgram_fake_batch, specgram_target_batch)
                self.train_losses['freq_l2'].append(loss_generator_frequency.item())

                # Get the L2 loss in embedding space
                loss_generator_autoencoder = torch.zeros(size=[1], device=self.device, requires_grad=True)
                if self.use_autoencoder:
                    # Get the embeddings
                    _, embedding_target_batch = self.autoencoder(target_batch)
                    _, embedding_generated_batch = self.autoencoder(generated_batch)
                    loss_generator_autoencoder = self.generator_autoencoder_criterion(embedding_generated_batch,
                                                                                      embedding_target_batch)
                    self.train_losses['autoencoder_l2'].append(loss_generator_autoencoder.item())

                # Combine the different losses
                loss_generator = self.lambda_adv * loss_generator_adversarial + loss_generator_time + \
                                 loss_generator_frequency + loss_generator_autoencoder

                # Back-propagate and update the generator weights
                loss_generator.backward()
                self.generator_optimizer.step()

                # Print message
                if not (i % 10):
                    message = 'Batch {}: \n' \
                              '\t Genarator: \n' \
                              '\t\t Time: {} \n' \
                              '\t\t Frequency: {} \n' \
                              '\t\t Autoencoder {} \n' \
                              '\t\t Adversarial: {} \n' \
                              '\t Discriminator: \n' \
                              '\t\t Real {} \n' \
                              '\t\t Fake {} \n'.format(i,
                                                       loss_generator_time.item(),
                                                       loss_generator_frequency.item(),
                                                       loss_generator_autoencoder.item(),
                                                       loss_generator_adversarial.item(),
                                                       loss_discriminator_real.item(),
                                                       loss_discriminator_generated.item())
                    print(message)

            # Evaluate the model
            with torch.no_grad():
                self.eval()

            # Save the trainer state
            self.save()
            # if self.need_saving:
            #     self.save()

            # Increment epoch counter
            self.epoch += 1
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        batch_losses = {'time_l2': [], 'freq_l2': []}
        for i in range(self.valid_batches_per_epoch):
            # Transfer to GPU
            local_batch = next(self.valid_loader_iter)
            input_batch, target_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

            generated_batch = self.generator(input_batch)

            # Get the spectrogram
            specgram_target_batch = self.spectrogram(target_batch)
            specgram_generated_batch = self.spectrogram(generated_batch)

            loss_generator_time = self.generator_time_criterion(generated_batch, target_batch)
            batch_losses['time_l2'].append(loss_generator_time.item())
            loss_generator_frequency = self.generator_frequency_criterion(specgram_generated_batch, specgram_target_batch)
            batch_losses['freq_l2'].append(loss_generator_frequency.item())

        # Store the validation losses
        self.valid_losses['time_l2'].append(np.mean(batch_losses['time_l2']))
        self.valid_losses['freq_l2'].append(np.mean(batch_losses['freq_l2']))

        # Display validation losses
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

