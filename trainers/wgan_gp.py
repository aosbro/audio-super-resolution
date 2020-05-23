from models.discriminator import Discriminator
from torchaudio.transforms import Spectrogram
from trainers.base_trainer import Trainer
from models.generator import Generator
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch import autograd
from torch import nn
import numpy as np
import torch
import os


class WGanGPTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, general_args, trainer_args):
        super(WGanGPTrainer, self).__init__(train_loader, test_loader, valid_loader, general_args)
        # Paths
        self.loadpath = trainer_args.loadpath
        self.savepath = trainer_args.savepath

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

        # Define labels
        self.real_label = 1
        self.generated_label = 0

        # Loss scaling factors
        self.lambda_adv = trainer_args.lambda_adversarial

        # Spectrogram converter
        self.spectrogram = Spectrogram(normalized=True).to(self.device)

        # Boolean indicating if the model needs to be saved
        self.need_saving = True

        self.gamma = 10

    def load_pretrained_generator(self, generator_path):
        """
        Loads a pre-trained generator. Can be used to stabilize the training.
        :param generator_path: location of the pre-trained generator (string).
        :return: None
        """
        checkpoint = torch.load(generator_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])

    def compute_gradient_penalty(self, input_batch, generated_batch):
        batch_size = input_batch.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand_as(input_batch).to(self.device)

        # Interpolate
        interpolation = epsilon * input_batch.data + (1 - epsilon) * generated_batch.data
        interpolation = interpolation.requires_grad_().to(self.device)

        # Computes the discriminator's prediction for the interpolated input
        interpolation_logits = self.discriminator(interpolation)

        # Computes a vector of outputs to make it works with 2 output classes if needed
        grad_outputs = torch.ones(interpolation_logits.size()).to(self.device)

        # Get the gradients
        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)

        # Computes the norm of the gradients and return the penalty
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def train_discriminator_step(self, input_batch, target_batch):
        # Set discriminator's gradients to zero
        batch_size = input_batch.shape[0]
        self.discriminator_optimizer.zero_grad()

        # Train the discriminator with real data
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

        # Return the generated batch and labels to avoid redundant computations
        return generated_batch, label

    def train_generator_step(self, target_batch, generated_batch, label):
        # Set generator's gradients to zero
        self.generator_optimizer.zero_grad()

        # Fake labels are real for the generator cost
        label.fill_(self.real_label)
        output = self.discriminator(generated_batch)

        # Compute the generator loss on fake data
        # Get the adversarial loss
        loss_generator_adversarial = self.adversarial_criterion(output, torch.unsqueeze(label, dim=1))
        self.train_losses['generator_adversarial'].append(loss_generator_adversarial.item())

        # Get the L2 loss in time domain
        loss_generator_time = self.generator_time_criterion(generated_batch, target_batch)
        self.train_losses['time_l2'].append(loss_generator_time.item())

        # Combine the different losses
        loss_generator = self.lambda_adv * loss_generator_adversarial + loss_generator_time

        # Back-propagate and update the generator weights
        loss_generator.backward()
        self.generator_optimizer.step()

    def train(self, epochs):
        """
        Trains the GAN for a given number of pseudo-epochs.
        :param epochs: Number of time to iterate over a part of the dataset (int).
        :return: None
        """
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            for i in range(self.train_batches_per_epoch):
                # Transfer to GPU
                local_batch = next(self.train_loader_iter)
                input_batch, target_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

                # Train the discriminator
                generated_batch, label = self.train_discriminator_step(input_batch, target_batch)

                # Train the generator
                self.train_generator_step(target_batch, generated_batch, label)

                # Print message
                if not (i % 10):
                    message = 'Batch {}: \n' \
                              '\t Genarator: \n' \
                              '\t\t Time: {} \n' \
                              '\t\t Adversarial: {} \n' \
                              '\t Discriminator: \n' \
                              '\t\t Real {} \n' \
                              '\t\t Fake {} \n'.format(i,
                                                       self.train_losses['time_l2'][-1],
                                                       self.train_losses['generator_adversarial'][-1],
                                                       self.train_losses['discriminator_adversarial']['real'][-1],
                                                       self.train_losses['discriminator_adversarial']['fake'][-1])
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
        # Set the models in evaluation mode
        self.generator.eval()
        self.discriminator.eval()
        batch_losses = {'time_l2': []}
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

        # Store the validation losses
        self.valid_losses['time_l2'].append(np.mean(batch_losses['time_l2']))

        # Display validation losses
        message = 'Epoch {}: \n' \
                  '\t Time: {} \n'.format(self.epoch, np.mean(np.mean(batch_losses['time_l2'])))
        print(message)

        # Check if the loss is decreasing
        self.check_improvement()

        # Set the models in train mode
        self.generator.train()
        self.discriminator.eval()

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

