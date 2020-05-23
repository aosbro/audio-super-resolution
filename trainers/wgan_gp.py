from models.discriminator import Discriminator
from trainers.base_trainer import Trainer
from models.generator import Generator
from torch.optim import lr_scheduler
from torch import autograd
from torch import nn
import numpy as np
import torch
import os


class WGanTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, general_args, trainer_args):
        super(WGanTrainer, self).__init__(train_loader, test_loader, valid_loader, general_args)
        # Paths
        self.loadpath = trainer_args.loadpath
        self.savepath = trainer_args.savepath

        # Load the generator
        self.generator = Generator(general_args=general_args).to(self.device)
        if trainer_args.generator_path and os.path.exists(trainer_args.generator_path):
            self.load_pretrained_generator(trainer_args.generator_path)

        self.discriminator = Discriminator(general_args=general_args).to(self.device)

        # Optimizers and schedulers
        self.generator_optimizer = torch.optim.RMSprop(params=self.generator.parameters(), lr=trainer_args.generator_lr)
        self.discriminator_optimizer = torch.optim.RMSprop(params=self.discriminator.parameters(),
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
        self.generator_time_criterion = nn.MSELoss()

        # Loss scaling factors
        self.lambda_adv = trainer_args.lambda_adversarial

        # Boolean indicating if the model needs to be saved
        self.need_saving = True

        # Overrides losses from parent class
        self.train_losses = {
            'time_l2': [],
            'generator_adversarial': [],
            'discriminator_adversarial': []
        }
        self.test_losses = {
            'time_l2': [],
            'generator_adversarial': [],
            'discriminator_adversarial': []
        }
        self.valid_losses = {
            'time_l2': [],
            'generator_adversarial': [],
            'discriminator_adversarial': []
        }

        # Select either wgan or wgan-gp method
        self.use_penalty = trainer_args.use_penalty
        self.gamma = trainer_args.gamma_wgan_gp
        self.clipping_limit = trainer_args.clipping_limit
        self.n_critic = trainer_args.n_critic

    def load_pretrained_generator(self, generator_path):
        """
        Loads a pre-trained generator. Can be used to stabilize the training.
        :param generator_path: location of the pre-trained generator (string).
        :return: None
        """
        checkpoint = torch.load(generator_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])

    def compute_gradient_penalty(self, input_batch, generated_batch):
        """
        Compute the gradient penalty as described in the original paper
        (https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf).
        :param input_batch: batch of input data (torch tensor).
        :param generated_batch: batch of generated data (torch tensor).
        :return: penalty as a scalar (torch tensor).
        """
        batch_size = input_batch.size(0)
        epsilon = torch.rand(batch_size, 1, 1)
        epsilon = epsilon.expand_as(input_batch).to(self.device)

        # Interpolate
        interpolation = epsilon * input_batch.data + (1 - epsilon) * generated_batch.data
        interpolation = interpolation.requires_grad_(True).to(self.device)

        # Computes the discriminator's prediction for the interpolated input
        interpolation_logits = self.discriminator(interpolation)

        # Computes a vector of outputs to make it works with 2 output classes if needed
        grad_outputs = torch.ones_like(interpolation_logits).to(self.device).requires_grad_(True)

        # Get the gradients and retain the graph so that the penalty can be back-propagated
        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)

        # Computes the norm of the gradients
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        return ((gradients_norm - 1) ** 2).mean()

    def train_discriminator_step(self, input_batch, target_batch):
        """
        Trains the discriminator for a single step based on the wasserstein gan-gp framework.
        :param input_batch: batch of input data (torch tensor).
        :param target_batch: batch of target data (torch tensor).
        :return: a batch of generated data (torch tensor).
        """
        # Activate gradient tracking for the discriminator
        # self.change_discriminator_grad_requirement(requires_grad=True)

        # Set the discriminator's gradients to zero
        self.discriminator_optimizer.zero_grad()

        # Generate a batch and compute the penalty
        generated_batch = self.generator(input_batch)

        # Compute the loss
        loss_d = self.discriminator(generated_batch.detach()).mean() - self.discriminator(target_batch).mean()
        if self.use_penalty:
            penalty = self.compute_gradient_penalty(input_batch, generated_batch.detach())
            loss_d = loss_d + penalty

        # Update the discriminator's weights
        loss_d.backward()
        self.discriminator_optimizer.step()

        # Apply the weight constraint if needed
        if not self.use_penalty:
            for p in self.discriminator.parameters():
                p.data.clamp_(min=-self.clipping_limit, max=self.clipping_limit)

        # Store the loss
        self.train_losses['discriminator_adversarial'].append(loss_d.item())

        # Return the generated batch to avoid redundant computation
        return generated_batch

    def train_generator_step(self, target_batch, generated_batch):
        """
        Trains the generator for a single step based on the wasserstein gan-gp framework.
        :param target_batch: batch of target data (torch tensor).
        :param generated_batch: batch of generated data (torch tensor).
        :return: None
        """
        # Deactivate gradient tracking for the discriminator
        # self.change_discriminator_grad_requirement(requires_grad=False)

        # Set generator's gradients to zero
        self.generator_optimizer.zero_grad()

        # Get the generator losses
        loss_g_adversarial = - self.discriminator(generated_batch).mean()
        loss_g_time = self.generator_time_criterion(generated_batch, target_batch)

        # Combine the different losses
        loss_g = self.lambda_adv * loss_g_adversarial + loss_g_time

        # Back-propagate and update the generator weights
        loss_g.backward()
        self.generator_optimizer.step()

        # Store the losses
        self.train_losses['generator_adversarial'].append(loss_g_adversarial.item())
        self.train_losses['time_l2'].append(loss_g_time.item())

    def change_discriminator_grad_requirement(self, requires_grad):
        """
        Changes the requires_grad flag of discriminator's parameters. This action is not absolutely needed as the
        discriminator's optimizer is not called after the generators update, but it reduces the computational cost.
        :param requires_grad: flag indicating if the discriminator's parameter require gradient tracking (boolean).
        :return: None
        """
        for p in self.discriminator.parameters():
            p.requires_grad_(requires_grad)

    def train(self, epochs):
        """
        Trains the WGAN-GP for a given number of pseudo-epochs.
        :param epochs: Number of time to iterate over a part of the dataset (int).
        :return: None
        """
        self.generator.train()
        self.discriminator.train()
        for epoch in range(epochs):
            for i in range(self.train_batches_per_epoch):
                # Transfer to GPU
                local_batch = next(self.train_loader_iter)
                input_batch, target_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

                # Train the discriminator
                generated_batch = self.train_discriminator_step(input_batch, target_batch)

                # Train the generator every n_critic
                if not (i % self.n_critic):
                    self.train_generator_step(target_batch, generated_batch)

                # Print message
                if not (i % 10):
                    message = 'Batch {}: \n' \
                              '\t Generator: \n' \
                              '\t\t Time: {} \n' \
                              '\t\t Adversarial: {} \n' \
                              '\t Discriminator: \n' \
                              '\t\t Adversarial {} \n'.format(i,
                                                              self.train_losses['time_l2'][-1],
                                                              self.train_losses['generator_adversarial'][-1],
                                                              self.train_losses['discriminator_adversarial'][-1])
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

            loss_g_time = self.generator_time_criterion(generated_batch, target_batch)
            batch_losses['time_l2'].append(loss_g_time.item())

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

