import os
from utils.utils import *
from datasets.datasets import *
from models.generator import *
from models.discriminator import *


class GanTrainer:
    def __init__(self, generator, discriminator, train_generator, test_generator, valid_generator, lr, savepath):
        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # Data generators
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_generator = valid_generator

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=lr)

        # Loss function and stored losses
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        # self.loss_function = nn.MSELoss()
        # self.train_losses = []
        # self.test_losses = []
        # self.valid_losses = []

        # Path to save to the class
        self.savepath = savepath

        # Epoch counter
        self.epoch_counter = 0

        # Define labels
        self.real_label = 1
        self.fake_label = 0

    def save(self):
        """
        Saves the complete trainer class
        :return: None
        """
        torch.save(self, self.savepath)

    def train(self, epochs):
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            for i, local_batch in enumerate(self.train_generator):
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
                loss_discriminator_real = self.adversarial_criterion(torch.squeeze(output), label)
                loss_discriminator_real.backward()

                # Train the discriminator with fake data
                fake_batch = self.generator(x_l_batch)
                label.fill_(self.fake_label)
                output = self.discriminator(fake_batch.detach())
                loss_discriminator_fake = self.adversarial_criterion(torch.squeeze(output), label)
                loss_discriminator_fake.backward()

                loss_discriminator = loss_discriminator_real + loss_discriminator_fake
                self.discriminator_optimizer.step()

                ############################
                # Update G network: maximize log(D(G(z)))
                ###########################
                self.generator_optimizer.zero_grad()

                # Fake labels are real for the generator cost
                label.fill_(self.real_label)
                output = self.discriminator(fake_batch)
                loss_generator = self.adversarial_criterion(torch.squeeze(output), label)
                loss_generator.backward()

                self.generator_optimizer.step()

                # Print message
                if not(i % 1):
                    message = 'Batch {}, train loss: {}, {}'.format(i, loss_discriminator.item(), loss_generator.item())
                    print(message)

            # Increment epoch counter
            self.epoch_counter += 1

    # def eval(self):
    #     with torch.no_grad():
    #         self.autoencoder.eval()
    #         batch_losses = []
    #         for i, local_batch in enumerate(self.test_generator):
    #             # Transfer to GPU
    #             local_batch = torch.cat(local_batch).to(self.device)
    #
    #             # Forward pass
    #             x_tilde, _ = self.autoencoder.forward(local_batch)
    #             loss = self.loss_function(input=x_tilde, target=local_batch)
    #
    #             # Store the batch loss
    #             batch_losses.append(loss.item())
    #
    #             # Print message
    #             if not(i % 100):
    #                 message = 'Batch {}, test loss: {}'.format(i, np.mean(batch_losses[-100:]))
    #                 print(message)
    #
    #         # Add the current epoch's average mean to the train losses
    #         self.test_losses.append(np.mean(batch_losses))


def create_gan(train_datapath, test_datapath, valid_datapath, savepath, batch_size):
    # Create the datasets
    train_dataset = DatasetBeethoven(train_datapath)
    test_dataset = DatasetBeethoven(test_datapath)
    valid_dataset = DatasetBeethoven(valid_datapath)

    # Create the generators
    train_params = {'batch_size': batch_size,
                    'shuffle': TRAIN_SHUFFLE,
                    'num_workers': NUM_WORKERS}
    test_params = {'batch_size': batch_size,
                   'shuffle': TEST_SHUFFLE,
                   'num_workers': NUM_WORKERS}
    valid_params = {'batch_size': batch_size,
                    'shuffle': VALID_SHUFFLE,
                    'num_workers': NUM_WORKERS}

    train_generator = data.DataLoader(train_dataset, **train_params)
    test_generator = data.DataLoader(test_dataset, **test_params)
    valid_generator = data.DataLoader(valid_dataset, **valid_params)

    # Load the models
    generator = Generator(kernel_sizes=KERNEL_SIZES,
                          channel_sizes_min=CHANNEL_SIZES_MIN,
                          p=DROPOUT_PROBABILITY,
                          n_blocks=N_BLOCKS_GENERATOR)
    discriminator = Discriminator(kernel_sizes=KERNEL_SIZES,
                                  channel_sizes_min=CHANNEL_SIZES_MIN,
                                  p=DROPOUT_PROBABILITY,
                                  n_blocks=N_BLOCKS_DISCRIMINATOR)

    gan_trainer = GanTrainer(generator=generator,
                             discriminator=discriminator,
                             train_generator=train_generator,
                             test_generator=test_generator,
                             valid_generator=valid_generator,
                             lr=LEARNING_RATE,
                             savepath=savepath)
    return gan_trainer


def train_gan(train_datapath, test_datapath, valid_datapath, gan_savepath, epochs, batch_size):
    # Get the trainer
    if os.path.exists(gan_savepath):
        gan_trainer = load_class(loadpath=gan_savepath)
    else:
        gan_trainer = create_gan(train_datapath=train_datapath,
                                 test_datapath=test_datapath,
                                 valid_datapath=valid_datapath,
                                 savepath=gan_savepath,
                                 batch_size=batch_size)

    # Start training
    gan_trainer.train(epochs=epochs)
    return gan_trainer


if __name__ == '__main__':
    gan_trainer = train_gan(train_datapath=TRAIN_DATAPATH,
                            test_datapath=TEST_DATAPATH,
                            valid_datapath=VALID_DATAPATH,
                            gan_savepath=GAN_SAVEPATH,
                            epochs=1,
                            batch_size=16)

    t = next(iter(gan_trainer.train_generator))
    print(t[1].shape)