import os
from utils.utils import *
from datasets.datasets import *
from models.generator import *
from models.discriminator import *
from trainers.base_trainer import *


class GanTrainer(Trainer):
    def __init__(self, generator, discriminator, train_loader, test_loader, valid_loader, lr, savepath):
        super(GanTrainer, self).__init__(train_loader, test_loader, valid_loader, savepath)

        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=lr)

        # Loss function and stored losses
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.generator_time_criterion = nn.MSELoss()
        self.generator_frequency_criterion = nn.MSELoss()

        # Define labels
        self.real_label = 1
        self.fake_label = 0

        # Loss scaling factors
        self.lambda_adv = 1e-3

        # Spectrogram converter
        self.spectrogram = Spectrogram(normalized=True).to(self.device)

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

                # Get the spectrogram
                specgram_h_batch = self.spectrogram(x_h_batch)
                specgram_fake_batch = self.spectrogram(fake_batch)

                # Fake labels are real for the generator cost
                label.fill_(self.real_label)
                output = self.discriminator(fake_batch)
                loss_generator = self.lambda_adv * self.adversarial_criterion(torch.squeeze(output), label) + \
                                 self.generator_time_criterion(fake_batch, x_h_batch) +\
                                 self.generator_frequency_criterion(specgram_fake_batch, specgram_h_batch)

                loss_generator.backward()

                self.generator_optimizer.step()

                # Print message
                if not (i % 1):
                    message = 'Batch {}, train loss: {}, {}'.format(i, loss_discriminator.item(), loss_generator.item())
                    print(message)

            # Increment epoch counter
            self.epoch += 1

    def eval(self, epoch):
        pass


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

    train_loader = data.DataLoader(train_dataset, **train_params)
    test_loader = data.DataLoader(test_dataset, **test_params)
    valid_loader = data.DataLoader(valid_dataset, **valid_params)

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
                             train_loader=train_loader,
                             test_loader=test_loader,
                             valid_loader=valid_loader,
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
                            gan_savepath=GAN_PATH,
                            epochs=1,
                            batch_size=16)

    t = next(iter(gan_trainer.train_loader))
    print(t[1].shape)
