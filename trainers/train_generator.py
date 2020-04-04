from trainers.base_trainer import *
from utils.utils import *
from models.generator import *
import os
from torch.optim import lr_scheduler


class GeneratorTrainer(Trainer):
    def __init__(self, generator, train_loader, test_loader, valid_loader, lr, savepath):
        super(GeneratorTrainer, self).__init__(train_loader, test_loader, valid_loader, savepath)
        # Model
        self.generator = generator.to(self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam(params=generator.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.3)

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

        # Time to frequency converter
        self.spectrogram = Spectrogram(normalized=True, n_fft=512, hop_length=128)

    def train(self, epochs):
        """
        Trains the model for a specified number of epochs on the train dataset
        :param epochs: Number of iterations over the complete dataset to perform
        :return: None
        """
        for epoch in range(epochs):
            self.generator.train()
            for i, local_batch in enumerate(self.train_loader):

                # Transfer to GPU
                x_h_batch, x_l_batch = local_batch[0].to(self.device), local_batch[1].to(self.device)

                # Reset all gradients in the graph
                self.optimizer.zero_grad()

                # Generates a fake batch
                fake_batch = self.generator(x_l_batch)

                # Get the spectrogram
                x_h_batch_freq = self.spectrogram(x_h_batch)
                fake_batch_freq = self.spectrogram(fake_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(fake_batch, x_h_batch)
                freq_l2_loss = self.frequency_criterion(fake_batch_freq, x_h_batch_freq)
                self.train_losses['time_l2'].append(time_l2_loss.item())
                self.train_losses['freq_l2'].append(freq_l2_loss.item())
                loss = time_l2_loss + freq_l2_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Print message
                if not (i % 10):
                    message = 'Batch {}, time l2: {}, freq l2: {}'.format(i, time_l2_loss.item(), freq_l2_loss.item())
                    print(message)

            # Increment epoch counter
            self.epoch_counter += 1

    def eval(self, epoch):
        """
        Evaluates the model on the test dataset
        :param epoch: Current epoch, used to print status information
        :return: None
        """
        pass


def create_genarator(train_datapath, test_datapath, valid_datapath, savepath, batch_size):
    # Create the datasets
    train_loader, test_loader, valid_loader = get_the_data_loaders(train_datapath, test_datapath, valid_datapath,
                                                                   batch_size)

    # Load the models
    generator = Generator(kernel_sizes=KERNEL_SIZES,
                          channel_sizes_min=CHANNEL_SIZES_MIN,
                          p=DROPOUT_PROBABILITY,
                          n_blocks=N_BLOCKS_GENERATOR)

    generator_trainer = GeneratorTrainer(generator=generator,
                                         train_loader=train_loader,
                                         test_loader=test_loader,
                                         valid_loader=valid_loader,
                                         lr=LEARNING_RATE,
                                         savepath=savepath)
    return generator_trainer


def train_generator(train_datapath, test_datapath, valid_datapath, generator_savepath, epochs, batch_size):
    # Get the trainer
    if os.path.exists(generator_savepath):
        generator_trainer = load_class(loadpath=generator_savepath)
    else:
        generator_trainer = create_genarator(train_datapath=train_datapath,
                                             test_datapath=test_datapath,
                                             valid_datapath=valid_datapath,
                                             savepath=generator_savepath,
                                             batch_size=batch_size)

    # Start training
    generator_trainer.train(epochs=epochs)
    return generator_trainer


if __name__ == '__main__':
    generator_trainer = train_generator(train_datapath=TRAIN_DATAPATH,
                                        test_datapath=TEST_DATAPATH,
                                        valid_datapath=VALID_DATAPATH,
                                        generator_savepath=GAN_SAVEPATH,
                                        epochs=1,
                                        batch_size=4)

