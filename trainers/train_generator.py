from trainers.base_trainer import *
from utils.utils import *
from models.generator import *
import os
from torch.optim import lr_scheduler


class GeneratorTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, lr, loadpath, savepath):
        super(GeneratorTrainer, self).__init__(train_loader, test_loader, valid_loader, loadpath, savepath)
        # Model
        self.generator = Generator(kernel_sizes=KERNEL_SIZES,
                                   channel_sizes_min=CHANNEL_SIZES_MIN,
                                   p=DROPOUT_PROBABILITY,
                                   n_blocks=N_BLOCKS_GENERATOR).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.generator.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1000, gamma=0.3)

        # Load saved states
        if os.path.exists(self.loadpath):
            self.load()

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

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
                # loss = time_l2_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Print message
                if not (i % 10):
                    message = 'Batch {}, time l2: {}, freq l2: {}'.format(i, time_l2_loss.item(), freq_l2_loss.item())
                    # message = 'Batch {}, time l2: {}'.format(i, time_l2_loss.item())
                    print(message)

            # Increment epoch counter
            self.epoch += 1

            # Save the trainer state
            self.save()

    def eval(self, epoch):
        """
        Evaluates the model on the test dataset
        :param epoch: Current epoch, used to print status information
        :return: None
        """
        pass

    def save(self):
        """
        Saves the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        torch.save({
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'valid_losses': self.valid_losses
        }, self.savepath)

    def load(self):
        """
        Loads the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        checkpoint = torch.load(self.loadpath)
        self.epoch = checkpoint['epoch']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.valid_losses = checkpoint['valid_losses']


def create_genarator(train_datapath, test_datapath, valid_datapath, loadpath, savepath, batch_size):
    # Create the datasets
    train_loader, test_loader, valid_loader = get_the_data_loaders(train_datapath, test_datapath, valid_datapath,
                                                                   batch_size)

    generator_trainer = GeneratorTrainer(train_loader=train_loader,
                                         test_loader=test_loader,
                                         valid_loader=valid_loader,
                                         lr=LEARNING_RATE,
                                         loadpath=loadpath,
                                         savepath=savepath)
    return generator_trainer


def train_generator(train_datapath, test_datapath, valid_datapath, loadpath, savepath, epochs, batch_size):
    # Instantiate the trainer class
    generator_trainer = create_genarator(train_datapath=train_datapath,
                                         test_datapath=test_datapath,
                                         valid_datapath=valid_datapath,
                                         loadpath=loadpath,
                                         savepath=savepath,
                                         batch_size=batch_size)

    # Start training
    generator_trainer.train(epochs)
    return generator_trainer


if __name__ == '__main__':
    generator_trainer = train_generator(train_datapath=TRAIN_DATAPATH,
                                        test_datapath=TEST_DATAPATH,
                                        valid_datapath=VALID_DATAPATH,
                                        loadpath=GENERATOR_L2T_PATH,
                                        savepath=GENERATOR_L2T_PATH,
                                        epochs=1,
                                        batch_size=4)

