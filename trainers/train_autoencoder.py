from models.autoencoder import *
from utils.utils import *
import os
from trainers.base_trainer import *
from torch.optim import lr_scheduler


class AutoEncoderTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, lr, loadpath, savepath):
        super(AutoEncoderTrainer, self).__init__(train_loader, test_loader, valid_loader, loadpath, savepath)

        # Model
        self.autoencoder = AutoEncoder(kernel_sizes=KERNEL_SIZES,
                                       channel_sizes_min=CHANNEL_SIZES_MIN,
                                       p=DROPOUT_PROBABILITY,
                                       n_blocks=N_BLOCKS_AUTOENCODER).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.autoencoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1000, gamma=0.3)

        # Load saved states
        if os.path.exists(self.loadpath):
            self.load()

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

        self.is_autoencoder = True

    def train(self, epochs):
        for epoch in range(epochs):
            self.autoencoder.train()
            for i, local_batch in enumerate(self.train_loader):
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                fake_batch, _ = self.autoencoder.forward(local_batch)

                # Get the spectrogram
                local_batch_freq = self.spectrogram(local_batch)
                fake_batch_freq = self.spectrogram(fake_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(fake_batch, local_batch)
                freq_l2_loss = self.frequency_criterion(fake_batch_freq, local_batch_freq)
                self.train_losses['time_l2'].append(time_l2_loss.item())
                self.train_losses['freq_l2'].append(freq_l2_loss.item())
                loss = time_l2_loss + freq_l2_loss

                # Print message
                if not(i % 10):
                    message = 'Batch {}, time l2: {}, freq l2: {}'.format(i, time_l2_loss.item(), freq_l2_loss.item())
                    print(message)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Increment epoch counter
            self.epoch += 1

            # Save the trainer state
            self.save()

    def eval(self):
        with torch.no_grad():
            self.autoencoder.eval()
            batch_losses = []
            for i, local_batch in enumerate(self.test_loader):
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)

                # Forward pass
                x_tilde, _ = self.autoencoder.forward(local_batch)
                loss = self.time_criterion(input=x_tilde, target=local_batch)

                # Store the batch loss
                batch_losses.append(loss.item())

                # Print message
                if not(i % 100):
                    message = 'Batch {}, test loss: {}'.format(i, np.mean(batch_losses[-100:]))
                    print(message)

            # Add the current epoch's average mean to the train losses
            self.test_losses.append(np.mean(batch_losses))

    def save(self):
        """
        Saves the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        torch.save({
            'epoch': self.epoch,
            'generator_state_dict': self.autoencoder.state_dict(),
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
        self.autoencoder.load_state_dict(checkpoint['generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.valid_losses = checkpoint['valid_losses']


def create_autoencoder(train_datapath, test_datapath, valid_datapath, loadpath, savepath, batch_size):
    # Create the datasets
    train_loader, test_loader, valid_loader = get_the_data_loaders(train_datapath, test_datapath, valid_datapath,
                                                                   batch_size)

    autoencoder_trainer = AutoEncoderTrainer(train_loader=train_loader,
                                             test_loader=test_loader,
                                             valid_loader=valid_loader,
                                             lr=LEARNING_RATE,
                                             loadpath=loadpath,
                                             savepath=savepath)
    return autoencoder_trainer


def train_autoencoder(train_datapath, test_datapath, valid_datapath, loadpath, savepath, epochs, batch_size):
    autoencoder_trainer = create_autoencoder(train_datapath=train_datapath,
                                             test_datapath=test_datapath,
                                             valid_datapath=valid_datapath,
                                             loadpath=loadpath,
                                             savepath=savepath,
                                             batch_size=batch_size)

    # Start training
    autoencoder_trainer.train(epochs=epochs)
    return autoencoder_trainer


if __name__ == '__main__':
    train_autoencoder(train_datapath=TRAIN_DATAPATH,
                      test_datapath=TEST_DATAPATH,
                      valid_datapath=VALID_DATAPATH,
                      loadpath=AUTOENCODER_PATH,
                      savepath=AUTOENCODER_PATH,
                      epochs=1,
                      batch_size=16)
