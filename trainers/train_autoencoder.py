from datasets.datasets import *
from models.autoencoder import *
from utils.utils import *
import os
from trainers.base_trainer import *


class AutoEncoderTrainer(Trainer):
    def __init__(self, autoencoder, train_loader, test_loader, valid_loader, lr, savepath):
        super(AutoEncoderTrainer, self).__init__(train_loader, test_loader, valid_loader, savepath)

        # Model
        self.autoencoder = autoencoder.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=lr)

        # Loss function
        self.loss_function = nn.MSELoss()

    def train(self, epochs):
        for epoch in range(epochs):
            self.autoencoder.train()
            batch_losses = []
            for i, local_batch in enumerate(self.train_loader):
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                x_tilde, _ = self.autoencoder.forward(local_batch)
                loss = self.loss_function(input=x_tilde, target=local_batch)

                # Store the batch loss
                batch_losses.append(loss.item())

                # Print message
                if not(i % 1):
                    message = 'Batch {}, train loss: {}'.format(i, np.mean(batch_losses[-1:]))
                    print(message)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Add the current epoch's average mean to the train losses
            self.train_time_losse.append(np.mean(batch_losses))

            # Evaluate
            self.eval()

            # Increment epoch counter
            self.epoch += 1

    def eval(self):
        with torch.no_grad():
            self.autoencoder.eval()
            batch_losses = []
            for i, local_batch in enumerate(self.test_loader):
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)

                # Forward pass
                x_tilde, _ = self.autoencoder.forward(local_batch)
                loss = self.loss_function(input=x_tilde, target=local_batch)

                # Store the batch loss
                batch_losses.append(loss.item())

                # Print message
                if not(i % 100):
                    message = 'Batch {}, test loss: {}'.format(i, np.mean(batch_losses[-100:]))
                    print(message)

            # Add the current epoch's average mean to the train losses
            self.test_losses.append(np.mean(batch_losses))


def create_autoencoder(train_datapath, test_datapath, valid_datapath, savepath, batch_size):
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

    # Load the autoencoder
    model = AutoEncoder(kernel_sizes=KERNEL_SIZES,
                        channel_sizes_min=CHANNEL_SIZES_MIN,
                        p=DROPOUT_PROBABILITY,
                        n_blocks=N_BLOCKS_AUTOENCODER)

    autoencoder_trainer = AutoEncoderTrainer(autoencoder=model,
                                             train_loader=train_loader,
                                             test_loader=test_loader,
                                             valid_loader=valid_loader,
                                             lr=LEARNING_RATE,
                                             savepath=savepath)
    return autoencoder_trainer


def train_autoencoder(train_datapath, test_datapath, valid_datapath, savepath, epochs, batch_size):
    # Get the trainer
    if os.path.exists(savepath):
        autoencoder_trainer = load_class(loadpath=savepath)
    else:
        autoencoder_trainer = create_autoencoder(train_datapath=train_datapath,
                                                 test_datapath=test_datapath,
                                                 valid_datapath=valid_datapath,
                                                 savepath=savepath,
                                                 batch_size=batch_size)

    # Start training
    autoencoder_trainer.train(epochs=epochs)
    return autoencoder_trainer


if __name__ == '__main__':
    train_autoencoder(train_datapath=TRAIN_DATAPATH,
                      test_datapath=TEST_DATAPATH,
                      valid_datapath=VALID_DATAPATH,
                      savepath=AUTOENCODER_SAVEPATH,
                      epochs=1,
                      batch_size=16)