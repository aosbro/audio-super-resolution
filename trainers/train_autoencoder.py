from datasets.datasets import *
from models.autoencoder import *
from utils.utils import *
import os


class AutoEncoderTrainer:
    def __init__(self, autoencoder, train_generator, test_generator, valid_generator, lr, savepath):
        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.autoencoder = autoencoder.to(self.device)

        # Data generators
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_generator = valid_generator

        # Optimizer
        self.optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=lr)

        # Loss function and stored losses
        self.loss_function = nn.MSELoss()
        self.train_losses = []
        self.test_losses = []
        self.valid_losses = []

        # Path to save to the class
        self.savepath = savepath

        # Epoch counter
        self.epoch_counter = 0

    def save(self):
        f = open(self.savepath, 'wb')
        f.write(pickle.dumps(self))
        f.close()

    def train(self, epochs):
        for epoch in range(epochs):
            self.autoencoder.train()
            batch_losses = []
            for i, local_batch in enumerate(self.train_generator):
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                x_tilde, _ = self.autoencoder.forward(local_batch)
                loss = self.loss_function(input=x_tilde, target=local_batch)

                # Print message
                if not(i % 100):
                    message = 'Batch {}, train loss: {}'.format(i, np.mean(batch_losses[-100:]))
                    print(message)

                # Store the batch loss
                batch_losses.append(loss.item())

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Add the current epoch's average mean to the train losses
            self.train_losses.append(np.mean(batch_losses))

            # Evaluate
            self.eval()

            # Increment epoch counter
            self.epoch_counter += 1

    def eval(self):
        self.autoencoder.eval()
        batch_losses = []
        for i, local_batch in enumerate(self.test_generator):
            # Transfer to GPU
            local_batch = torch.cat(local_batch).to(self.device)

            # Forward pass
            x_tilde, _ = self.autoencoder.forward(local_batch)
            loss = self.loss_function(input=x_tilde, target=local_batch)

            # Print message
            if not(i % 100):
                message = 'Batch {}, test loss: {}'.format(i, np.mean(batch_losses[-100:]))
                print(message)

            batch_losses.append(loss.item())

        # Add the current epoch's average mean to the train losses
        self.test_losses.append(np.mean(batch_losses))


def create_autoencoder(train_datapath, test_datapath, valid_datapath, savepath):
    # Create the datasets
    train_dataset = DatasetBeethoven(train_datapath)
    test_dataset = DatasetBeethoven(test_datapath)
    valid_dataset = DatasetBeethoven(valid_datapath)

    # Create the generators
    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': TRAIN_SHUFFLE,
                    'num_workers': NUM_WORKERS}
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': TEST_SHUFFLE,
                   'num_workers': NUM_WORKERS}
    valid_params = {'batch_size': BATCH_SIZE,
                    'shuffle': VALID_SHUFFLE,
                    'num_workers': NUM_WORKERS}

    train_generator = data.DataLoader(train_dataset, **train_params)
    test_generator = data.DataLoader(test_dataset, **test_params)
    valid_generator = data.DataLoader(valid_dataset, **valid_params)

    # Load the autoencoder
    model = AutoEncoder(kernel_sizes=KERNEL_SIZES,
                        channel_sizes=CHANNEL_SIZES,
                        bottleneck_channels=BOTTLENECK_CHANNELS,
                        p=DROPOUT_PROBABILITY,
                        n_blocks=N_BLOCKS)


    autoencoder_trainer = AutoEncoderTrainer(autoencoder=model,
                                             train_generator=train_generator,
                                             test_generator=test_generator,
                                             valid_generator=valid_generator,
                                             lr=LEARNING_RATE,
                                             savepath=savepath)
    return autoencoder_trainer


def main(train_datapath, test_datapath, valid_datapath, savepath, epochs):
    # Get the trainer
    if os.path.exists(savepath):
        autoencoder_trainer = load_class(loadpath=savepath)
    else:
        autoencoder_trainer = create_autoencoder(train_datapath=train_datapath,
                                                 test_datapath=test_datapath,
                                                 valid_datapath=valid_datapath,
                                                 savepath=savepath)

    # Start training
    autoencoder_trainer.train(epochs=epochs)
    return autoencoder_trainer


if __name__ == '__main__':
    main(train_datapath=TRAIN_DATAPATH,
         test_datapath=TEST_DATAPATH,
         valid_datapath=VALID_DATAPATH,
         savepath=AUTOENCODER_SAVEPATH)