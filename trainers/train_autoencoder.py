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
        """
        Saves the complete trainer class
        :return: None
        """
        f = open(self.savepath, 'wb')
        f.write(pickle.dumps(self))
        f.close()

    def plot_reconstruction_time_domain(self, index):
        batch_size = self.test_generator.batch_size
        index = index % batch_size
        self.autoencoder.eval()
        test_input = torch.cat(next(iter(self.test_generator)))
        test_output, test_phi = self.autoencoder(test_input.to(self.device))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[0, 0].plot(test_input[index].cpu().detach().numpy().squeeze())
        axes[0, 0].set_title('Original, high quality', fontsize=16)
        axes[0, 1].plot(test_output[index].cpu().detach().numpy().squeeze())
        axes[0, 1].set_title('Reconstruction, high quality', fontsize=16)
        axes[1, 0].plot(test_input[index + batch_size].cpu().detach().numpy().squeeze())
        axes[1, 0].set_title('Original, low quality', fontsize=16)
        axes[1, 1].plot(test_output[index + batch_size].cpu().detach().numpy().squeeze())
        axes[1, 1].set_title('Reconstruction, low quality', fontsize=16)
        plt.show()

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
            self.train_losses.append(np.mean(batch_losses))

            # Evaluate
            self.eval()

            # Increment epoch counter
            self.epoch_counter += 1

    def eval(self):
        with torch.no_grad():
            self.autoencoder.eval()
            batch_losses = []
            for i, local_batch in enumerate(self.test_generator):
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

    train_generator = data.DataLoader(train_dataset, **train_params)
    test_generator = data.DataLoader(test_dataset, **test_params)
    valid_generator = data.DataLoader(valid_dataset, **valid_params)

    # Load the autoencoder
    model = AutoEncoder(kernel_sizes=KERNEL_SIZES,
                        channel_sizes_min=CHANNEL_SIZES_MIN,
                        p=DROPOUT_PROBABILITY,
                        n_blocks=N_BLOCKS_AUTOENCODER)

    autoencoder_trainer = AutoEncoderTrainer(autoencoder=model,
                                             train_generator=train_generator,
                                             test_generator=test_generator,
                                             valid_generator=valid_generator,
                                             lr=LEARNING_RATE,
                                             savepath=savepath)
    return autoencoder_trainer


def main(train_datapath, test_datapath, valid_datapath, savepath, epochs, batch_size):
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
    main(train_datapath=TRAIN_DATAPATH,
         test_datapath=TEST_DATAPATH,
         valid_datapath=VALID_DATAPATH,
         savepath=AUTOENCODER_SAVEPATH,
         epochs=1)