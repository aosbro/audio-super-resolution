from datasets.datasets import *
from models.autoencoder import *
from utils.utils import *


class AutoEncoderTrainer:
    def __init__(self, model, train_generator, test_generator, valid_generator, lr, savepath):
        # Model
        self.model = model

        # Data generators
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_generator = valid_generator

        # Optimizer
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        # Device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

        # Loss function and stored losses
        self.loss_function = nn.MSELoss()
        self.train_losses = []
        self.test_losses = []
        self.valid_losses = []

        # Path to save to the class
        self.savepath = savepath

    def save(self):
        f = open(self.savepath, 'wb')
        f.write(pickle.dumps(self))
        f.close()

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for local_batch in self.train_generator:
                # Transfer to GPU
                local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                x_tilde, _ = self.model.forward(local_batch)
                loss = self.loss_function(input=x_tilde, target=local_batch)
                print(loss.item())

                # Backward pass
                loss.backward()
                self.optimizer.step()

    def eval(self):
        self.model.eval()
        for local_batch in self.test_generator:
            # Transfer to GPU
            local_batch = torch.cat(local_batch).to(self.device)

            # Forward pass
            x_tilde, _ = self.model.forward(local_batch)
            loss = self.loss_function(input=x_tilde, target=local_batch)
            print(loss.item())


def main(train_datapath, test_datapath, valid_datapath):
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

    # Load the model
    model = AutoEncoder(kernel_sizes=KERNEL_SIZES,
                        channel_sizes=CHANNEL_SIZES,
                        bottleneck_channels=BOTTLENECK_CHANNELS,
                        p=DROPOUT_PROBABILITY,
                        n_blocks=N_BLOCKS)

    # Define the optimizer, loss and device
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to save the trainer
    savepath = '../objects/autoencoder_trainer.txt'

    autoencoder_trainer = AutoEncoderTrainer(model=model,
                                             train_generator=train_generator,
                                             test_generator=test_generator,
                                             valid_generator=valid_generator,
                                             lr=1e-3,
                                             savepath=savepath)


    autoencoder_trainer.save()

    autoencoder_trainer_2 = load_class(savepath)
    autoencoder_trainer_2.train(1)


if __name__ == '__main__':
    # Define the datapaths
    train_datapath = TRAIN_DATAPATH
    test_datapath = TEST_DATAPATH
    valid_datapath = VALID_DATAPATH

    main(train_datapath=train_datapath, test_datapath=test_datapath, valid_datapath=valid_datapath)