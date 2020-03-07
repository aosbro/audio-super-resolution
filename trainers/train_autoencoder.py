from datasets.datasets import *
from models.autoencoder import *


def train_autoencoder(model, epochs, train_generator, test_generator, optimizer, loss, device):
    for epoch in range(epochs):
        model.train()
        for local_batch in train_generator:
            # Transfer to GPU
            local_batch = torch.cat(local_batch).to(device)
            optimizer.zero_grad()

            # Forward pass
            x_tilde, _ = model.forward(local_batch)

            # backward pass
            loss.backward()
            optimizer.step()


def main():
    # Define the datapaths
    train_datapath = '/media/thomas/Samsung_T5/VITA/data/music/music_train_.npy'
    test_datapath =  '/media/thomas/Samsung_T5/VITA/data/music/music_test_.npy'

    # Create the datasets
    train_dataset = DatasetBeethoven(train_datapath)
    test_dataset = DatasetBeethoven(test_datapath)

    # Create the generators
    train_params = {'batch_size': 10,
                    'shuffle': True,
                    'num_workers': 6}
    test_params = {'batch_size': 10,
                   'shuffle': False,
                   'num_workers': 6}
    train_generator = data.DataLoader(train_dataset, **train_params)
    test_generator = data.DataLoader(test_dataset, **test_params)

    # Load the model
    model = AutoEncoder(kernel_sizes=kernel_sizes,
                        channel_sizes=channel_sizes,
                        bottleneck_channels=bottleneck_channels,
                        p=p,
                        n_blocks=n_block)

    optimizer = torch.optim.Adam()
    loss = nn.MSELoss()


if __name__ == '__main__':
    main()