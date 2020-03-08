from datasets.datasets import *
from models.autoencoder import *


def train_autoencoder(model, epochs, train_generator, test_generator, optimizer, loss_function, device):
    model.train()
    for epoch in range(epochs):
        for local_batch in train_generator:
            # Transfer to GPU
            local_batch = torch.cat(local_batch).to(device)
            optimizer.zero_grad()

            # Forward pass
            x_tilde, _ = model.forward(local_batch)
            loss = loss_function(input=x_tilde, target=local_batch)
            print(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()


def main(train_datapath, test_datapath):
    # Create the datasets
    train_dataset = DatasetBeethoven(train_datapath)
    test_dataset = DatasetBeethoven(test_datapath)

    # Create the generators
    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': TRAIN_SHUFFLE,
                    'num_workers': NUM_WORKERS}
    test_params = {'batch_size': BATCH_SIZE,
                   'shuffle': TEST_SHUFFLE,
                   'num_workers': NUM_WORKERS}
    train_generator = data.DataLoader(train_dataset, **train_params)
    test_generator = data.DataLoader(test_dataset, **test_params)

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

    # Start training
    train_autoencoder(model=model.to(device), epochs=1, train_generator=train_generator, test_generator=test_generator,
                      optimizer=optimizer, loss_function=loss_function, device=device)
    # x = train_generator.dataset.__getitem__(0)
    # y = torch.stack(x)
    # print(y.shape)
    # model(y)


if __name__ == '__main__':
    # Define the datapaths
    train_datapath = TRAIN_DATAPATH
    test_datapath = TEST_DATAPATH

    main(train_datapath=train_datapath, test_datapath=test_datapath)