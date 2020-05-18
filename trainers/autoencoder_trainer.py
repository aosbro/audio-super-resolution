from models.autoencoder import AutoEncoder
from trainers.base_trainer import Trainer
from torch.optim import lr_scheduler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os


class AutoEncoderTrainer(Trainer):
    def __init__(self, train_loader, test_loader, valid_loader, general_args, trainer_args):
        super(AutoEncoderTrainer, self).__init__(train_loader, test_loader, valid_loader, general_args)
        # Paths
        self.loadpath = trainer_args.loadpath
        self.savepath = trainer_args.savepath

        # Model
        self.autoencoder = AutoEncoder(general_args=general_args).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.autoencoder.parameters(), lr=trainer_args.lr)
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=trainer_args.scheduler_step,
                                             gamma=trainer_args.scheduler_gamma)

        # Load saved states
        if os.path.exists(trainer_args.loadpath):
            self.load()

        # Loss function
        self.time_criterion = nn.MSELoss()
        self.frequency_criterion = nn.MSELoss()

        # Boolean to differentiate generator from auto-encoder
        self.is_autoencoder = True

    def train(self, epochs):
        for epoch in range(epochs):
            self.autoencoder.train()
            for i in range(self.train_batches_per_epoch):
                data_batch = next(self.train_loader_iter)
                # Transfer to GPU
                input_batch, target_batch = data_batch[0].to(self.device), data_batch[1].to(self.device)

                # Concatenate the input and target signals along first dimension and transfer to GPU
                # local_batch = torch.cat(local_batch).to(self.device)
                self.optimizer.zero_grad()

                # Train with input samples
                generated_batch, _ = self.autoencoder(input_batch)
                specgram_input_batch = self.spectrogram(input_batch)
                specgram_generated_batch = self.spectrogram(generated_batch)

                # Compute the input losses
                input_time_l2_loss = self.time_criterion(generated_batch, input_batch)
                input_freq_l2_loss = self.frequency_criterion(specgram_generated_batch, specgram_input_batch)
                input_loss = input_time_l2_loss + input_freq_l2_loss
                input_loss.backward()

                # Train with target samples
                generated_batch, _ = self.autoencoder(target_batch)
                specgram_target_batch = self.spectrogram(target_batch)
                specgram_generated_batch = self.spectrogram(generated_batch)

                # Compute the input losses
                target_time_l2_loss = self.time_criterion(generated_batch, target_batch)
                target_freq_l2_loss = self.frequency_criterion(specgram_generated_batch, specgram_target_batch)
                target_loss = target_time_l2_loss + target_freq_l2_loss
                target_loss.backward()

                # Update weights
                self.optimizer.step()

                # Store losses
                self.train_losses['time_l2'].append((input_time_l2_loss + target_time_l2_loss).item())
                self.train_losses['freq_l2'].append((input_freq_l2_loss + target_freq_l2_loss).item())

            # Print message
            message = 'Train, epoch {}: \n' \
                      '\t Time: {} \n' \
                      '\t Frequency: {} \n'.format(
                self.epoch, np.mean(self.train_losses['time_l2'][-self.train_batches_per_epoch:]),
                np.mean(self.train_losses['freq_l2'][-self.train_batches_per_epoch:]))
            print(message)

            with torch.no_grad():
                self.eval()

            # Save the trainer state
            if self.need_saving:
                self.save()

            # Increment epoch counter
            self.epoch += 1
            self.scheduler.step()

    def eval(self):
        with torch.no_grad():
            self.autoencoder.eval()
            batch_losses = {'time_l2': [], 'freq_l2': []}
            for i in range(self.valid_batches_per_epoch):
                # Transfer to GPU
                data_batch = next(self.valid_loader_iter)
                data_batch = torch.cat(data_batch).to(self.device)

                # Forward pass
                generated_batch, _ = self.autoencoder.forward(data_batch)

                # Get the spectrogram
                specgram_batch = self.spectrogram(data_batch)
                specgram_generated_batch = self.spectrogram(generated_batch)

                # Compute and store the loss
                time_l2_loss = self.time_criterion(generated_batch, data_batch)
                freq_l2_loss = self.frequency_criterion(specgram_generated_batch, specgram_batch)
                batch_losses['time_l2'].append(time_l2_loss.item())
                batch_losses['freq_l2'].append(freq_l2_loss.item())

            # Store the validation losses
            self.valid_losses['time_l2'].append(np.mean(batch_losses['time_l2']))
            self.valid_losses['freq_l2'].append(np.mean(batch_losses['freq_l2']))

            # Display the validation loss
            message = 'Epoch {}: \n' \
                      '\t Time: {} \n' \
                      '\t Frequency: {} \n'.format(self.epoch,
                                                   np.mean(np.mean(batch_losses['time_l2'])),
                                                   np.mean(np.mean(batch_losses['freq_l2'])))
            print(message)

            # Check if the loss is decreasing
            self.check_improvement()

    def save(self):
        """
        Saves the model(s), optimizer(s), scheduler(s) and losses
        :return: None
        """
        torch.save({
            'epoch': self.epoch,
            'autoencoder_state_dict': self.autoencoder.state_dict(),
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
        checkpoint = torch.load(self.loadpath, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.valid_losses = checkpoint['valid_losses']

    def plot_autoencoder_embedding_space(self, n_batches, fig_savepath=None):
        """
        Plots a 2D representation of the embedding space. Can be useful to determine whether or not the auto-encoder's
        features can be used to improve the generation of realistic samples.
        :param n_batches: number of batches to use for the plot.
        :param fig_savepath: location where to save the figure
        :return: None
        """
        n_pairs = n_batches * self.valid_loader.batch_size
        n_features = 9
        with torch.no_grad():
            autoencoder = self.autoencoder.eval()
            embeddings = []
            for k in range(n_batches):
                # Transfer to GPU
                data_batch = next(self.valid_loader_iter)
                data_batch = torch.cat(data_batch).to(self.device)

                # Forward pass
                _, embedding_batch = autoencoder(data_batch)

                # Store the embeddings
                embeddings.append(embedding_batch)

            # Convert list to tensor
            embeddings = torch.cat(embeddings)

        # Randomly select features from the channel dimension
        random_features = np.random.randint(embeddings.shape[1], size=n_features)
        # Plot embeddings
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, random_feature in enumerate(random_features):
            # Map embedding to a 2D representation
            tsne = TSNE(n_components=2, verbose=0, perplexity=50)
            tsne_results = tsne.fit_transform(embeddings[:, random_feature, :].detach().cpu().numpy())
            for k in range(2):
                label = ('input' if k == 0 else 'target')
                axes[i // 3][i % 3].scatter(tsne_results[k * n_pairs: (k + 1) * n_pairs, 0],
                                            tsne_results[k * n_pairs: (k + 1) * n_pairs:, 1], label=label)
                axes[i // 3][i % 3].set_title('Channel {}'.format(random_feature), fontsize=14)
                axes[i // 3][i % 3].set_xlabel('Learned dimension 1', fontsize=14)
                axes[i // 3][i % 3].set_ylabel('Learned dimension 2', fontsize=14)
                axes[i // 3][i % 3].legend()

        # Save plot if needed
        if fig_savepath:
            plt.savefig(fig_savepath)
        plt.show()
