from torchaudio.transforms import Spectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
from datasets.datasets import DatasetBeethoven
import numpy as np


def main():
    datapath = '/media/thomas/Samsung_T5/VITA/data/music/music_train.npy'
    fs = 16000
    ratio = 4
    overlap = 0.5
    dataset = DatasetBeethoven(datapath, ratio, overlap)
    print(dataset.__len__())
    print(dataset.window_number)

    x_h, x_l = dataset.__getitem__(1)
    x_h_np = x_h.numpy().squeeze()
    x_l_np = x_l.numpy().squeeze()

    # plot_spectrograms(x_h_np, x_l_np, 16000)

    specgram_l = Spectrogram(normalized=True, n_fft=512, hop_length=128)(x_l)
    specgram_h = Spectrogram(normalized=True, n_fft=512, hop_length=128)(x_h)
    print(specgram_h.shape)

    specgram_l_db = AmplitudeToDB(top_db=80)(specgram_l)
    specgram_h_db = AmplitudeToDB(top_db=80)(specgram_h)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(np.flip(specgram_h_db[0].numpy(), axis=0))
    axes[1].imshow(np.flip(specgram_l_db[0].numpy(), axis=0))
    plt.show()

    # full_sample = overlap_and_add_samples(batch_l.view(31, 1, 8192))


if __name__ == '__main__':
    main()