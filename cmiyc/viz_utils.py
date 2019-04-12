from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import math

import dataset_utils

from vanilla_vae import VanillaVae


def plot_history(history):
    """
    Plots the training and validation losses
    """
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    # plt.yscale(value='log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_encoded_2d(x, label):
    """
    Plot the encoded x given the label (sig_id or type).
    The encoded dimensionality (x) should be 2.
    """
    plt.scatter(x[:, 0], x[:, 1],
                c=label,
                cmap='jet',
                marker='o',
                s=100,
                alpha=0.5)
    plt.colorbar()
    plt.show()


def plot_encoded_3d(x, label):
    """
    Plot the encoded x given the label (sig_id or type).
    The encoded dimensionality (x) should be 3.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2],
               c=label,
               cmap='jet',
               marker='o',
               s=100,
               alpha=0.5)
    plt.show()


def plot_manifolds_2d(decoder, n=15, size=128, std_dev=1):
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)
    figure = np.zeros((size * n, size * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * std_dev
            x_decoded = decoder.predict(z_sample)
            im = x_decoded.reshape(size, size)
            figure[i * size: (i + 1) * size, j * size: (j + 1) * size] = im

    plt.figure()
    plt.imshow(figure, cmap='gray')
    plt.show()


def plot_dataset(dataset, size=128):
    """Plot the entire given dataset (arranged in a square figure)
    """
    n = math.ceil(math.sqrt(len(dataset)))
    figure = np.zeros((size*n, size*n))
    for i, image in enumerate(dataset):
        if i >= len(dataset):
            break
        x = i // n
        y = i % n
        figure[x*size: (x+1)*size, y*size: (y+1)*size] = image.reshape(size,
                                                                       size)
    plt.figure()
    plt.imshow(figure, cmap='gray')
    plt.show()


def plot_dataset_random(dataset, n=100, size=128):
    """Plot a random subset of the given dataset
    """
    sample = np.random.randint(0, len(dataset), n)
    plot_dataset(dataset[sample, ])


def generate_from_random(decoder, latent_dim, image_res=128):
    """
    Samples form the latent space and return a generated output.
    """
    sample = np.random.normal(size=(1, latent_dim))
    pred = decoder.predict(sample).reshape((image_res, image_res))
    plt.imshow(pred, cmap='gray')
    plt.show()


def encode_plot_tsne(x, y, encoder):
    """
    Plot t-SNE of the encoded signatures color-coded with the labels
    """
    x = encoder.predict(x)
    tsne = TSNE()
    results = tsne.fit_transform(x)
    plt.scatter(results[:, 0], results[:, 1],
                c=y,
                cmap='jet',
                marker='o',
                s=100,
                alpha=0.5)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':

    image_res = 128
    intermediate_dim = 512
    latent_dim = 256
    sig_id = 1
    save_dir = 'saved-models/models.h5'

    # Load model
    vanilla_vae = VanillaVae(image_res * image_res, intermediate_dim, latent_dim)
    vanilla_vae.load_weights(save_dir)

    # Load data
    x, y = dataset_utils.load_clean_train(sig_type='genuine',
                                          sig_id=[1, 2, 3, 4],
                                          id_as_label=True)

    # Viz t-SNE
    encode_plot_tsne(x, y, vanilla_vae.encoder)

