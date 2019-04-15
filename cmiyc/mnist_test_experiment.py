import numpy as np
from keras.datasets import mnist

import viz_utils
from vanilla_vae import VanillaVae


def mnist_test_experiment():

    # Params
    image_res = 28
    intermediate_dim = 512
    latent_dim = 2
    val_split = 0.2
    epochs = 15
    batch_size = 128

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, image_res * image_res)
    x_test = x_test.reshape(-1, image_res * image_res)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Instantiate the VAE
    vanilla_vae = VanillaVae(image_res * image_res, intermediate_dim, latent_dim)

    # Train the VAE
    vanilla_vae.fit(x_train, val_split, epochs, batch_size)

    # Generate from random sampling
    viz_utils.plot_manifolds_2d(vanilla_vae.decoder, 15, 28, 1)


def small_dataset_mnist_test_experiment():

    # Params
    image_res = 28
    intermediate_dim = 512
    latent_dim = 2
    val_split = 0.2
    epochs = 15
    batch_size = 16
    # sample_size = 2500  # about the same as for signatures
    sample_size = 200  # about 12% of the input dimension

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, image_res * image_res)
    x_test = x_test.reshape(-1, image_res * image_res)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Shuffle and sample dataset
    x_train = x_train[np.random.choice(x_train.shape[0],
                                       sample_size, replace=False), :]

    # Instantiate the VAE
    vanilla_vae = VanillaVae(image_res * image_res, intermediate_dim, latent_dim)

    # Train the VAE
    vanilla_vae.fit(x_train, val_split, epochs, batch_size)

    # Generate from random sampling
    viz_utils.plot_manifolds_2d(vanilla_vae.decoder, 15, 28, 1)


def simple_net_mnist_test_experiment():

    # Params
    image_res = 28
    intermediate_dim = 25  # With sig this is about 3% of the input dim
    latent_dim = 2
    val_split = 0.2
    epochs = 15
    batch_size = 128

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, image_res * image_res)
    x_test = x_test.reshape(-1, image_res * image_res)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Instantiate the VAE
    vanilla_vae = VanillaVae(image_res * image_res, intermediate_dim, latent_dim)

    # Train the VAE
    vanilla_vae.fit(x_train, val_split, epochs, batch_size)

    # Generate from random sampling
    viz_utils.plot_manifolds_2d(vanilla_vae.decoder, 15, 28, 1)


if __name__ == '__main__':
    small_dataset_mnist_test_experiment()
    # simple_net_mnist_test_experiment()
