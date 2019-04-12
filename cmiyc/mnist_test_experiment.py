from keras.datasets import mnist

import viz_utils
from vanilla_vae import VanillaVae

if __name__ == '__main__':

    # Params
    image_res = 28
    intermediate_dim = 512
    latent_dim = 2
    val_split = 0.2
    epochs = 15
    batch_size = 128

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, image_res*image_res)
    x_test = x_test.reshape(-1, image_res*image_res)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Instantiate the VAE
    vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)

    # Train the VAE
    vanilla_vae.fit(x_train, val_split, epochs, batch_size)

    # Generate from random sampling
    viz_utils.plot_manifolds_2d(vanilla_vae.decoder, 15, 28, 1)
