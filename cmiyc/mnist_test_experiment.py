from keras.datasets import mnist

from vanilla_vae import VanillaVae

if __name__ == '__main__':

    # Params
    image_res = 28
    intermediate_dim = 128
    latent_dim = 16

    # Load the MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Instantiate the VAE
    vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)

    # Train the VAE
    vanilla_vae.train