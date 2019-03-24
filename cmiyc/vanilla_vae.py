import os
import time

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.losses import mse

import keras.backend as K
import numpy as np

import dataset_utils


class VanillaVae():
    def __init__(self, input_dim, intermediate_dim, latent_dim):

        # Encoder
        inputs = Input(shape=(input_dim, ))
        h = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        # Latent space
        args = [z_mean, z_log_var]
        z = Lambda(self.sampling, output_shape=(latent_dim,))(args)

        # Instantiate encoder
        self.encoder = Model(inputs, [z_mean, z_log_var, z])

        # Decoder
        decoder_inputs = Input(shape=(latent_dim,))
        decoder_h = Dense(intermediate_dim, activation='relu')(decoder_inputs)
        outputs = Dense(input_dim, activation='sigmoid')(decoder_h)

        # Instantiate decoder
        self.decoder = Model(decoder_inputs, outputs)

        # Instantiate VAE
        vae_outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, vae_outputs)

        # Setup and compile
        self.vae.add_loss(self.vae_loss(
            inputs, vae_outputs, input_dim, z_mean, z_log_var))
        self.vae.compile(optimizer='adam')

    @staticmethod
    def sampling(args):
        """ Function used for the reparameterizaton trick
        """
        z_mean, z_log_sigma =  args
        batch_size = K.shape(z_mean)[0]
        latent_dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

    @staticmethod
    def vae_loss(inputs, outputs, original_dim, z_mean, z_log_var):
        """ VAE loss = mse_loss (reconstruction) + kl_loss
        """
        reconstruction_loss = mse(inputs, outputs) * original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * K.sum(kl_loss, axis=1)
        return K.mean(reconstruction_loss + kl_loss)

    def fit(self, x_train, x_val, epochs, batch_size, save_path=None):
        """ Train the model and save the weights is a `save_path` is set.
        """
        self.vae.fit(x_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(x_val, None))
        if save_path:
            self.vae.save_weights(save_path)

    def load_weights(self, weight_path):
        """
        Load weights from previous training.
        """
        self.vae.load_weights(weight_path)

    def predict(self, processed_img):
        """
        Take in an preprocessed image file, run through net, and return output.
        """
        self.vae.predict(processed_img)


if __name__ == '__main__':

    # Parameters
    image_res = 256
    intermediate_dim = 256
    latent_dim = 64
    val_frac = 0.1
    epochs = 1
    batch_size = 16

    # Load data
    x_train, _ = dataset_utils.load_clean_train()

    # Shuffle and split for validation
    idx_shuffle = np.array(range(len(x_train)))
    np.random.shuffle(idx_shuffle)
    val_split = int(val_frac * len(x_train))
    x_val = x_train[:val_split]
    x_train = x_train[val_split:]

    # Instantiate network
    vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)

    save_dir = 'saved-networks/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Train network and save weights
    start = time.time()
    vanilla_vae.fit(x_train,
                    x_val,
                    epochs,
                    batch_size,
                    save_dir + 'vanilla-vae.h5')
    print("Total train time: {0:.2f} sec".format(time.time() - start))

