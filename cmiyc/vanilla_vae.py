import os
import time

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.losses import binary_crossentropy, mse
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np

import dataset_utils
import viz_utils


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

        self.encoder = Model(inputs, z_mean)

        # Decoder
        decoder_inputs = Input(shape=(latent_dim,))
        decoder_h = Dense(intermediate_dim, activation='relu')(decoder_inputs)
        outputs = Dense(input_dim, activation='sigmoid')(decoder_h)

        self.decoder = Model(decoder_inputs, outputs)

        # end-to-end vae
        vae_outputs = self.decoder(z)
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
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    def get_data(self, sig_id=1, sig_type='genuine'):
        '''
        Load the specified sig id and signature type.
        '''
        x_train, y_train = dataset_utils.load_clean_train(sig_type=sig_type,
                                                      sig_id=sig_id,
                                                      id_as_label=False)

        self.x_train = x_train
        self.y_train = y_train

    def fit(self, val_split, epochs, batch_size, save_dir=None, fn=None):
        """ Train the model and save the weights if a `save_dir` is set.
        """
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # Setup checkpoint to save best model
        callbacks = [
            ModelCheckpoint(save_dir+fn, monitor='val_loss', verbose=1,
                            save_best_only=True)
        ] if save_dir else []

        start = time.time()

        history = self.vae.fit(self.x_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=val_split,
                     shuffle=True,
                     callbacks=callbacks,
                     verbose=1)
        print("Total train time: {0:.2f} sec".format(time.time() - start))

        if save_dir:
            self.vae.save_weights(save_dir+fn)
        return history

    def load_weights(self, weight_path):
        """
        Load weights from previous training.
        """
        self.vae.load_weights(weight_path)

    def predict(self, processed_img):
        """
        Take in an preprocessed image file, run through net, and return output.
        """
        return self.vae.predict(processed_img)


if __name__ == '__main__':

    # Parameters
    sig_id = 1
    sig_type = 'genuine'
    image_res = 128
    intermediate_dim = 512
    latent_dim = 256
    val_frac = 0.1
    epochs = 250
    batch_size = 32
    save_dir = 'saved-models/'
    fn = 'models_{}_sigid{}_res{}_id{}_ld{}_epoch{}.h5'.format(
        sig_type,
        sig_id,
        image_res, 
        intermediate_dim,
        latent_dim,
        epochs )

    # Instantiate network
    vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)

    # Get training data
    vanilla_vae.get_data(sig_id, sig_type)

    # Train
    history = vanilla_vae.fit(val_frac, epochs, batch_size, save_dir, fn)

    # # Plot the losses after training
    # viz_utils.plot_history(history)
    #
    # # Sample the latent space
    # viz_utils.generate_from_random(vanilla_vae.decoder, latent_dim, image_res)
    #
    # # Visualize 2D latent space with colored label (genuine or forgery)
    # x_encoded = vanilla_vae.encoder.predict(x_train)
    # viz_utils.plot_encoded_2d(x_encoded, y_train)
    #
    # # Visualize 2D manifolds
    # viz_utils.plot_manifolds_2d(vanilla_vae.decoder)
    
    # # Plot the original input image
    # viz_utils.plot_original_image(x_train)
