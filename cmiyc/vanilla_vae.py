from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

import dataset_utils

# Parameters
image_res = 128
input_dim = image_res * image_res
intermediate_dim = 1024
latent_dim = 64

validation_split = 0.1
epochs = 100
batch_size = 16

save_path = 'saved-models/vanilla-vae.h5'


# Reparameterization trick
def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(z_log_sigma) * epsilon


# Define loss function
def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = binary_crossentropy(inputs, outputs)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=1)
    return K.mean(reconstruction_loss + kl_loss)


def define_model():

    # Encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # Latent space
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Instantiate encoder
    encoder = Model(inputs, [z_mean, z_log_var, z])

    # Decoder
    decoder_inputs = Input(shape=(latent_dim,))
    decoder_h = Dense(intermediate_dim, activation='relu')(decoder_inputs)
    outputs = Dense(input_dim, activation='sigmoid')(decoder_h)

    # Instantiate decoder
    decoder = Model(decoder_inputs, outputs)

    # Instantiate VAE
    vae_outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, vae_outputs)

    # Setup and compile
    vae.add_loss(vae_loss(inputs, vae_outputs, z_mean, z_log_var))
    vae.compile(optimizer='adam')

    return encoder, decoder, vae


def generate_from_random(vae, decoder, latent_dim, image_res):
    vae = vae.load_weights(save_path)
    sample = np.random.normal(size=(1, latent_dim))
    output = decoder.predict(sample).reshape((image_res, image_res))
    plt.imshow(output, cmap='gray')

# Load data
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# original_dim = 784
# x_train = x_train.reshape((-1, original_dim))
# x_test = x_test.reshape((-1, original_dim))
# x_train = x_train / 255
# x_val = x_test / 255

x_train, _ = dataset_utils.load_clean_train()
# Setup checkpoint to save best model

checkpoint = ModelCheckpoint(save_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
# Train the model
_, _, vae = define_model()


vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=[checkpoint],
        verbose=1)
