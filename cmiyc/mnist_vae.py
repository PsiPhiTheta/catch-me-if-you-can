from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import mse

import keras.backend as K


# Sampling (reparameterization trick)
def sampling(args):
    z_mean, z_log_sigma =  args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


# VAE loss = mse_loss (reconstruction) + kl_loss
def vae_loss(inputs, outputs, original_dim, z_mean, z_log_var):
    reconstruction_loss = mse(inputs, outputs) * original_dim
    kl_loss =  1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=1)
    kl_loss = -0.5 * kl_loss
    return K.mean(reconstruction_loss + kl_loss)


# Network parameters
original_dim = 784
intermediate_dim = 255
latent_dim = 3

batch_size = 128
epochs = 50


# Encoder
inputs = Input(shape=(original_dim,))
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
outputs = Dense(original_dim, activation='sigmoid')(decoder_h)

# Instantiate decoder
decoder = Model(decoder_inputs, outputs)

# Instantiate VAE
vae_outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, vae_outputs)

# Setup and compile
vae.add_loss(vae_loss(inputs, vae_outputs, original_dim, z_mean, z_log_var))
vae.compile(optimizer='adam')

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, original_dim))
x_test = x_test.reshape((-1, original_dim))
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Train and save weights
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
vae.save_weights('saved-networks/mnist-vae.h5')
