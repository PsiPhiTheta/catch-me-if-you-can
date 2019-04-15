'''
Some tests to see if we have posterior collapse.

Idea: plot means of q(z|x) and see if it converges to p(z) over
the course of training.

Another idea: Set KL loss incredibly high 
and see if classification outcomes are similar
'''

import os

from keras.losses import mse, binary_crossentropy
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.callbacks import ModelCheckpoint, Callback

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

import dataset_utils
import viz_utils
from vanilla_vae import VanillaVae

class AccuracyHistory(Callback):
	def on_train_begin(self, logs={}):
		self.acc = []

	def on_epoch_end(self, batch, logs={}):
		self.acc.append("boop")

class CustomVae(VanillaVae):
	'''
	An altered version of VanillaVae that allows us to investigate:

	- whether we have posterior collapse
	- try using a different reconstruction loss metric
	- try using different values to weight the recon loss vs KL div	
	'''

	def __init__(self, input_dim, intermediate_dim, latent_dim, recon_type='mse'):

		self.recon_loss = None
		self.kl_loss = None
		self.total_loss = None

		self.recon_type = recon_type

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
		self.vae.add_loss(self.vae_loss(inputs, vae_outputs, input_dim, z_mean, z_log_var))
		self.vae.compile(
			optimizer='adam',
			metrics=['']
				)

	def vae_loss(self, inputs, outputs, original_dim, z_mean, z_log_var):
		""" VAE loss = mse_loss (reconstruction) + kl_loss
		"""
		self.recon_loss = mse(inputs, outputs) * original_dim

		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		self.kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

		self.total_loss = K.mean(self.recon_loss + self.kl_loss)
		return self.total_loss

	def loss_wrapper(self, inputs, outputs, original_dim, z_mean, z_log_var):
		'''
		A wrapper for the VAE loss
		So we can save the recon portion and the KL portion
		Separately to history.
		'''

		""" VAE loss = mse_loss (reconstruction) + kl_loss
		"""
		
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

		def seq2seq_loss(input, output):
			""" Final loss calculation function to be passed to optimizer"""
			# Reconstruction loss
			recon_loss = mse(inputs, outputs) * original_dim
			total_loss = K.mean(recon_loss + kl_loss)
			
		return seq2seq_loss

	def fit(self, val_split, epochs, batch_size, save_dir=None, fn=None):
		""" Train the model and save the weights if a `save_dir` is set.
		"""
		if save_dir:
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

		temp_fn = "incomplete_" + fn
		
		# Setup checkpoint to save best model
		test_callback = AccuracyHistory()
		
		callbacks = [
			ModelCheckpoint(save_dir+temp_fn, monitor='val_loss', verbose=1,
							save_best_only=True),
			test_callback
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

		print(test_callback.acc)
		
		if save_dir:
			# Rename to proper filename after all epochs successfully run
			os.rename(save_dir+temp_fn, save_dir+fn)
			self.vae.save_weights(save_dir+fn)
			print("Saved final weights to {}".format(save_dir+fn))
		return history

def main():
	# Parameters
	args = {
		'sig_id': 1,
		'sig_type': 'genuine',
		'image_res': 128,
		'intermediate_dim': 512,
		'latent_dim': 256,
		'val_frac': 0.1,
		'epochs': 5,
		'batch_size': 32,
		'save_dir': CustomVae.SAVE_DIR,
		'recon_type': 'xent'
	}

	args['fn'] = 'alt_models_{}_sigid{}_res{}_id{}_ld{}_epoch{}.h5'.format(
			args['sig_type'],
			args['sig_id'],
			args['image_res'], 
			args['intermediate_dim'],
			args['latent_dim'],
			args['epochs'] 
		)
	

	# Instantiate network
	custom_vae = CustomVae(
		args['image_res']*args['image_res'], 
		args['intermediate_dim'], 
		args['latent_dim'],
		args['recon_type'])

	# Get training data
	custom_vae.load_data(args['sig_id'], args['sig_type'])

	# Train
	history = custom_vae.fit(
		args['val_frac'], 
		args['epochs'], 
		args['batch_size'], 
		args['save_dir'], 
		args['fn']
		)

	print(history.history)
	print(custom_vae.recon_list)
	print(custom_vae.kl_list)


if __name__ == "__main__":
	main()