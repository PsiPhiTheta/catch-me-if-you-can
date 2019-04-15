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

class SplitReconKL(Callback):
	def on_train_begin(self, logs={}):
		self.total_losses = []
		self.recon_losses = []
		self.kl_losses = []

	def on_epoch_end(self, batch, logs={}):

		total = logs.get('loss')
		mse = logs.get('mse')
		kl = logs.get('kl')

		# Assert that kl + mse = total loss within some error as % of total,
		# or else we have a problem
		thresh = 0.0000001
		assert (mse + kl) - total < thresh * total, "Problem: recorded mse + kl != total loss, check SplitReconKL. ({} + {} != {})".format(mse, kl, total)

		self.total_losses.append(total)
		self.recon_losses.append(mse)
		self.kl_losses.append(kl)

	def on_train_end(self):
		'''
		Make a plot of the breakdown of loss over time and save,
		write data to a file so we can process it
		'''

		
class CustomVae(VanillaVae):
	'''
	An altered version of VanillaVae that allows us to investigate:

	- whether we have posterior collapse
	- try using a different reconstruction loss metric
	- try using different values to weight the recon loss vs KL div, if time	
	'''

	def __init__(self, input_dim, intermediate_dim, latent_dim, recon_type='mse', beta=1.0):

		self.recon_loss = None
		self.kl_loss = None
		self.total_loss = None

		self.recon_type = recon_type # nothing is done with this yet
		self.beta = beta

		self.z_mean = None
		self.z_log_var = None

		# Encoder
		inputs = Input(shape=(input_dim, ))
		h = Dense(intermediate_dim, activation='relu')(inputs)
		z_mean = Dense(latent_dim)(h)
		z_log_var = Dense(latent_dim)(h)

		# Not sure if this works
		self.z_mean = z_mean
		self.z_log_var = z_log_var

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
			metrics=[] # this doesn't actually work; metrics get ignored when using add_loss, see keras issue 9459
				)

		self.vae.metrics_tensors.append(CustomVae.calc_mse_alone(inputs, vae_outputs, input_dim))
		self.vae.metrics_names.append("mse")

		self.vae.metrics_tensors.append(self.calc_kl_alone(beta=self.beta))
		self.vae.metrics_names.append("kl")

	def calc_kl_alone(self, beta=1.0):
		kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
		kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
		kl_loss = K.mean(beta * kl_loss)

		return kl_loss

	@staticmethod
	def calc_mse_alone(input_x, output_x, original_dim):
		# original_dim = 128 * 128
		return K.mean(mse(input_x, output_x) * original_dim)

	def vae_loss(self, inputs, outputs, original_dim, z_mean, z_log_var):
		""" VAE loss = mse_loss (reconstruction) + kl_loss

		Note - it may not make sense to use cross-ent loss here if the input
		images are not binarized!!

		beta is a weight that we put on the kl_loss component. Defaults to 1.

		TODO: add xent to this
		"""
		self.recon_loss = mse(inputs, outputs) * original_dim

		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		self.kl_loss = -0.5 * K.sum(kl_loss, axis=-1)

		self.total_loss = K.mean(self.recon_loss + self.beta * self.kl_loss)
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
		
		# Custom callback to keep track of KL and Recon loss split
		# during training
		split_recon_kl = SplitReconKL()
		
		# Setup checkpoint to save best model
		callbacks = [
			ModelCheckpoint(save_dir+temp_fn, monitor='val_loss', verbose=1,
							save_best_only=True),
			split_recon_kl
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

		print('Recon losses saved by callback:')
		print(split_recon_kl.recon_losses)

		print('Total losses saved by callback:')
		print(split_recon_kl.total_losses)
		
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
		'epochs': 100,
		'batch_size': 32,
		'save_dir': CustomVae.SAVE_DIR,
		'recon_type': 'mse', # mse or xent
		'beta': 1.0
	}

	args['fn'] = 'alt_models_{}_sigid{}_res{}_id{}_ld{}_epoch{}_{}_b{}.h5'.format(
			args['sig_type'],
			args['sig_id'],
			args['image_res'], 
			args['intermediate_dim'],
			args['latent_dim'],
			args['epochs'],
			args['recon_type'],
			args['beta'] 
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


if __name__ == "__main__":
	main()