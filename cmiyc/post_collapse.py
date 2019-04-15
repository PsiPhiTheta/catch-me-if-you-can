'''
A custom class that overrides VanillaVae to test for posterior collapse.
Tracks the MSE and KL portions of the loss separately
And plots over the course of training.
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
import seaborn as sns

import pickle
import time

import dataset_utils
import viz_utils
from vanilla_vae import VanillaVae

class SplitReconKL(Callback):
	'''
	Custom callback to keep track of the Recon/KL split of the
	total loss during training.

	Generates an image at the end.

	Saves info to a file for re-use too.
	'''

	SAVE_DIR = 'saved-models/loss_splits/'

	def __init__(self, fn):
		self.fn = fn
		self.parse_fn()
		super(SplitReconKL, self).__init__()

	def parse_fn(self):
		'''
		Parse filenames - one to save to .npy, one to save an image
		'''
		self.npyfn = os.path.splitext(self.fn)[0]+'.npy'
		self.imgfn = os.path.splitext(self.fn)[0]+'.png'

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

	def on_train_end(self, logs={}):
		self.plot_and_save()

	def plot_and_save(self, save_img=True):
		'''
		Make a plot of the breakdown of loss over time and save,
		write data to a file so we can further process it if we wish
		'''
		sns.set() # pretty seaborn styles

		if not os.path.exists(SplitReconKL.SAVE_DIR):
			os.makedirs(SplitReconKL.SAVE_DIR)
		
		# Save the three series to a .npy in case we want it later
		data = np.array((
			np.asarray(self.total_losses), 
			np.asarray(self.recon_losses), 
			np.asarray(self.kl_losses)))
		print(data)
		with open(SplitReconKL.SAVE_DIR + self.npyfn, 'wb') as npfile:
			np.save(npfile, data)

		# Save a stacked chart of the loss breakdown over time

		# Generate a x spanning the number of epochs
		epoch_n = data.shape[1]
		x = np.linspace(1, epoch_n, epoch_n)
		y1 = data[0,:] # total
		y2 = data[1,:] # recon
		y3 = data[2,:] # KL
		
		plt.figure(figsize=(10,8))
		plt.stackplot(x, y2, y3, labels=['Reconstruction term','KL term'])
		
		plt.title("Training log-loss: Reconstruction vs KL term breakdown")
		plt.xlabel("Epoch")
		plt.xticks(np.arange(min(x), max(x)+1, 1))

		plt.ylabel("Log-Loss")
		plt.yscale("log")

		plt.legend(loc='upper left')
		if save_img:
			plt.savefig(SplitReconKL.SAVE_DIR+self.imgfn)
		else:
			plt.show()

class CustomVae(VanillaVae):
	'''
	An altered version of VanillaVae that allows us to investigate:

	- whether we have posterior collapse
	- try using a different reconstruction loss metric
	- try using different values to weight the recon loss vs KL div, if time	
	'''

	def __init__(self, input_dim, intermediate_dim, latent_dim, fn, recon_type='mse', beta=1.0):

		self.recon_loss = None
		self.kl_loss = None
		self.total_loss = None

		self.recon_type = recon_type # nothing is done with this yet
		self.beta = beta

		self.z_mean = None
		self.z_log_var = None

		self.fn = fn

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

	def fit(self, val_split, epochs, batch_size, save_dir=None, fn=None):
		""" Train the model and save the weights if a `save_dir` is set.
		"""
		if save_dir:
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

		temp_fn = "incomplete_" + fn
		
		# Custom callback to keep track of KL and Recon loss split
		# during training
		split_recon_kl = SplitReconKL(fn=self.fn)
		
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
			str(args['beta']).replace('.', '_') 
		)
	

	# Instantiate network
	custom_vae = CustomVae(
		args['image_res']*args['image_res'], 
		args['intermediate_dim'], 
		args['latent_dim'],
		args['fn'],
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


if __name__ == "__main__":
	main()