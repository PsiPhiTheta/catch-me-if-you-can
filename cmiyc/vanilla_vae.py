import os
import time
import pickle

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape
from keras.losses import binary_crossentropy, mse
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split

import dataset_utils
import data_augmentation
import viz_utils


class VanillaVae():
	SAVE_DIR = 'saved-models/'

	def __init__(self, input_dim, intermediate_dim, latent_dim):

		self.recon_loss = None
		self.kl_loss = None
		self.total_loss = None
		self.beta = 1  # Beta that controls disentanglement

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

	def set_beta(self, beta):
		self.beta = beta

	def vae_loss(self, inputs, outputs, original_dim, z_mean, z_log_var):
		""" VAE loss = mse_loss (reconstruction) + kl_loss
		"""
		self.recon_loss = mse(inputs, outputs) * original_dim
		kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
		self.kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
		self.total_loss = K.mean(self.recon_loss + self.beta*self.kl_loss)
		return self.total_loss

	def fit(self, x_train, val_split, epochs, batch_size, save_dir=None, fn=''):
		""" Train the model and save the weights if a `save_dir` is set.
		"""
		if save_dir:
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

		temp_fn = "incomplete_" + fn
		
		# Setup checkpoint to save best model
		callbacks = [
			ModelCheckpoint(save_dir+temp_fn, monitor='val_loss', verbose=1,
							save_best_only=True)
		] if save_dir else []

		start = time.time()

		history = self.vae.fit(x_train,
					 epochs=epochs,
					 batch_size=batch_size,
					 validation_split=val_split,
					 shuffle=True,
					 callbacks=callbacks,
					 verbose=1)
		
		print("Total train time: {0:.2f} sec".format(time.time() - start))

		if save_dir:
			# Rename to proper filename after all epochs successfully run
			os.rename(save_dir+temp_fn, save_dir+fn)
			self.vae.save_weights(save_dir+fn)
			print("Saved final weights to {}".format(save_dir+fn))
		return history

	def fit_generator(self, train_generator, steps_per_epoch, epochs,
					  val_generator, val_steps, save_dir=None, fn=''):
		""" Train the model just like fit but with a data augmentation generator
		in place of the x_train
		"""

		if save_dir:
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)

		temp_fn = "incomplete_" + fn

		# Setup checkpoint to save best model
		callbacks = [
			ModelCheckpoint(save_dir + temp_fn, monitor='val_loss', verbose=1,
							save_best_only=True)
		] if save_dir else []

		start = time.time()

		# Internal generator wrapper to reshape the input
		def vae_generator(generator):
			for x_batch, y_batch in generator:
				x_batch = x_batch.reshape(x_batch.shape[0], -1)
				yield x_batch, None

		history = self.vae.fit_generator(
			vae_generator(train_generator),
			steps_per_epoch,
			epochs,
			validation_data=vae_generator(val_generator),
			validation_steps=val_steps,
			callbacks=callbacks,
			verbose=1)

		print("Total train time: {0:.2f} sec".format(time.time() - start))

		if save_dir:
			# Rename to proper filename after all epochs successfully run
			os.rename(save_dir + temp_fn, save_dir + fn)
			self.vae.save_weights(save_dir + fn)
			print("Saved final weights to {}".format(save_dir + fn))
		return history

	def get_gens(self, sig_type, sig_id, val_frac, image_res, batch_size):

		x_train, y_train, _, _ = dataset_utils.load_clean_train_test(
			vae_sig_type=sig_type,
			sig_id=sig_id,
			id_as_label=False)
		x_train, x_val, y_train, y_val = train_test_split(
		x_train, y_train, test_size=val_frac)

		train_gen = data_augmentation.get_datagen().flow(
			x_train.reshape(-1, image_res, image_res, 1),
			y_train,
			batch_size=batch_size)

		val_gen = data_augmentation.get_datagen().flow(
			x_val.reshape(-1, image_res, image_res, 1),
			y_val,
			batch_size=batch_size)

		return train_gen, val_gen

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

def train_all_sigs(sig_type='genuine', epochs=250, frac=0.5, seed=4):
	'''
	Helper function to train and save VAE weights 
	for a set of genuine training signatures.

	Skips a signature if the associated weight file has already been created
	in the anticipated directory.

	Default seed=4 is set for reproducibility.
	'''

	if not os.path.exists(VanillaVae.SAVE_DIR + 'logs/'):
		os.makedirs(VanillaVae.SAVE_DIR + 'logs/')

	if not os.path.exists(VanillaVae.SAVE_DIR + 'history/'):
		os.makedirs(VanillaVae.SAVE_DIR + 'history/')

	sig_id_list = dataset_utils.get_sig_ids(sig_type='genuine')

	# Save this list for reference later, in case we need it
	# ts = time.strftime("%Y%m%d-%H%M%S")
	# logfile = VanillaVae.SAVE_DIR + 'logs/' + 'train_sig_list_{}.txt'.format(ts)
	# print("Made logfile at {}".format(logfile))

	start = time.time()

	for sig_id in sig_id_list:

		# Parameters

		image_res = 128
		intermediate_dim = 512
		latent_dim = 256
		val_frac = 0.1
		batch_size = 32
		steps_per_epoch = 64
		val_steps=8
		save_dir = VanillaVae.SAVE_DIR
		fn = 'models_{}_sigid{}_res{}_id{}_ld{}_epoch{}.h5'.format(
			sig_type,
			sig_id,
			image_res, 
			intermediate_dim,
			latent_dim,
			epochs)

		# Skip this sig_id if the weight file has already been created:
		# Anticipating that this will take a long time to run,
		# So make it easy to restart where we left off
		weight_exists = os.path.isfile(save_dir+fn)
		if weight_exists:
			print("{} already exists, skipping".format(fn))
			continue

		vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)

		train_gen, val_gen = vanilla_vae.get_gens(sig_type, sig_id, val_frac, image_res, batch_size)

		# Train
		history = vanilla_vae.fit_generator(
		train_gen, steps_per_epoch, epochs, val_gen, val_steps, save_dir, fn)

		# Write history to pickle in case we want it later
		hist_pickle_filename = 'history_{}_sigid{}_res{}_id{}_ld{}_epoch{}.pkl'.format(
			sig_type,
			sig_id,
			image_res, 
			intermediate_dim,
			latent_dim,
			epochs )
		
		with open(VanillaVae.SAVE_DIR + 'history/' + hist_pickle_filename, 'wb') as fp:
			pickle.dump(history.history, fp)
		print("History saved to {}".format(VanillaVae.SAVE_DIR + 'history/' + hist_pickle_filename))

	print("train_all_sigs completed in {} sec".format(time.time()- start))

	return sig_id_list


if __name__ == '__main__':

	# Parameters
	sig_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	sig_type = 'genuine'
	image_res = 128
	intermediate_dim = 512
	latent_dim = 16
	val_frac = 0.1
	epochs = 50
	batch_size = 32
	steps_per_epoch = 64
	val_steps=8
	save_dir = VanillaVae.SAVE_DIR
	fn = 'models_{}_sigid{}_res{}_id{}_ld{}_epoch{}.h5'.format(
			sig_type,
			sig_id,
			image_res, 
			intermediate_dim,
			latent_dim,
			epochs 
		)

	# Instantiate network
	vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)

	# Get training data

	train_gen, val_gen = vanilla_vae.get_gens(sig_type, sig_id, val_frac, image_res, batch_size)

	# Train
	history = vanilla_vae.fit_generator(
		train_gen, steps_per_epoch, epochs, val_gen, val_steps, save_dir, fn)

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
	viz_utils.plot_manifolds_2d(vanilla_vae.decoder, n=5)

	# # Plot the original input image
	# viz_utils.plot_original_image(x_train)
