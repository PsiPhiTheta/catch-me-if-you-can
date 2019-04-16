import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(data, save_dir, save_fn, title=None):
	# Generate a x spanning the number of epochs
		epoch_n = data.shape[1]
		x = np.linspace(1, epoch_n, epoch_n)
		y1 = data[0,:] # total
		y2 = data[1,:] # recon
		y3 = data[2,:] # KL
		
		plt.figure(figsize=(10,8))
		plt.stackplot(x, y2, y3, labels=['Reconstruction term','KL term'])
		
		if title:
			plt.title(title)
		else:
			plt.title("Training log-loss: Reconstruction vs KL term breakdown")
		
		plt.xlabel("Epoch")
		if epoch_n < 20:
			x_tick_interval = 1
		elif epoch_n <= 100:
			x_tick_interval = 5
		else:
			x_tick_interval = 10

		plt.xticks(np.arange(min(x), max(x)+1, x_tick_interval))

		plt.ylabel("Log-Loss")
		plt.yscale("log")

		plt.legend(loc='upper left')
		if save_img:
			plt.savefig(saved_dir+save_fn)
			print("Saved {}".format(saved_dir+save_fn))
		else:
			plt.show()

def plot_from_npy():
	'''
	Using all .npy files spit out by train-all-sigs, generate a stacked area
	chart for % split of KL vs recon loss over all images, for a given beta
	'''

	# Load all .npy to a single array
	# Assumes this format for filename: alt_models_genuine_sigid1_res128_id512_ld256_epoch5_mse_b1_0.npy
	last_id = 69

	sig_id = 1
	data_list = []
	while sig_id < last_id:
		# cycle thru all sigs
		potential_fn = "alt_models_genuine_sigid{}_res128_id512_ld256_epoch100_mse_b1_0.npy".format(sig_id)
		fp = './saved-models/loss_splits/' + potential_fn
		if os.path.isfile(fp):
			data = np.load(fp)

			# Convert values of col 1 and 2 to percentage of col 0,
			# to normalize for different loss scales on different sigs
			data[0] /= data[0]
			data[1] /= data[0]
			data[2] /= data[0] 

			data_list.append(data)
			print("File found for sig_id={}, shape is {}".format(sig_id, data.shape))

		sig_id += 1
	print("All sig_ids processed.")

	all_data = np.array(data_list)
	
	# Take mean down the first axis
	mean_data = np.mean(all_data, axis=0)
	print(mean_data.shape)

def main():

	###################
	# Make combined graph from .npy files spit out by CustomVae.train_all_sigs()
	###################
	plot_from_npy()


if __name__ == "__main__":
	main()