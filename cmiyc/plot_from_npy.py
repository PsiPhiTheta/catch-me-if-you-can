import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
			data_list.append(data)
			print("File found for sig_id={}, shape is {}".format(sig_id, data.shape))

		sig_id += 1
	print("All sig_ids processed.")

	all_data = np.array(data_list)
	print(all_data.shape)

def main():

	###################
	# Make combined graph from .npy files spit out by CustomVae.train_all_sigs()
	###################
	plot_from_npy()


if __name__ == "__main__":
	main()