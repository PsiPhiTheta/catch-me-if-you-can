'''
Run this to train and save weights on all signatures.
'''
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

from vanilla_vae import VanillaVae, train_all_sigs

if __name__ == '__main__':
	sig_id_list = train_all_sigs(sig_type='genuine', epochs=250)