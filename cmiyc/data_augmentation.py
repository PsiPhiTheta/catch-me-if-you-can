import matplotlib.pyplot as plt
import dataset_utils
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

K.set_image_dim_ordering('th')

PATH_SAVE = 'data/augmented'

PATH_TRAIN_GENUINE = 'data/augmented/train-dutch-offline-genuine.npy'
PATH_TRAIN_FORGERIES = 'data/augmented/train-dutch-offline-forgeries.npy'

LEN_SIZE = 9

GENUINE = True

if (GENUINE):
    x_train, _ = dataset_utils.load_clean_train(sig_type='genuine', sig_id=4)
else:
    x_train, _ = dataset_utils.load_clean_train(sig_type='forgery', sig_id=4)

x_train = x_train.reshape(x_train.shape[0], 1, 128, 128).astype('float32')

x_train = x_train[0]

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zca_epsilon=1e-06,
                             rotation_range=45, # modify this
                             width_shift_range=10.0, # modify this
                             height_shift_range=10.0, # modify this
                             brightness_range=None,
                             shear_range=0.0,
                             zoom_range=0.1, # modify this
                             channel_shift_range=0.5, # modify this
                             fill_mode='constant', # specify this depending on the usecase
                             cval=1.0,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=1, # modify this
                             preprocessing_function=None,
                             data_format=None,
                             validation_split=0.0,
                             dtype=None)

datagen.fit(x_train.reshape(1,1,128,128))

for i in range(LEN_SIZE):
    for x_batch in datagen.flow(x_train.reshape(1,1,128,128), batch_size=1, save_to_dir=PATH_SAVE, save_prefix='aug', save_format='png'):
        print('Generated image:', i)
        break