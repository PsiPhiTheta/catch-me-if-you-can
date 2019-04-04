import dataset_utils
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

K.set_image_dim_ordering('th')

PATH_SAVE = 'data/augmented'

PATH_TRAIN_GENUINE = 'data/augmented/train-dutch-offline-genuine.npy'
PATH_TRAIN_FORGERIES = 'data/augmented/train-dutch-offline-forgeries.npy'

GENUINE = True

if (GENUINE):
    x_train, _ = dataset_utils.load_clean_train(sig_type='genuine')
else:
    x_train, _ = dataset_utils.load_clean_train(sig_type='forgery')

x_train = x_train.reshape(x_train.shape[0], 1, 128, 128).astype('float32')

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zca_epsilon=1e-06,
                             rotation_range=0,
                             width_shift_range=0.0,
                             height_shift_range=0.0,
                             brightness_range=None,
                             shear_range=0.0,
                             zoom_range=0.0,
                             channel_shift_range=0.0,
                             fill_mode='nearest',
                             cval=0.0,
                             horizontal_flip=False,
                             vertical_flip=False,
                             rescale=None,
                             preprocessing_function=None,
                             data_format=None,
                             validation_split=0.0,
                             dtype=None)

datagen.fit(x_train)

for x_batch in datagen.flow(x_train, batch_size=9, save_to_dir=PATH_SAVE, save_prefix='aug', save_format='png'):
    # create a grid of 3x3 images
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_batch[i].reshape(128, 128), cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()
    break
