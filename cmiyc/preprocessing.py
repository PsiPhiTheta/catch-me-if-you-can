import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from keras.preprocessing.image import ImageDataGenerator

from scipy.stats import mode
from PIL import Image, ImageFilter


PATH_SAVE = 'data/clean/'
PATH_TRAIN = 'data/clean/train-dutch-offline.pkl'
PATH_TRAIN_GENUINE = 'data/clean/train-dutch-offline-genuine.npy'
PATH_TRAIN_FORGERIES = 'data/clean/train-dutch-offline-forgeries.npy'


def preprocess_image(image, final_res=256, padding=False, plot=False):
    """ Pre-process a single image.

    The input should be an Image object and the output is a numpy array.

    If `padding` is set to true, the image is padded to be a square before being
    resized to a square of lower resolution (default 256x256 px)

    If `plot` is set, a plot of the images after each step of the pre-processing
    pipeline will be generated
    """

    # Keep track of changes
    images = [('original', image, None)]

    # Convert to gray scale
    image = image.convert('L')
    images.append(('gray scale', image, 'gray'))

    # Find the mode (background color) and convert to black and white with a
    # threshold based on it.
    threshold = int(0.90 * mode(image.getdata())[0][0])
    lut = [0] * threshold + [255] * (256 - threshold)
    image = image.point(lut)
    images.append(('black & white', image, 'gray'))

    # Add padding to form a square if option is set
    if padding:
        image = pad_image_square_center(image)
        images.append(('padding', image, 'gray'))

    # Resize with bilinear interpolation (best results)
    image = image.resize((final_res, final_res), Image.BILINEAR)
    images.append(('resize', image, 'gray'))

    # Plot images if option is set
    if plot:
        fig = plt.figure()
        fig.tight_layout()
        n = len(images)
        for i, (title, im, cmap) in enumerate(images):
            ax = plt.subplot(1, n, i+1)
            ax.set_title(title)
            ax.imshow(im, cmap)
        plt.show()

    # Convert to numpy array
    return np.array(image.getdata()).reshape((final_res, final_res)) / 255


def pad_image_square_center(image):
    """ Pads the given image so that the original is centered on a square.
    """
    new_size = max(image.size)
    new_image = Image.new(image.mode, (new_size, new_size), 'white')
    position = ((new_size - image.size[0]) // 2,
                (new_size - image.size[1]) // 2)
    new_image.paste(image, position)
    return new_image


def fetch_all_raw():
    """ Returns a list with all the signature files
    """
    paths = [
        'data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine/*.*',
        'data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries/*.*',
        'data/raw/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Reference(646)/**/*.*',
        'data/raw/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Questioned(1287)/**/*.*'
    ]
    files = []
    for path in paths:
        files += glob.glob(path, recursive=True)
    assert len(files) == 2295, 'was expecting 2295 files but got {}'.format(len(files))
    return files


def get_type_and_id_from_file(file_path):
    """ Given the full file path, return the label and the id,
    """
    label, sig_id = -1, -1
    if 'Genuine' in file_path:
        label = 1
        sig_id = int(file_path[-10:-7])
    elif 'Forgeries' in file_path:
        label = 0
        sig_id = int(file_path[-10:-7])
    else:
        label = 1 if file_path[-8] == '_' else 0
        sig_id = int(file_path[-7:-4])
    assert label != -1 and sig_id != -1
    return label, sig_id


def fetch_all_raw_genuine():
    """ Returns a list of all the genuine signature files from the raw data.
    That includes the train and test files.
    """

    # Create list for training set
    path = 'data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/' \
           'Offline Genuine/'
    files = glob.glob(path + '*.PNG')

    # Create list for test set
    path = 'data/raw/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/' \
           'Reference(646)/'
    files += glob.glob(path + '**/*.*', recursive=True)
    path = 'data/raw/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/' \
           'Questioned(1287)/'
    files += glob.glob(path + '**/*_' + '[0-9]' * 3 + '.*', recursive=True)
    return files


def fetch_all_raw_forgeries():
    """ Returns a list of all the forged signature files from the raw data.
    That includes the train and test files.
    """

    # Create list for training set
    path = 'data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/' \
           'Offline Forgeries/'
    files = glob.glob(path + '*.png')

    # Create list for test set
    path = 'data/raw/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/' \
           'Questioned(1287)/'
    files += glob.glob(path + '**/*_' + '[0-9]'*7 + '.*', recursive=True)
    return files


def batch_preprocess(files_list, dest_file, final_res, padding):
    """ Executes the pre-processing pipeline on all images listed in the given
    files list. The dataset of pre-processed images are saved as a numpy array
    to the given destination file.

    The source folder should not contain any other files apart from the images
    to pre-process. The folder name should be of the form 'path/to/folder/'.
    """

    num_files = len(files_list)
    dataset = pd.DataFrame(columns=['label', 'sig_id', 'sig'])
    for row, file in enumerate(files_list):
        print('\r{}/{}'.format(row+1, num_files), end='')
        im = Image.open(file)
        im = preprocess_image(im, final_res, padding)
        label, sig_id =get_type_and_id_from_file(file)
        dataset = dataset.append({
            'label': label,
            'sig_id': sig_id,
            'sig': im.reshape(1, -1)},
            ignore_index=True)

    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    dataset.to_pickle(dest_file)
    print(' - Done!')

def batch_preprocess_aug(files_list, dest_file, final_res, padding, aug_size):
    """ Executes the pre-processing pipeline on all images listed in the given
    files list. The dataset of pre-processed images are saved as a numpy array
    to the given destination file.

    The source folder should not contain any other files apart from the images
    to pre-process. The folder name should be of the form 'path/to/folder/'.
    """

    num_files = len(files_list)
    dataset = pd.DataFrame(columns=['label', 'sig_id', 'sig'])
    for row, file in enumerate(files_list):
        print('\r{}/{}'.format(row+1, num_files), end='')
        im = Image.open(file)
        im = preprocess_image(im, final_res, padding)
        datagen = ImageDataGenerator(featurewise_center=False,
                                     samplewise_center=False,
                                     featurewise_std_normalization=False,
                                     samplewise_std_normalization=False,
                                     zca_whitening=False,
                                     zca_epsilon=1e-06,
                                     rotation_range=45,  # modify this
                                     width_shift_range=10.0,  # modify this
                                     height_shift_range=10.0,  # modify this
                                     brightness_range=None,
                                     shear_range=0.0,
                                     zoom_range=0.1,  # modify this
                                     channel_shift_range=0.5,  # modify this
                                     fill_mode='constant',  # specify this depending on the usecase
                                     cval=1.0,
                                     horizontal_flip=False,
                                     vertical_flip=False,
                                     rescale=1,  # modify this
                                     preprocessing_function=None,
                                     data_format=None,
                                     validation_split=0.0,
                                     dtype=None)
        label, sig_id =get_type_and_id_from_file(file)
        dataset = dataset.append({
            'label': label,
            'sig_id': sig_id,
            'sig': im.reshape(1, -1)},
            ignore_index=True)
        for i in range(aug_size):
            im = datagen.random_transform(im.reshape(1,128,128), seed=None)
            dataset = dataset.append({
                'label': label,
                'sig_id': sig_id,
                'sig': im.reshape(1, -1)},
                ignore_index=True)

    if not os.path.exists(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    dataset.to_pickle(dest_file)
    print(' - Done!')

if __name__ == '__main__':

    final_res = 128
    padding = True

    files = fetch_all_raw()

    # batch_preprocess(files, PATH_TRAIN, final_res, padding)
    batch_preprocess_aug(files, PATH_TRAIN, final_res, padding, aug_size=16)
