from keras.preprocessing.image import ImageDataGenerator

def get_datagen():
    return ImageDataGenerator(featurewise_center=False,
                              samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              zca_epsilon=1e-06,
                              rotation_range=20,  # modify this
                              width_shift_range=10.0,  # modify this
                              height_shift_range=10.0,  # modify this
                              brightness_range=None,
                              shear_range=0.0,
                              zoom_range=0.1,  # modify this
                              channel_shift_range=0,  # modify this
                              fill_mode='constant',  # specify this depending on the usecase
                              cval=1.0,
                              horizontal_flip=False,
                              vertical_flip=False,
                              rescale=None,  # modify this
                              preprocessing_function=None,
                              data_format=None,
                              validation_split=0.0,
                              dtype=None)
