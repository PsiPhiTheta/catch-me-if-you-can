import numpy as np

import preprocessing as pre


def load_clean_train():
    """Utility function to load the cleaned training set with labels.
    Return x_train, y_train as a tuple.
    """

    genuine = np.load(pre.PATH_TRAIN_GENUINE)
    forgeries = np.load(pre.PATH_TRAIN_FORGERIES)

    genuine_labels = np.ones((genuine.shape[0], 1))
    forgeries_labels = np.zeros((forgeries.shape[0], 1))

    x_train = np.vstack((genuine, forgeries))
    y_train = np.vstack((genuine_labels, forgeries_labels))

    return x_train, y_train
