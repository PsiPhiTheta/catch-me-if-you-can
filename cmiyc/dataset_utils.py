import numpy as np

import preprocessing as pre


def load_clean_train(sig_type='all'):
    """Utility function to load the cleaned training set with labels.
    Return x_train, y_train as a tuple.

    Input: 
        sig_type: 'genuine', 'forgery', or 'all' (default)
    """

    if sig_type in ['genuine', 'all']:
        genuine = np.load(pre.PATH_TRAIN_GENUINE)
        genuine_labels = np.ones((genuine.shape[0], 1))

        if sig_type == 'genuine':
            return genuine, genuine_labels 
    
    if sig_type in ['forgery', 'all']:
        forgeries = np.load(pre.PATH_TRAIN_FORGERIES)
        forgeries_labels = np.zeros((forgeries.shape[0], 1))

        if sig_type == 'forgery':
            return forgeries, forgeries_labels

    x_train = np.vstack((genuine, forgeries))
    y_train = np.vstack((genuine_labels, forgeries_labels))

    return x_train, y_train
