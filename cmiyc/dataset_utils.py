import numpy as np
import pandas as pd

import preprocessing as pre


def load_clean_train(sig_type='all', sig_id='all'):
    """Utility function to load the cleaned training set with labels.
    Return x_train, y_train as a tuple.

    Input:
        sig_type: 'genuine', 'forgery', or 'all' (default)
        sig_id: integer or 'all' (default)
    """

    df = pd.read_pickle(pre.PATH_TRAIN)
    if sig_type == 'genuine':
        df = df[df['label'] == 1]
    elif sig_type == 'forgery':
        df = df[df['label'] == 0]

    if not sig_id == 'all':
        df = df[df['sig_id'] == sig_id]

    print(df)
    return np.vstack(df['sig'].to_numpy()), df['label'].to_numpy()
