import numpy as np
import pandas as pd

import preprocessing as pre


def load_clean_train(sig_type='all', sig_id='all', id_as_label='false'):
    """Utility function to load the cleaned training set with labels.
    Return x_train, y_train as a tuple.

    Input:
        sig_type: 'genuine', 'forgery', or 'all' (default)
        sig_id: integer, list of integer or 'all' (default)
    """

    sig_id = [sig_id] if isinstance(sig_id, int) else sig_id

    df = pd.read_pickle(pre.PATH_TRAIN)
    if sig_type == 'genuine':
        df = df[df['label'] == 1]
    elif sig_type == 'forgery':
        df = df[df['label'] == 0]

    if not sig_id == 'all':
        df = df[df['sig_id'].isin(sig_id)]

    labels = 'sig_id' if id_as_label else 'label'
    return np.vstack(df['sig'].to_numpy()), df[labels].to_numpy()
