import numpy as np
import pandas as pd
import random
import glob

from sklearn.model_selection import train_test_split

import preprocessing as pre


def load_clean_train(sig_type='all', sig_id='all', id_as_label='false', frac=0.5):
    """Utility function to load the cleaned training set with labels.
    Return x_train, y_train as a tuple.

    Input:
        sig_type: 'genuine', 'forgery', or 'all' (default)
        sig_id: integer, list of integer or 'all' (default)
    """

    sig_id = [sig_id] if isinstance(sig_id, int) else sig_id

    df = pd.read_pickle(pre.PATH_ALL)
    if sig_type == 'genuine':
        df = df[df['label'] == 1]
    elif sig_type == 'forgery':
        df = df[df['label'] == 0]

    if not sig_id == 'all':
        df = df[df['sig_id'].isin(sig_id)]

    labels = 'sig_id' if id_as_label else 'label'
    return np.vstack(df['sig'].to_numpy()), df[labels].to_numpy().astype('int')

def load_clean_train_test(vae_sig_type='genuine', sig_id=1, id_as_label='false', frac=0.5, random_state=4):
    '''
    For a single sig_id, splits out train and test data as follows:

    train: frac * {genuine examples from the Train folder}
    test: (1-frac) * {genuine examples from the Train folder} + 
                      {fake examples from the Train folder}

    Args:

    vae_sig_type: the signature type that is used to train the VAE (default genuine)
    sig_id: cannot be 'all' here. Must be int

    Random state is set to 4 for reproducibility.

    Returns:
    x_train, y_train, x_test, y_test
    
    '''

    sig_id = [sig_id] if isinstance(sig_id, int) else sig_id

    df = pd.read_pickle(pre.PATH_ALL)
    if vae_sig_type == 'genuine':
        vae_df = df[df['label'] == 1]
        exp_df = df[df['label'] == 0]
    elif vae_sig_type == 'forgery':
        vae_df = df[df['label'] == 0]
        exp_df = df[df['label'] == 1]

    
    vae_df = df[df['sig_id'].isin(sig_id)]

    # split into train and test; train needs to be vae_sig_type only
    vae_df_train, vae_df_gen_test = train_test_split(vae_df, test_size=1.0-frac, random_state=random_state)

    exp_df_test = exp_df.append(vae_df_gen_test)
    
    # Shuffle
    exp_df_test = exp_df_test.sample(frac=1).reset_index(drop=True)

    return vae_df_train, exp_df_test

def get_sig_ids(sig_type='genuine', mode='folder'):
    '''
    Get a list of training sig_ids to use when training all our VAEs.

    Only use what's in the Training folders as your train set.

    Arguments:

        sig_type: 'genuine' or 'forgery'
    '''

    df = pd.read_pickle(pre.PATH_ALL)

    if mode == 'all':
        # Returns all IDs regardless of whether they are in test or train folders
        return df['sig_id'].unique()

    elif mode == 'folder':
        # Get the sig_ids that correspond to the sigs in the Train folder
        forgery_paths = ['data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries/*.*']
        genuine_paths = ['data/raw/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine/*.*']
    
        if sig_type == 'genuine': 
            paths = genuine_paths
        elif sig_type == 'forgery':
            paths = forgery_paths
        
        files = []
        for path in paths:
            files += glob.glob(path, recursive=True)

        # Parse filenames to ids
        train_ids = []
        for file in files:
            label, sig_id = pre.get_type_and_id_from_file(file)
            train_ids.append(sig_id)

        return list(set(train_ids))

    else:
        print('get_train_sig_ids called with invalid mode={}'.format(mode))

if __name__ == "__main__":
    pass
