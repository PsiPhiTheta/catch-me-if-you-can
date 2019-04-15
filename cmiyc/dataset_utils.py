import numpy as np
import pandas as pd
import random
import glob

from sklearn.model_selection import train_test_split

import preprocessing as pre

TYPE_MAP = {'genuine': 1,
            'forgery': 0}


def old_load_clean_train(sig_type='all', sig_id='all', id_as_label=False, frac=0.5):
    """
    NOT IN USE - keeping just in case

    Utility function to load the cleaned training set with labels.
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

    vae_sig_type: the signature type that is used to train the VAE. 
    (default = 'genuine', reverse the description above if not 'genuine')
    sig_id: cannot be 'all' here. Must be int

    Random state is set to 4 for reproducibility.

    Returns:
    x_train, y_train, x_test, y_test
    
    '''

    sig_id = [sig_id] if isinstance(sig_id, int) else sig_id

    df = pd.read_pickle(pre.PATH_ALL)

    df = df[df['sig_id'].isin(sig_id)]

    if vae_sig_type == 'genuine':
        vae_df = df[df['label'] == 1]
        exp_df = df[df['label'] == 0]
    elif vae_sig_type == 'forgery':
        vae_df = df[df['label'] == 0]
        exp_df = df[df['label'] == 1]

    # split into:
    # - A train set of genuines for the VAE
    # - A test set of genuines + all fakes for the Experiment test
    vae_df_train, vae_df_genuine_test = train_test_split(vae_df, test_size=1.0-frac, random_state=random_state)
    exp_df_test = exp_df.append(vae_df_genuine_test)
    
    # Shuffle Experiment test set
    exp_df_test = exp_df_test.sample(frac=1).reset_index(drop=True)

    # Unit test to make sure our splits are sound
    # split_test(vae_df_train, exp_df_test, sig_id, vae_sig_type, verbose=False)

    if id_as_label:
        labels = 'sig_id'
    else: 
        labels = 'label' 

    x_train = np.vstack(vae_df_train['sig'].to_numpy())
    y_train = vae_df_train[labels].to_numpy().astype('int')

    assert list(set(y_train)) == [TYPE_MAP[vae_sig_type]], "y_train contains bad values: {}".format(list(set(y_train)))

    x_test = np.vstack(exp_df_test['sig'].to_numpy())
    y_test = exp_df_test[labels].to_numpy().astype('int')

    assert (1 in list(set(y_test)) and 0 in list(set(y_test))), "y_test contains bad values: {}".format(list(set(y_test)))

    return x_train, y_train, x_test, y_test


def get_sig_ids(sig_type='genuine', mode='all'):
    '''
    Get a list of training sig_ids to use when training all our VAEs.

    Arguments:

        sig_type: 'genuine' or 'forgery'
        mode:     'folder' or 'all'. 

        If 'folder', only use what's in the Training folders as your train set.
        Otherwise, gets all ids.

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


def split_test(train_split, test_split, sig_id, vae_sig_type, verbose=True):
    '''
    Quick unit test to check that the split in load_clean_train_test() works as expected.
    '''
    if verbose:
        print("Running test_load_clean_train_test() \n" + "-"*20)

    test_pass = True

    #################
    # Check x_train['sig_id'] and x_test['sig_id'] only contain 
    # the correct sig_id.
    #################
    x_train_ids = train_split['sig_id'].unique()
    x_test_ids  = test_split['sig_id'].unique()
    
    if x_train_ids != [sig_id]:
        test_pass = False
        if verbose:
            print("train_split contains invalid sig_id values: {}".format(x_train_ids))

    if x_test_ids != [sig_id]:
        test_pass = False
        if verbose:
            print("test_split contains invalid sig_id values: {}".format(x_test_ids))

    #################
    # Check x_train only contains genuines
    # Check x_test contains a mix of genuine and forgery
    #################
    x_train_types = train_split['label'].unique().tolist()
    x_test_types  = test_split['label'].unique().tolist()

    if x_train_types != [TYPE_MAP[vae_sig_type]]:
        test_pass = False
        if verbose:
            print("train_split['label'] contains invalid values - expected only {}, got {}".format(
                [TYPE_MAP[vae_sig_type]],
                x_train_types))

    if not (1 in x_test_types and 0 in x_test_types):
        test_pass = False
        if verbose:
            print("test_split['label'] does not contain both {}, got {}".format(
                [0,1],
                x_test_types))

    if test_pass:
        if verbose:
            print("[OK] split_test() passes.")
    else:
        assert test_pass, "[TEST FAILURE] split_test() did not pass."


if __name__ == "__main__":
    pass