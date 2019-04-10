import numpy as np
import pandas as pd
import random
import glob

import preprocessing as pre


def load_clean_train(sig_type='all', sig_id='all', id_as_label='false'):
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

def get_train_sig_ids(sig_type="genuine", mode='random', frac=0.25, seed=4):
    '''
    Get a list of training sig_ids to use when training all our VAEs.

    preprocessing.py currently mashes all data (train+test) into a single .pkl
    saved at PATH_ALL.

    mode=random (default) picks a random subset of sig_ids and 
    grabs the instances of data for those which are sig_type (genuine by default)

    mode=folder will just use whatever was in the Training folders as your train set.

    Default seed=4 is set for reproducibility.

    Arguments:

        sig_type: 'genuine' or 'forgery'
        mode:     'random' or 'folder'
    '''

    df = pd.read_pickle(pre.PATH_ALL)

    if mode == 'all':
        # Returns all IDs regardless of whether they are in test or train sets
        return df['sig_id'].unique()

    elif mode == 'random':
        random.seed(seed)
        all_ids = df['sig_id'].unique()
        random.shuffle(all_ids)

        up_to = int(float(frac) * len(all_ids))

        return all_ids[0:up_to]

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
        print('get_train_sig_ids called with invalid mode={}, using random'.format(mode))
    
    df = pd.read_pickle(pre.PATH_ALL)
    return sorted(df['sig_id'].unique())

def test_get_train_sig_ids():
    print("Running unit test for test_get_train_sig_ids()")
    print("-" * 20)
    
    print("Random subset of sig_ids:")
    train_list = get_train_sig_ids(sig_type="genuine", mode='random', frac=0.25, seed=4)
    print(train_list)
    print('\n')

    print("All sig_ids")
    all_list = get_train_sig_ids(sig_type="genuine", mode='all', frac=0.25, seed=4)
    print(all_list)
    print('\n')

    print("Train folder sig_ids")
    folder_list = get_train_sig_ids(sig_type="genuine", mode='folder', frac=0.25, seed=4)
    print(folder_list)
    print('\n')

if __name__ == "__main__":
    pass
    # test_get_train_sig_ids()
