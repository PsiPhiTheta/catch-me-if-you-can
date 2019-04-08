from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import dataset_utils
import viz_utils
from vanilla_vae import VanillaVae

def classification(sig_id=1, classifier='tree'):

    image_res = 128
    intermediate_dim = 512
    latent_dim = 256
    save_dir = 'saved-models/models.h5'

    # Load model
    vanilla_vae = VanillaVae(image_res*image_res, intermediate_dim, latent_dim)
    vanilla_vae.load_weights(save_dir)

    # Load data
    x, y = dataset_utils.load_clean_train(sig_type='all',
                                          sig_id=sig_id,
                                          id_as_label=False)

    # Encode the signatures
    x = vanilla_vae.encoder.predict(x)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # print(x_train.shape, x_train, y_train.shape, y_train)

    # Create and train the classifier
    if classifier == 'tree':
        clf = DecisionTreeClassifier()
    elif classifier == 'knn':
        clf = KNeighborsClassifier()
    else:
        raise Exception('Unrecognized classifier type {}'.format(classifier))

    clf = clf.fit(x_train, y_train)

    # # Test the classifier
    y_pred = clf.predict(x_test)
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred))
    print('F1 Score:', metrics.f1_score(y_test, y_pred))


if __name__ == '__main__':
    classification(sig_id=1, classifier='tree')
    classification(sig_id=1, classifier='knn')
