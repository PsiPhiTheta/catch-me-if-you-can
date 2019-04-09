from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import dataset_utils
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
    elif classifier == 'forest':
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif classifier == 'knn':
        clf = KNeighborsClassifier()
    else:
        raise Exception('Unrecognized classifier type {}'.format(classifier))

    clf = clf.fit(x_train, y_train)

    # # Test the classifier
    print('============== Classification using', classifier, "==============")
    y_pred = clf.predict(x_test)
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred))
    print('F1 Score:', metrics.f1_score(y_test, y_pred))
    fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test, y_pred)
    print('-----------------------------------------------------')
    print('ROC Curve / AUC as follows:')
    print('False positive rates for each possible threshold:', fpr_keras)
    print('True positive rates for each possible threshold:', tpr_keras)
    auc_keras = metrics.auc(fpr_keras, tpr_keras)
    print('AUC:', auc_keras)
    plot_AUC(fpr_keras, tpr_keras, auc_keras, classifier)
    print('')

def plot_AUC(fpr_keras, tpr_keras, auc_keras, classifier):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=(classifier, '(area = {:.3f})'.format(auc_keras)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    classification(sig_id=1, classifier='tree')
    classification(sig_id=1, classifier='forest')
    classification(sig_id=1, classifier='knn')
