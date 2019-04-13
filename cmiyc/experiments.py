import os

from keras.losses import mse
import keras.backend as K

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

import dataset_utils
import viz_utils
from vanilla_vae import VanillaVae


class Experiment():
    '''
    Runs one single experiment and writes results to a file if specified.
    '''
    IMG_DIR = 'exp_outputs/imgs/'
    EXP_DIR = 'exp_outputs/'

    def __init__(self, args):
        self.args = args
        self.vanilla_vae = VanillaVae(
            args['image_res']*args['image_res'],
            args['intermediate_dim'],
            args['latent_dim'])
        self.vanilla_vae.load_weights(args['save_dir'])

        self.sig_id = args['sig_id']
        self.trained_on = args['trained_on']

        self.acc = None
        self.recal = None
        self.F1 = None
        self.auc_keras = None

        self.x_train = None
        self.x_test  = None
        self.y_train = None
        self.y_test  = None
        self.losses  = None
        self.image_res = args['image_res']

        self.load_data()

        self.classifier_type = args['classifier']
        self.clf = None
        self.fit_clf(classifier=self.classifier_type)

        output = self.get_output(print_output=args['print_output'])

        self.write_to_txt(output)

    def load_data(self):
        # Get the original test set - the stuff that wasn't used to train the VAE.
        '''
        TODO: wasteful to be calling this here,
        ideally we should move this data loading call outside into a loop
        that wraps the Vae training and the Experiments
        But this will do for now
        '''
        _x_train, _y_train, _x_test, _y_test = dataset_utils.load_clean_train_test(vae_sig_type=self.trained_on,
                                              sig_id=self.sig_id,
                                              id_as_label=False)

        # Encode the signatures & extract the losses
        x_encoded = self.vanilla_vae.encoder.predict(_x_test)

        # Extract the losses from each input image
        x_reconstructed = self.vanilla_vae.decoder.predict(x_encoded)
        self.losses = (mse(_x_test, x_reconstructed) * self.image_res).eval(session=K.get_session()).reshape(-1, 1)

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x_encoded, _y_test, test_size=0.2)  # use latent vector
        # x_train, x_test, y_train, y_test = train_test_split(self.losses, _y_test, test_size=0.2)  # use recon_loss
        # x_reconstructed[:, :-1] = self.losses # use both
        # x_train, x_test, y_train, y_test = train_test_split(x_reconstructed, _y_test, test_size=0.2) # use both

        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def fit_clf(self, classifier='forest'):

        # Create and fit the classifier
        if classifier == 'tree':
            clf = DecisionTreeClassifier()
        elif classifier == 'forest':
            clf = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif classifier == 'knn':
            clf = KNeighborsClassifier(n_neighbors=10)
        else:
            raise Exception('Unrecognized classifier type {}'.format(classifier))

        self.clf = clf.fit(self.x_train, self.y_train)

    def get_output(self, print_output=True):
        '''
        Returns experiment output as a dict.
        Args:

            print_output: if True, prints results to stdout
        '''

        ## Test the classifier, save outputs
        y_pred = self.clf.predict(self.x_test)

        ## AUC
        y_proba = self.clf.predict_proba(self.x_test)[:, 1]
        fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(self.y_test, y_proba)
        self.auc_keras = metrics.auc(fpr_keras, tpr_keras)

        self.acc = metrics.accuracy_score(self.y_test, y_pred)
        self.recal = metrics.recall_score(self.y_test, y_pred)
        self.F1 = metrics.f1_score(self.y_test, y_pred)

        output = {
            'sig_id':     self.sig_id,
            'trained_on': self.trained_on,
            'accuracy':   self.acc,
            'recall':     self.recal,
            'f1':         self.F1,
            'fpr_keras':  fpr_keras,
            'tpr_keras':  tpr_keras,
            'thresholds_keras': thresholds_keras,
            'auc_keras':        self.auc_keras
        }

        if print_output:
            print('============== Classification using', self.classifier_type, "==============")
            print('Accuracy:', output['accuracy'])
            print('Recall: ',  output['recall'])
            print('F1 Score:', output['f1'])
            print('-----------------------------------------------------')
            print('ROC Curve / AUC as follows:')
            print('Thresholds: ', output['thresholds_keras'])
            print('False positive rates for each possible threshold:', output['fpr_keras'])
            print('True positive rates for each possible threshold:', output['tpr_keras'])
            print('AUC:', output['auc_keras'])

            save_img = True
            self.plot_AUC(
                output['fpr_keras'],
                output['tpr_keras'],
                output['auc_keras'],
                self.classifier_type,
                save_img=save_img)

            print('')

        return output

    def plot_AUC(self, fpr_keras, tpr_keras, auc_keras, classifier, save_img=False):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label=(classifier, '(area = {:.3f})'.format(auc_keras)))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        if save_img:
            if not os.path.exists(Experiment.IMG_DIR):
                os.makedirs(Experiment.IMG_DIR)
            figname = 'sig_id_{}.png'.format(self.sig_id)
            plt.savefig(Experiment.IMG_DIR+figname)
        else:
            plt.show()

    def write_to_txt(self, output):
        '''
        Write this experiment's output to a text file for future use.
        '''

        if not os.path.exists(Experiment.EXP_DIR):
                os.makedirs(Experiment.EXP_DIR)

        file = Experiment.EXP_DIR + 'sig_id{}_trainedon{}_clf{}_ir{}_id{}_ld{}.txt'.format(
            self.sig_id,
            self.trained_on,
            self.classifier_type,
            self.args['image_res'],
            self.args['intermediate_dim'],
            self.args['latent_dim'])

        with open(file, 'a') as fn:
            fn.write(str(output))

        print('Successfully wrote file to {}'.format(file))


if __name__ == '__main__':

    forest_results = []
    knn_results = []

    # should loop through all sigs avoiding missing ones

    for i in range(1,5):

        args = {
            'classifier': 'forest',
            'sig_id':     i,
            'trained_on': 'genuine',
            'image_res':  128,
            'intermediate_dim': 512,
            'latent_dim': 256,
            'save_dir': 'saved-models/models_genuine_sigid{}_res128_id512_ld256_epoch250.h5'.format(i),
            'print_output': True
        }

        forest_results.append(Experiment(args))

        args = {
            'classifier': 'knn',
            'sig_id': i,
            'trained_on': 'genuine',
            'image_res': 128,
            'intermediate_dim': 512,
            'latent_dim': 256,
            'save_dir': 'saved-models/models_genuine_sigid{}_res128_id512_ld256_epoch250.h5'.format(i),
            'print_output': True
        }

        knn_results.append(Experiment(args))

        plt.clf()

    for i in range(6,7):

        args = {
            'classifier': 'forest',
            'sig_id':     i,
            'trained_on': 'genuine',
            'image_res':  128,
            'intermediate_dim': 512,
            'latent_dim': 256,
            'save_dir': 'saved-models/models_genuine_sigid{}_res128_id512_ld256_epoch250.h5'.format(i),
            'print_output': True
        }

        forest_results.append(Experiment(args))

        args = {
            'classifier': 'knn',
            'sig_id': i,
            'trained_on': 'genuine',
            'image_res': 128,
            'intermediate_dim': 512,
            'latent_dim': 256,
            'save_dir': 'saved-models/models_genuine_sigid{}_res128_id512_ld256_epoch250.h5'.format(i),
            'print_output': True
        }

        knn_results.append(Experiment(args))

        plt.clf()

    for i in range(12,70):

        args = {
            'classifier': 'forest',
            'sig_id':     i,
            'trained_on': 'genuine',
            'image_res':  128,
            'intermediate_dim': 512,
            'latent_dim': 256,
            'save_dir': 'saved-models/models_genuine_sigid{}_res128_id512_ld256_epoch250.h5'.format(i),
            'print_output': True
        }

        forest_results.append(Experiment(args))

        args = {
            'classifier': 'knn',
            'sig_id': i,
            'trained_on': 'genuine',
            'image_res': 128,
            'intermediate_dim': 512,
            'latent_dim': 256,
            'save_dir': 'saved-models/models_genuine_sigid{}_res128_id512_ld256_epoch250.h5'.format(i),
            'print_output': True
        }

        knn_results.append(Experiment(args))

        plt.clf()

    average_acc = 0
    average_rec = 0
    average_F1 = 0

    for i in range(len(forest_results)):
        average_acc += forest_results[i].acc
        average_rec += forest_results[i].recal
        average_F1 += forest_results[i].F1
        average_acc += knn_results[i].acc
        average_rec += knn_results[i].recal
        average_F1 += knn_results[i].F1

    average_acc /= (len(forest_results)*2)
    average_rec /= (len(forest_results)*2)
    average_F1 /= (len(forest_results)*2)

    print('The average accuracy is:', average_acc)
    print('The average recall is:', average_rec)
    print('The average F1 score is:', average_F1) 