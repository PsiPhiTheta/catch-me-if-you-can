import viz_utils
import dataset_utils
from vanilla_vae import VanillaVae
from custom_vae import CustomVae


def train_beta_models(args):

    # Load data
    x_train, y_train, x_test, y_test = dataset_utils.load_clean_train_test(
        vae_sig_type=args['sig_type'],
        sig_id=args['sig_id'],
        id_as_label=False)

    for beta in args['betas']:

        # Instantiate network and set beta param
        vanilla_vae = VanillaVae(args['image_res']**2, args['intermediate_dim'],
                                 args['latent_dim'])
        vanilla_vae.set_beta(beta)

        fn = args['fn'].format(
            args['sig_type'], args['sig_id'], args['image_res'],
            args['intermediate_dim'], args['latent_dim'], args['epochs'],
            beta)

        history = vanilla_vae.fit(x_train, args['val_split'], args['epochs'],
                                  args['batch_size'], args['save_dir'], fn)

def train_beta_models_w_loss_viz(args):
    '''
    Copy of the above function, but to use the CustomVAE class to generate
    loss visualizations.
    '''

    # Load data
    x_train, y_train, x_test, y_test = dataset_utils.load_clean_train_test(
        vae_sig_type=args['sig_type'],
        sig_id=args['sig_id'],
        id_as_label=False)

    for beta in args['betas']:

        args['fn'] = 'alt_models_{}_sigid{}_res{}_id{}_ld{}_epoch{}_{}_b{}.h5'.format(
            args['sig_type'],
            args['sig_id'],
            args['image_res'], 
            args['intermediate_dim'],
            args['latent_dim'],
            args['epochs'],
            'mse',
            str(beta).replace('.', '_') 
        )

        # Instantiate network and set beta param
        custom_vae = CustomVae(
            args['image_res']*args['image_res'], 
            args['intermediate_dim'], 
            args['latent_dim'],
            args['fn'])
        custom_vae.set_beta(beta)

        custom_vae.x_train = x_train # hacky

        history = custom_vae.fit(args['val_split'], args['epochs'],
                                  args['batch_size'], args['save_dir'], args['fn'])

def visualize_samples_from_beta_models(args):
    for beta in args['betas']:

        # Instantiate network and set beta param
        vanilla_vae = VanillaVae(args['image_res']**2, args['intermediate_dim'],
                                 args['latent_dim'])
        vanilla_vae.set_beta(beta)

        # Load weights
        fn = args['fn'].format(
            args['sig_type'], args['sig_id'], args['image_res'],
            args['intermediate_dim'], args['latent_dim'], args['epochs'],
            beta)
        weights = VanillaVae.SAVE_DIR + fn
        vanilla_vae.load_weights(weights)

        # Sample latent space
        viz_utils.generate_from_random_variation(
            vanilla_vae.decoder, args['latent_dim'])


if __name__ == '__main__':

    ######################
    # Betas and latent visualizations
    ######################

    # # Parameters
    # args = {
    #     'sig_id': [1, 2, 3, 4],
    #     'sig_type': 'genuine',
    #     'image_res': 128,
    #     'intermediate_dim': 512,
    #     'latent_dim': 256,
    #     'val_split': 0.1,
    #     'batch_size': 32,
    #     'epochs': 50,
    #     'save_dir': VanillaVae.SAVE_DIR,
    #     # 'betas': [0.25, 0.5, 1, 1.25, 1.5, 1.75, 2],
    #     'betas': [5, 10, 15, 20],
    #     'fn': 'models_{}_sigid{}_res{}_id{}_ld{}_epoch{}_beta{}.h5'
    # }

    # train_beta_models(args)
    # visualize_samples_from_beta_models(args)

    ######################
    # Corresponding loss plots
    ######################

    # Params
    args = {
        'sig_id': [1, 2, 3, 4],
        'sig_type': 'genuine',
        'image_res': 128,
        'intermediate_dim': 512,
        'latent_dim': 256,
        'val_split': 0.1,
        'batch_size': 32,
        'epochs': 50,
        'save_dir': CustomVae.SAVE_DIR,
        # 'betas': [0.25, 0.5, 1, 1.25, 1.5, 1.75, 2],
        'betas': [5, 10, 15, 20],
    }

    train_beta_models_w_loss_viz(args)


