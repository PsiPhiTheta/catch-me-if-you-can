from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Plots the training and validation losses
    """
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    # plt.yscale(value='log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_encoded_2d(x, label):
    """
    Plot the encoded x given the label (sig_id or type).
    The encoded dimensionality (x) should be 2.
    """
    plt.scatter(x[:, 0], x[:, 1],
                c=label,
                cmap='jet',
                marker='o',
                s=100,
                alpha=0.5)
    plt.colorbar()
    plt.show()


def plot_encoded_3d(x, label):
    """
    Plot the encoded x given the label (sig_id or type).
    The encoded dimensionality (x) should be 3.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2],
               c=label,
               cmap='jet',
               marker='o',
               s=100,
               alpha=0.5)
    plt.show()


def plot_manifolds_2d(decoder, n=8, size=128, std_dev=1):
    grid_x = np.linspace(-8, 8, n)
    grid_y = np.linspace(-8, 8, n)
    figure = np.zeros((size * n, size * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * std_dev
            x_decoded = decoder.predict(z_sample)
            im = x_decoded.reshape(size, size)
            figure[i * size: (i + 1) * size, j * size: (j + 1) * size] = im

    plt.figure()
    plt.imshow(figure, cmap='gray')
    plt.show()


def plot_random_x(x, n=8, size=128):
    figure = np.zeros((size * n, size * n))
    sample = np.random.randint(0, len(x), (n, n))
    for i, row in enumerate(sample):
        for j, ix in enumerate(row):
            figure[i*size : (i+1)*size, j*size : (j+1)*size] = x[ix].reshape(size, size)
    plt.figure()
    plt.imshow(figure, cmap='gray')
    plt.show()

