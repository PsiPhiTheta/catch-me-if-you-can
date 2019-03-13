import matplotlib.pyplot as plt
import numpy as np
import glob

from scipy.stats import mode
from PIL import Image, ImageFilter


def preprocess_image(image, final_res=256, padding=False, plot=False):
    """ Pre-process a single image.

    The input should be an Image object and the output is a numpy array.

    If `padding` is set to true, the image is padded to be a square before being
    resized to a square of lower resolution (default 256x256 px)

    If `plot` is set, a plot of the images after each step of the pre-processing
    pipeline will be generated
    """

    # Keep track of changes
    images = [('original', image, None)]

    # Convert to gray scale
    image = image.convert('L')
    images.append(('gray scale', image, 'gray'))

    # Find the mode (background color) and convert to black and white with a
    # threshold based on it.
    threshold = int(0.90 * mode(image.getdata())[0][0])
    lut = [0] * threshold + [255] * (256 - threshold)
    image = image.point(lut)
    images.append(('black & white', image, 'gray'))

    # Add padding to form a square if option is set
    if padding:
        image = pad_image_square_center(image)
        images.append(('padding', image, 'gray'))

    # Resize with bilinear interpolation (best results)
    image = image.resize((final_res, final_res), Image.BILINEAR)
    images.append(('resize', image, 'gray'))

    # Plot images if option is set
    if plot:
        fig = plt.figure()
        fig.tight_layout()
        n = len(images)
        for i, (title, im, cmap) in enumerate(images):
            ax = plt.subplot(1, n, i+1)
            ax.set_title(title)
            ax.imshow(im, cmap)
        plt.show()

    # Convert to numpy array
    return np.array(image.getdata()).reshape((final_res, final_res))


def pad_image_square_center(image):
    """ Pads the given image so that the original is centered on a square.
    """
    new_size = max(image.size)
    new_image = Image.new(image.mode, (new_size, new_size), 'white')
    position = ((new_size - image.size[0]) // 2,
                (new_size - image.size[1]) // 2)
    new_image.paste(image, position)
    return new_image


if __name__ == '__main__':

    # Load the image
    im = Image.open('data/trainingSet/OfflineSignatures/Dutch/TrainingSet/'
                    'Offline Genuine/002_01.PNG')

    # Preprocess the image
    r = preprocess_image(im, padding=True, plot=True)


