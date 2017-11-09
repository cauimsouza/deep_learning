"""

This file contain functionalities for data augmentation.
Since this is dataset dependent, the functions defined here
are better for the MNIST dataset.

For the MNIST dataset, I thought that vertical/horizontal flipping would
be a good idea: the images created are very similar to the real images and thus
easily classifiables.
The images that were used are those whose flipping produce probable images, e.g,
images corresponding to '0's, '1's and '8's

For the CIFAR dataset, I thought that vertical/horizontal flipping could be done
for all images.

"""

import numpy

def left_right_flip(imgs, image_shape):
    """ Returns horizontally flipped images

    :type imgs: numpy 2D array of floats 
    :param imgs: each line representes an image

    """
    flip_imgs = numpy.zeros_like(imgs)

    for i in range(len(imgs)):
        flip_img = numpy.reshape(imgs[i, :], image_shape)
        for color in range(flip_img.shape[0]):
            for j in range(flip_img.shape[1]):
                flip_img[color, j, :] = flip_img[color, j, ::-1]
        flip_img = numpy.reshape(flip_img, (1, numpy.prod(image_shape)))
        flip_imgs[i, :] = flip_img

    return flip_imgs


def up_down_flip(imgs, image_shape):
    """ Returns vertically flipped images

    :type imgs: numpy 2D array of floats 
    :param imgs: each line representes an image

    """
    flip_imgs = numpy.zeros_like(imgs)
    for i in range(len(flip_imgs)):
        flip_img = numpy.reshape(imgs[i,:], image_shape)
        for color in range(flip_img.shape[0]):
            flip_img[color] = flip_img[color, ::-1] 
        flip_img = numpy.reshape(flip_img, (1, numpy.prod(image_shape)))
        flip_imgs[i,:] = flip_img

    return flip_imgs

def data_augmentation_cifar10(data):
    """ Returns set of images after data augmentation

    :type data: tuple of floats
    :param data: the first element consists of 2D numpy array containing images
                 the second element consists of 1D numpy array containing labels

    """
    x = data[0]
    y = data[1]

    x_lr_flip = left_right_flip(x, image_shape=(3, 32, 32))
    x_ud_flip = up_down_flip(x, image_shape=(3, 32, 32))

    x = numpy.concatenate((x, x_lr_flip))
    y = numpy.concatenate((y, y))

    shuffle_idx = numpy.arange(len(x))
    numpy.random.shuffle(shuffle_idx)

    x = x[shuffle_idx]
    y = y[shuffle_idx]

    return x, y



def data_augmentation_mnist(data):
    """ Returns set of images after data augmentation

    :type data: tuple of floats
    :param data: the first element consists of 2D numpy array containing images
                 the second element consists of 1D numpy array containing labels

    """

    x = data[0]
    y = data[1]

    x_0 = x[y == 0]
    x_1 = x[y == 1]
    x_8 = x[y == 8]

    x_0_left_right = left_right_flip(x_0, image_shape=(1, 28, 28))
    x_1_left_right = left_right_flip(x_1, image_shape=(1, 28, 28))
    x_8_left_right = left_right_flip(x_8, image_shape=(1, 28, 28))

    x_0_up_down = up_down_flip(x_0, image_shape=(1, 28, 28))
    x_1_up_down = up_down_flip(x_1, image_shape=(1, 28, 28))
    x_8_up_down = up_down_flip(x_8, image_shape=(1, 28, 28))

    x = numpy.concatenate((x, x_0_left_right, x_1_left_right,
                            x_8_left_right, x_0_up_down, x_1_up_down, x_8_up_down))
    y = numpy.concatenate((y, numpy.zeros(len(x_0)), numpy.ones(
        len(x_1)), 8 * numpy.ones(len(x_8)), numpy.zeros(len(x_0)), numpy.ones(
        len(x_1)), 8 * numpy.ones(len(x_8))))

    shuffle_idx = numpy.arange(len(x))
    numpy.random.shuffle(shuffle_idx)

    x = x[shuffle_idx]
    y = y[shuffle_idx]

    return x, y
