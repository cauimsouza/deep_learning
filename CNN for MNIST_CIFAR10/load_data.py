"""

This file contain functionalities for loading datasets

It also contains specific functions for loading the 
MNIST and CIFAR-10 datasets.

"""

import six.moves.cPickle as pickle
import gzip
import os
import sys
import tarfile

import numpy

import theano
import theano.tensor as T

from data_augmentation import data_augmentation_mnist, data_augmentation_cifar10

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def load_cifar10(n_out=10, augmentation=True):
    """ Function that loads and returns the dataset CIFAR-10
    (training set, validation set, testing set) as theano.SharedVariables

    Here we'll be using the first data_batch (10000 images) for training,
    the second data_batch (10000 images) for validation
    and the test_batch for test

    n_out: number of classes to classify
    """
    print('loading CIFAR-10 data with %d classes' % n_out)

    dir_name = os.path.split(os.path.realpath(__file__))[0]

    train_set_path = os.path.join(
        dir_name, 'cifar-10-batches-py', 'data_batch_1')
    valid_set_path = os.path.join(
        dir_name, 'cifar-10-batches-py', 'data_batch_2')
    test_set_path = os.path.join(
        dir_name, 'cifar-10-batches-py', 'test_batch')
    
    def unpickle(file, n_out=10):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        data = numpy.asarray(dict[b'data'])
        labels = numpy.asarray(dict[b'labels'])

        # chosen should contain the indices of up to n_out different image classes
        # feel free to alter it as you want
        chosen = labels < n_out 

        return data[chosen], labels[chosen]

    valid_set_x, valid_set_y = shared_dataset(unpickle(valid_set_path, n_out=n_out))
    test_set_x, test_set_y = shared_dataset(unpickle(test_set_path, n_out=n_out))
    if augmentation:
        train_set_x, train_set_y = shared_dataset(
            data_augmentation_cifar10(unpickle(train_set_path, n_out=n_out)))
    else:
        train_set_x, train_set_y = shared_dataset(
            unpickle(train_set_path, n_out=n_out))
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval
    

def load_mnist(augmentation=True):
    """ Function that loads and returns the dataset MNIST
    (training set, validation set, testing set) as theano.SharedVariables

    The datafile is already organized in training, validation and testing sets
    """
    print('loading MNIST data')
    
    dir_name = os.path.split(os.path.realpath(__file__))[0]
    dir_name = os.path.join(dir_name, 'mnist.pkl.gz')
    # Load the dataset
    with gzip.open(dir_name, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    if augmentation:
        train_set_x, train_set_y = shared_dataset( data_augmentation_mnist(train_set) )
    else:
        train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval
    

def load_data(dataset, n_out=10, augmentation=True):
    """ Function that loads and returns the dataset
    (training set, validation set, testing set) as theano.SharedVariables

    Currently it allows loading only the MNIST and the CIFAR-10 datasets 
    """
    if dataset == 'mnist.pkl.gz':
        return load_mnist(augmentation=True)
    elif dataset == 'cifar-10':
        return load_cifar10(n_out=n_out, augmentation=True)
