import os
import sys
import timeit
import six.moves.cPickle as pickle

import numpy

import theano
import theano.tensor as T

from cnn_layers import HiddenLayer, ConvPoolLayer, LogisticRegression
from load_data import load_data
from updates import nesterov_momentum

class LeNet(object):
    
    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights

    def __setstate__(self, weights):
        for p, w in zip(self.params, weights):
            p.set_value(w)

    def __init__(self, input, rng, image_shape=(1, 28, 28),
                 nkerns=[20, 50, 50], batch_size=500, n_out=10):
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('building the model')

        # Reshape matrix of rasterized images of shape (batch_size, scales, height, width)
        # to a 4D tensor, compatible with our ConvPoolLayer
        # (1, 28, 28) is the size of MNIST images.
        # (3, 32, 32) is the size of CIFAR-10 images.
        self.layer0_input = input.reshape((batch_size,) + image_shape)

        # Construct the first convolutional pooling layer
        # with nkerns[0] kernels and 5x5 receptive field size
        map_height = image_shape[1]  # 28 for MNIST
        map_width = image_shape[2]  # 28 for MNIST
        self.layer0 = ConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size,) + image_shape,
            filter_shape=(nkerns[0], image_shape[0], 5, 5),
            poolsize=(2, 2)
        )
        map_height = (map_height - 5 + 1) // 2  # 12 for MNIST
        map_width = (map_width - 5 + 1) // 2  # 12 for MNIST

        # Construct the second convolutional pooling layer
        # with nkerns[1] kernels and 5x5 receptive field size
        self.layer1 = ConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], map_height, map_width),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )
        map_height = (map_height - 5 + 1) // 2  # 4 for MNIST
        map_width = (map_width - 5 + 1) // 2  # 4 for MNIST

        # Construct the third convolutional layer
        # filtering does not reduce the image
        # no maxpooling is applied in this layer
        self.layer2 = ConvPoolLayer(
            rng,
            input=self.layer1.output,
            image_shape=(batch_size, nkerns[1], map_height, map_width),
            filter_shape=(nkerns[2], nkerns[1], 3, 3),
            border_mode='half',
            max_pool=False
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[2] * 4 * 4) for MNIST,
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer3_input = self.layer2.output.flatten(2)

        # construct a fully-connected ReLu layer
        self.layer3 = HiddenLayer(
            rng,
            input=self.layer3_input,
            n_in=nkerns[2] * map_height * map_width,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected ReLu layer
        self.layer4 = LogisticRegression(
            input=self.layer3.output, n_in=500, n_out=n_out)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer4.params + self.layer3.params + \
            self.layer2.params + self.layer1.params + self.layer0.params
        self.batch_size = batch_size


def fit(dataset='mnist.pkl.gz', image_shape=(1, 28, 28), nkerns=[20, 50, 50],
        learning_rate=0.1, momentum=0.9, L1_reg=0., L2_reg=0.,
        n_epochs=200, n_out=10, batch_size=500, seed=8000):
    """Trains LeNet CNN classifier for MNIST/CIFAR-10 dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                        gradient with Nesterov momentum)
    
    :type momentum: float
    :param momentum: momentum used (factor for the stochastic
                        gradient with Nesterov momentum)
    
    :type L1_reg: float
    :param L1_reg: L1 regularization strength used
    
    :type L2_reg: float
    :param L2_reg: L2 regularization strength used

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: name of the dataset used for training /testing (MNIST or CIFAR-10)

    :type image_shape: tuple of ints
    :param image_shape: shape of images in the dataset

    :type n_out: int
    :param n_out: number of image classes

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer (3 layers)

    :type batch_size: int
    :param batch_size: number of images per minibatch

    ::type seed: int
    """
    rng = numpy.random.RandomState(seed)
    
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    datasets = load_data(dataset, n_out=n_out)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    lenet = LeNet(
        input=x,
        rng=rng,
        image_shape=image_shape,
        nkerns=nkerns,
        batch_size=batch_size,
        n_out=n_out
    )

    # the cost we minimize during training is the NLL of the model
    cost = lenet.layer4.negative_log_likelihood(y) + \
        lenet.layer4.reg_loss(L1_reg=L1_reg, L2_reg=L2_reg) + \
        lenet.layer3.reg_loss(L1_reg=L1_reg, L2_reg=L2_reg) + \
        lenet.layer2.reg_loss(L1_reg=L1_reg, L2_reg=L2_reg) + \
        lenet.layer1.reg_loss(L1_reg=L1_reg, L2_reg=L2_reg) + \
        lenet.layer0.reg_loss(L1_reg=L1_reg, L2_reg=L2_reg)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        lenet.layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        lenet.layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of gradients for all model parameters
    grads = T.grad(cost, lenet.params)

    # transform learning_rate into shared variable
    learning_rate = theano.shared(learning_rate)

    # transform momentum into shared variable
    momentum = theano.shared(momentum)

    # SGD with Nesterov momentum 
    updates = nesterov_momentum(
        lenet.params, grads, learning_rate, momentum)

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('training')
    
    # early-stopping parameters
    patience = 10000  
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)

            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                    in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                    improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                        'best model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                        test_score * 100.))
                    
                    # save the best model
                    dir_name = os.path.split(os.path.realpath(__file__))[0]
                    with open(os.path.join(dir_name, 'lenet_weights.pkl'), 'wb') as f:
                        pickle.dump(lenet.__getstate__(), f, protocol=-1)

            if patience <= iter:
                done_looping = True
                break
        
        # learning_rate: 5 % decrease after 10 epochs "step decay"
        # momentum: 5 % increase after 10 epochs "step "
        if epoch % 10 == 0:
            learning_rate.set_value(learning_rate.get_value() * 0.95)
            momentum.set_value(momentum.get_value() * 1.05)

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
        'with test performance %f %%' %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


def predict(test_dataset='mnist.pkl.gz', image_shape=(1, 28, 28),
          nkerns=[20, 50, 50], n_out=10, batch_size=500, seed=8000):
    """Uses trained LeNet CNN for MNIST / CIFAR-10 dataset classification
       Returns numpy.array with the predicted classes
    """
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    datasets = load_data(dataset=test_dataset, n_out=n_out)

    test_set_x = datasets[2][0]

    N = len(test_set_x.get_value()) // batch_size

    classifier = LeNet(
        input=x,
        rng=numpy.random.RandomState(seed),
        image_shape=image_shape,
        nkerns=nkerns,
        batch_size=batch_size
    )

    # loading model weights
    dir_name = os.path.split(os.path.realpath(__file__))[0]
    with open(os.path.join(dir_name, 'lenet_weights.pkl'), 'rb') as f:
        classifier.__setstate__(pickle.load(f))

    predict_labels = theano.function(
        [index],
        classifier.layer4.y_pred,
        givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size]
    })

    p = [predict_labels(i) for i in range(N)]
    return numpy.array(p).reshape(-1)


def train_and_predict(dataset='mnist.pkl.gz', image_shape=(1, 28, 28), 
                      nkerns=[20, 50, 50], learning_rate=0.1, momentum=0.9, 
                      L1_reg=0., L2_reg=0., n_epochs=200, 
                      n_out=10, batch_size=100, seed=8000):
    # train LeNet and save weights
    fit(dataset=dataset, image_shape=image_shape, nkerns=nkerns,
        learning_rate=learning_rate, momentum=momentum, L1_reg=L1_reg, L2_reg=L2_reg,
        n_epochs=n_epochs, n_out=n_out, batch_size=batch_size, seed=seed)
    

    # load weights from pre-trained LeNet CNN and predict image classes
    predictions = predict(test_dataset=dataset, image_shape=image_shape,
                          nkerns=nkerns, n_out=n_out, batch_size=batch_size,
                          seed=seed)
    
    return predictions

if __name__ == '__main__':
    # train LeNet and predict classes
    
    if len(sys.argv) < 2:
    	print("Invalid argument: please pass 'mnist' or 'cifar10' as argument")
    else:
        if sys.argv[1] == 'mnist':
            predictions = train_and_predict(dataset='mnist.pkl.gz', image_shape=(1, 28, 28),
		        nkerns=[20, 50, 50], learning_rate=0.1, momentum=0.9, L1_reg=0., L2_reg=0.,
		        n_epochs=200, n_out=10, batch_size=100, seed=8000)
		    
            print(predictions)
		
        elif sys.argv[1] == 'cifar10':
            n_out = 10
            if len(sys.argv) > 2:
                try:
                    n_out = int(sys.argv[2])
                except:
                    print('Invalid argument: the second argument should be an integer <= 10')
                    sys.exit()
            predictions = train_and_predict(dataset='cifar-10', image_shape=(3, 32, 32),
		        nkerns=[20, 50, 50], learning_rate=0.1, momentum=0.9,L1_reg=0., L2_reg=0.,
		        n_epochs=200, n_out=n_out, batch_size=100, seed=8000)
            
            print(predictions)
		
        else:
            print("Invalid argument: dataset should be 'mnist' or 'cifar10'")
