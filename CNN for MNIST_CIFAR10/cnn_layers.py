"""
This file contain classes of three types of CNN layers:

-   HiddenLayer: fully connected layer
-   ConvPoolLayer:  convolutional layer with the option of adding a max pooling layer over
-   LogisticRegression: final CNN layer for classification images

"""

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        """
        Typical hidden layer of an MLP (multilayer perceptron): units are fully-connected and have
        relu, tanh or sigmoid activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here by default is ReLu

        Hidden unit activation is given by: relu(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-12./(n_in+n_hidden)) and sqrt(12./(n_in+n_hidden))
        # for relu activation function (He initialization)
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4/sqrt(2) times larger initial weights for sigmoid
        #        compared to relu, and 1/sqrt(2) times initial weights for tanh
        #        compare to relu
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(12. / (n_in + n_out)),
                    high=numpy.sqrt(12. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 2 * numpy.sqrt(2)
            elif activation == theano.tensor.tanh:
                W_values /= numpy.sqrt(2)

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # parameters of the model
        self.params = [self.W, self.b]

    # reguralization loss of this layer
    def reg_loss(self, L1_reg=0., L2_reg=0.):
        return L1_reg * abs(self.W).sum() + L2_reg * (self.W ** 2).sum()


class ConvPoolLayer(object):
    """ConvPool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape,
                 image_shape, border_mode='valid',
                 max_pool=True, poolsize=(2, 2),
                 activation=T.nnet.relu):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        
        :type border_mode: string
        :param border_mode: type of zero padding for conv2d

        :type no_pool: bool
        :param no_pool: if True then do max pooling after convolution

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        if max_pool:
            fan_out = (
                filter_shape[0] * numpy.prod(filter_shape[2:])) // numpy.prod(poolsize)
        else:
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])

        # initialize weights with random weights
        # the initial distribution can vary depending on activation function used
        W_bound = numpy.sqrt(12. / (fan_in + fan_out))
        if activation is T.nnet.sigmoid:
            W_bound *= 2 * numpy.sqrt(2)
        elif activation is T.tanh:
            W_bound /= numpy.sqrt(2)
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=border_mode
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ws=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = (
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x') if max_pool
            else conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def reg_loss(self, L1_reg=0., L2_reg=0.):
        return L1_reg * abs(self.W).sum() + L2_reg * (self.W ** 2).sum()


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression layer

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize weights
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        # initialize bias
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        
    
    def reg_loss(self, L1_reg=0., L2_reg=0.):
        return L1_reg * abs(self.W).sum() + L2_reg * (self.W ** 2).sum()

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
