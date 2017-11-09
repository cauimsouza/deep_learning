"""

This file contain functionalities for parameters updating.

After evaluation the loss function of the CNN for a batch,
the gradient for each parameter of the CNN allows us to
update the net parameters in different ways, here I provide functions
for the following techniques (all based on gradient descent):
-   Stochastic Gradient Descent updates (sgd)
-   Stochastic Gradient Descent updates with momentum (sgd_momentum)
-   Stochastic Gradient Descent updates with Nesterov momentum (nesterov_momentum)

"""

import theano
import numpy
from collections import OrderedDict

def sgd(params, grads, learning_rate):
    """ Return a dictionary of updates performing simple SGD

    params: list of net parameters as theano.SharedVariables
    grads: list of net gradients as theano.SharedVariables
    learning_rate: learning rate as theano.SharedVariable
    """
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

def sgd_momentum(params, grads, learning_rate, momentum):
    """ Return a dictionary of updates performing SGD with momentum

    params: list of net parameters as theano.SharedVariables
    grads: list of net gradients as theano.SharedVariables
    learning_rate: learning rate as theano.SharedVariable
    """
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(numpy.zeros(value.shape, dtype=value.dtype))
        updates[param] = param + momentum * velocity - learning_rate * grad
        updates[velocity] = momentum * velocity - learning_rate * grad
    return updates

def nesterov_momentum(params, grads, learning_rate, momentum):
    """ Return a dictionary of updates performing SGD with Nesterov momentum

    params: list of net parameters as theano.SharedVariables
    grads: list of net gradients as theano.SharedVariables
    learning_rate: learning rate as theano.SharedVariable
    """
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(numpy.zeros(value.shape, dtype=value.dtype))
        updates[velocity] = momentum * velocity - learning_rate * grad
        updates[param] = param + momentum * velocity - learning_rate * grad
    return updates
