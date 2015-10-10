"""
    This file implements the neural network layers needed for the project.
"""


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import LogisticRegression, load_data

try:
    import theano.tensor.nnet.relu as relu
except ImportError:
    print "No theano.tensor.nent.relu found, using self-defined version"

    def relu(x):
        return T.switch(x > 0, x, 0)


gpu_usage = True

try:
    from pynvml import *
except ImportError:
    print 'package nvidia-ml-py doesn''t exist, cannot show GPU usage data'
    gpu_usage = False


class FCLayer(object):
    def __init__(self, rng, data, n_in, n_out,
                 W=None, b=None, activation=T.tanh):
        """
        A fully connected layer in the neural network.
        Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type data: theano.tensor.dmatrix
        :param data: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = data
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # parameters of the model
        self.params = [self.W, self.b]

        lin_output = T.dot(self.input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        #self.in_shape = n_in

        self.n_out = n_out

#    rand = theano.tensor.round(srng.uniform(size=(3,), ndim=1))


class DropoutLayer(object):
    def __init__(self, data, n_in, srng, p, train_flag):
        """
        This implements the dropout layer in neural network.

        :type data: theano.tensor.dmatrix
        :param data: a symbolic tensor of shape (n_examples, n_in)

        :type srng: theano.sandbox.rng_mrg.MRG_RandomStreams
        :param srng: symbolic random number generator

        :type n_in: int
        :param n_in: dimensionality of input

        :type p: float
        :param p: the probability of dropping out

        :type train_flag: bool
        :param train_flag: whether or not it's training
        """

        self.input = data

        self.in_shape = n_in

        self.params = []

        rand = T.round(srng.uniform(size=(n_in,), ndim=1))

        multiplier = 1.0 / p

        self.output = T.switch(train_flag, data * rand, data * multiplier)

        #self.out_shape = self.in_shape


class ConvLayer(object):
    def __init__(self, rng, data, image_shape, filter_shape,
                 activation=T.tanh, pad='valid', strides=(1, 1)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type data: theano.tensor.dtensor4
        :param data: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        """

        pad_list = ['valid', 'full', 'same']

        assert image_shape[1] == filter_shape[1]
        assert pad in pad_list
        self.input = data

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(4. / fan_in)
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
        if pad != 'same':
            conv_out = conv.conv2d(
                input=data,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode=pad,
                subsample=strides
            )
        else:
            x_temp = (filter_shape[2] - 1) / 2
            y_temp = (filter_shape[3] - 1) / 2
            conv_out_temp = conv.conv2d(
                input=data,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode='full',
                subsample=strides
            )
            conv_out = conv_out_temp[:, :, x_temp:x_temp+image_shape[2]-1, y_temp:y_temp+image_shape[3]-1]

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # self.out_shape = conv_out.shape.eval()


class PoolLayer(object):
    def __init__(self, data,
                 poolsize=(2, 2), stride=(2, 2), pad=(0, 0)):
        """
        Pooling Layer. Seems only max pooling works.

        :type data: theano.tensor.dtensor4
        :param data: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type stride: tuple or list of length 2
        :param stride: the distance between two adjacent pooling operation in pixels (#rows, #cols)

        :type pad: tuple or list of length 2
        :param pad: # of extra pixels added on margins
        """

        self.input = data

        pooled_out = downsample.max_pool_2d(
            input=data,
            ds=poolsize,
            ignore_border=True,
            st=stride,
            padding=pad
        )

        self.output = pooled_out

        self.params = []

        #self.out_shape = pooled_out.shape.eval()


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    layers = []

    srng = RandomStreams(25252)

    train_flag = T.bscalar('train_flag')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

    layer0 = ConvLayer(
        rng,
        data=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5)
    )

    layers.append(layer0)

    layer1 = PoolLayer(
        data=layer0.output
    )
    layers.append(layer1)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer2 = ConvLayer(
        rng,
        data=layer1.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5)
    )
    layers.append(layer2)

    layer3 = PoolLayer(
        data=layer2.output
    )
    layers.append(layer3)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layerd_input = layer3.output.flatten(2)

    layerd = DropoutLayer(
        data=layerd_input,
        n_in=nkerns[1] * 4 * 4,
        srng=srng,
        p=.5,
        train_flag=train_flag
    )

    # construct a fully-connected sigmoidal layer
    layer4 = FCLayer(
        rng,
        data=layerd.output,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=relu
    )
    layers.append(layer4)

    # classify the values of the fully-connected sigmoidal layer
    layer5 = LogisticRegression(input=layer4.output, n_in=500, n_out=10)
    layers.append(layer5)

    # the cost we minimize during training is the NLL of the model
    cost = layer5.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            train_flag: numpy.cast['int8'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            train_flag: numpy.cast['int8'](0)
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer5.params + layerd.params + layer4.params + layer3.params + \
             layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    delta_before = []

    for param_i in params:
        delta_before_i = theano.shared(
            value=numpy.zeros(param_i.get_value().shape, dtype=theano.config.floatX),
            borrow=True
        )
        delta_before.append(delta_before_i)

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    updates = []

    for param_i, grad_i, delta_before_i in zip(params, grads, delta_before):
        delta_i = momentum * delta_before_i - weight_decay * learning_rate * param_i - learning_rate * grad_i
        updates.append((delta_before_i, delta_i))
        updates.append((param_i, param_i + delta_i))

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
            train_flag: numpy.cast['int8'](1)
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    if gpu_usage is True:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print "Total memory:", info.total
        print "Free memory:", info.free

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)






# Main:
# srng = RandomStreams(seed=seed)
