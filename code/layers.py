"""
    This file implements the neural network layers needed for the project.
"""


import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse


def relu(x):
    return 0.5 * (x + abs(x))

gpu_usage = True

try:
    from pynvml import *
except ImportError:
    print 'package nvidia-ml-py doesn''t exist, cannot show GPU usage data'
    gpu_usage = False


class ElasticLayer(object):
    def __init__(self, srng, data, image_shape, train_flag):
        """

        :param srng:

        :type data: theano.tensor.dtensor4
        :param data: symbolic image tensor, of shape image_shape

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type train_flag: symbolic boolean
        :param train_flag: whether or not it's training


        """

        if train_flag is False:
            self.output = data
            return

        p = srng.uniform(size=(1,), ndim=1)[0]

        temp = ifelse(p > .5, data, data[:, :, :, ::-1])

        pad_x = 2
        pad_y = 2

        temp_padded = theano.shared(
            numpy.zeros(
                shape=(image_shape[0], image_shape[1], image_shape[2] + pad_y * 2, image_shape[3] + pad_x * 2),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # TODO: simplify this, only need 1 number
        rand_x = T.iround(srng.uniform(size=(1,), low=-0.49, high=4.49, ndim=1))[0]
        rand_y = T.iround(srng.uniform(size=(1,), low=-0.49, high=4.49, ndim=1))[0]

        temp_padded = T.set_subtensor(
            temp_padded[:, :, rand_y:rand_y+image_shape[2], rand_x:rand_x+image_shape[3]],
            temp
        )

        self.output = temp_padded[:, :, pad_y:pad_y+image_shape[2], pad_x:pad_x+image_shape[3]]

        self.params = []




class FCLayer(object):
    def __init__(self, rng, data, n_in, n_out,
                 W=None, b=None, activation=T.nnet.sigmoid):
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

        :type W: theano.config.floatX
        :param W: the initial weights, of shape (n_in, n_out)

        :type b: theano.config.floatX
        :param b: the initial bias, of shape(n_out,)
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

        :type train_flag: symbolic boolean
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
                 activation=None, pad='valid', strides=(1, 1)):
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

        :type pad:

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
            conv_out = conv_out_temp[:, :, x_temp:x_temp+image_shape[2], y_temp:y_temp+image_shape[3]]

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # store parameters of this layer
        self.params = [self.W, self.b]

        # self.out_shape = conv_out.shape.eval()


class PoolLayer(object):
    def __init__(self, data,
                 poolsize=(2, 2), stride=(2, 2), pad=(0, 0), activation=None):
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

        # TODO: stride = poolsize by default


        self.input = data

        pooled_out = downsample.max_pool_2d(
            input=data,
            ds=poolsize,
            ignore_border=True,
            st=stride,
            padding=pad
        )

        self.output = (
            pooled_out if activation is None
            else activation(pooled_out)
        )

        self.output = pooled_out

        self.params = []

        #self.out_shape = pooled_out.shape.eval()


