"""
    This file implements the neural network layers needed for the project.
"""

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
import warnings

try:
    from theano.tensor.nnet import relu
except ImportError:
    def relu(x):
        return 0.5 * (x + abs(x))

if 'gpu' in theano.config.device:
    try:
        from pynvml import *
    except ImportError:
        print 'package nvidia-ml-py doesn''t exist, cannot show GPU usage data'


class ElasticLayer(object):
    def __init__(self, srng, data, image_shape, train_flag):
        """
        A layer that applies random transformation to the input image

        :type srng: theano.sandbox.rng_mrg.MRG_RandomStreams
        :param srng: symbolic random number generator

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
                 init_W=None, init_b=None, activation=T.nnet.sigmoid):
        # TODO: Change W and b to init_W and init_b for consistency.
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

        :type init_W: theano.config.floatX
        :param init_W: the initial weights, of shape (n_in, n_out)

        :type init_b: theano.config.floatX
        :param init_b: the initial bias, of shape(n_out,)
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
        if init_W is None:
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

            init_W = theano.shared(value=W_values, name='W', borrow=True)

        if init_b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            init_b = theano.shared(value=b_values, name='init_b', borrow=True)

        self.W = init_W
        self.b = init_b

        # parameters of the model
        self.params = [self.W, self.b]

        lin_output = T.dot(self.input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.n_out = n_out


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


class ConvLayer(object):
    def __init__(self, rng, data, image_shape, filter_shape,
                 activation=None, init_W=None, init_b=None, pad='valid', stride=(1, 1)):
        """
        A convolution layer

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

        :type init_W: theano.config.floatX
        :param init_W: the initial weights, of shape (n_in, n_out)

        :type init_b: theano.config.floatX
        :param init_b: the initial bias, of shape(n_out,)

        :type pad: string
        :param pad: the choice of padding:
            'valid': no padding
            'full': full padding
            'valid': certain amount of padding such that the output has the same image_shape as input

        :type stride: tuple or list of length 2
        :param stride: the distance of two adjacent convolution op in pixels (#rows, #cols)

        """

        pad_list = ['valid', 'full', 'same']
        assert image_shape[1] == filter_shape[1]
        assert pad in pad_list

        self.input = data

        if init_W is None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # initialize weights with random weights
            W_bound = numpy.sqrt(4. / fan_in)

            init_W = numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(init_W, borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        if init_b is None:
            init_b = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(init_b, borrow=True)

        # convolve input feature maps with filters
        if pad != 'same':
            conv_out = conv.conv2d(
                input=data,
                filters=self.W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode=pad,
                subsample=stride
            )
        else:
            x_temp = (filter_shape[2] - 1) / 2
            y_temp = (filter_shape[3] - 1) / 2

            # The following implements convolution with same padding.
            # Parameters in cuDNN library is different, and it requires the filter shape to be even.

            if 'gpu' in theano.config.device and filter_shape[2] % 2 == 1 and filter_shape[3] % 2 == 1:
                conv_out = theano.sandbox.cuda.dnn.dnn_conv(
                    img=data,
                    kerns=self.W,
                    border_mode=((filter_shape[2] - 1) / 2, (filter_shape[3] - 1) / 2)
                )
            else:
                if 'gpu' in theano.config.device:
                    warnings.warn("The filter shape must be even to use cuDNN,"
                                  " implementing a slower veresion of conv2d")
                conv_out_temp = conv.conv2d(
                    input=data,
                    filters=self.W,
                    filter_shape=filter_shape,
                    image_shape=image_shape,
                    border_mode='full',
                    subsample=stride
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


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out,
                 init_W=None, init_b=None):
        """ Initialize the parameters of the logistic regression

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
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if init_W is None:
            init_W = numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            )
        self.W = theano.shared(init_W, name='W', borrow=True)

        # initialize the biases b as a vector of n_out 0s
        if init_b is None:
            init_b = numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )
        self.b = theano.shared(value=init_b, name='b', borrow=True)

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
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

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
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
