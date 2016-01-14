"""
    This file implements the training process of the model
"""

# TODO: leave the definition of the model as a seperate file

import os
import sys
import timeit

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import *
from load_new import load_data
import cPickle as pickle


def evaluate_cifar10(n_epochs=800, nkerns=[128, 128, 128], batch_size=100,
                    n_params=8, load_weight=None, save_weight="fresh_model.pickle"):
    """ CNN on CIFAR-10 dataset

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_size: size of a mini batch

    :type n_params: int
    :param n_params: number of parameters in the model

    :type load_weight: string
    :param load_weight: the name of the file from which to load the weights

    :type save_weight: string
    :param save_weight: the name of the file in which the weights are saved
    """

    rng = numpy.random.RandomState(23455)

    train_set_x, train_set_y = load_data([1,2,3,4])
    valid_set_x, valid_set_y = load_data([5])
    test_set_x, test_set_y = load_data([6])

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

    params = []

    srng = RandomStreams(25252)

    train_flag = T.bscalar('train_flag')

    if load_weight is None:
        load_weight = [None] * n_params
    else:
        f = open(os.path.join("model", load_weight), "rb")
        load_weight = pickle.load(f)
        f.close()

    param_count = 0

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32) to a 4D tensor

    layer0_input = x.reshape((batch_size, 3, 32, 32))

    layerx = ElasticLayer(
        srng=srng,
        data=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        train_flag=train_flag
    )
    params += layerx.params
    param_count += 0

    layer0 = ConvLayer(
        rng,
        data=layerx.output,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        pad='same',
        activation=relu,
        init_W=load_weight[param_count],
        init_b=load_weight[param_count+1]
    )
    params += layer0.params
    param_count += 2

    layer1 = PoolLayer(
        data=layer0.output,
        stride=(2, 2),
        poolsize=(2, 2)
    )
    params += layer1.params
    param_count += 0

    layer2 = ConvLayer(
        rng,
        data=layer1.output,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        pad='same',
        activation=relu,
        init_W=load_weight[param_count],
        init_b=load_weight[param_count+1]
    )
    params += layer2.params
    param_count += 2

    layer3 = PoolLayer(
        data=layer2.output,
        poolsize=(2, 2),
        stride=(2, 2)
    )
    params += layer3.params
    param_count += 0

    layer4 = ConvLayer(
        rng,
        data=layer3.output,
        image_shape=(batch_size, nkerns[1], 8, 8),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        pad='same',
        activation=relu,
        init_W=load_weight[param_count],
        init_b=load_weight[param_count+1]
    )
    params += layer4.params
    param_count += 2

    layer5 = PoolLayer(
        data=layer4.output
    )
    params += layer5.params
    param_count += 0

    layer6_input = layer5.output.flatten(2)

    layer5d = DropoutLayer(
        data=layer6_input,
        n_in=nkerns[2] * 4 * 4,
        srng=srng,
        p=.5,
        train_flag=train_flag
    )
    params += layer5d.params
    param_count += 0

    layer6 = LogisticRegression(
        input=layer5d.output,
        n_in=nkerns[2] * 4 * 4,
        n_out=10,
        init_W=load_weight[param_count],
        init_b=load_weight[param_count+1]
    )
    params += layer6.params
    param_count += 2

    # the cost we minimize during training is the NLL of the model
    cost = layer6.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            train_flag: numpy.cast['int8'](0)
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size],
            train_flag: numpy.cast['int8'](0)
        }
    )

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
    train_time = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    if 'gpu' in theano.config.device:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print "Total memory:", info.total
        print "Free memory:", info.free

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_start_time = timeit.default_timer()

        if epoch == 2 or epoch == 11:
            print ('     average trainning time per epoch = %.2fs' % (train_time / (epoch - 1)))

        if epoch % 50 == 0:
            learning_rate *= .5

        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                train_time += timeit.default_timer() - train_start_time

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
                    print(('epoch %3i, minibatch %i/%i, validation error %.2f%%, test error of '
                           'best model %.2f%%') %
                          (epoch, minibatch_index + 1, n_train_batches, 100. * this_validation_loss,
                           test_score * 100.))
                    if epoch >= min(60, n_epochs):
                        print('     saving model...')
                        param_values = [numpy.asarray(param.eval()) for param in params]  # otherwise CudaNdarray object
                        f = open(os.path.join("model", save_weight), "wb")
                        pickle.dump(param_values, f)
                        f.close()
                else:
                    print('epoch %3i, minibatch %i/%i, validation error %.2f%%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           100. * this_validation_loss))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (100. * best_validation_loss, best_iter + 1, 100. * test_score))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    evaluate_cifar10()
