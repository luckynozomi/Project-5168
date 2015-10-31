"""
    This file implements the function to load cifar10 data file
"""

import os
import numpy

import theano
import theano.tensor as T


def load_data(batch_numbers):
    """ This function loads the cifar-10 dataset

    :type batch_numbers: char
    :param batch_numbers: the number of batches to be loaded

    """

    import cPickle
    import tarfile

    input_file = 'data/cifar-10-python.tar.gz'

    tar_file = tarfile.open(input_file, 'r:gz')

    batch_numbers = numpy.asarray(batch_numbers, dtype='int32')

    train_batches = []

    for batch in batch_numbers:
        if batch == 6:
            fo = tar_file.extractfile(os.path.join('cifar-10-batches-py', 'test_batch'))
        else:
            fo = tar_file.extractfile(os.path.join('cifar-10-batches-py', 'data_batch_%d' % batch))
        array = cPickle.load(fo)
        train_batches.append(array)
        fo.close()

    x = numpy.concatenate([batch['data'] for batch in train_batches])

    x = (x / 255. - .5) * 2

    y = numpy.concatenate(
        [numpy.array(batch['labels'], dtype=numpy.uint8) for batch in train_batches]
    )

    tar_file.close()

    print x.shape

    shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(y, dtype=theano.config.floatX),
                             borrow=True)

    return shared_x, T.cast(shared_y, 'int32')