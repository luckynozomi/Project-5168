"""
    This file implements the function to load cifar10 data file
"""


import numpy

import theano
import theano.tensor as T

def load_data(batch_number):
    """ This function loads the cifar-10 dataset

    :type batch_number: char
    :param batch_number: the number of batches to be loaded

    """

    import cPickle
    import tarfile

    input_file = 'data/cifar10.tar.gz'

    tar_file = tarfile.open(input_file, 'r:gz')

    if batch_number == 6:
        fo = tar_file.extractfile('test_batch')
    else:
        fo = tar_file.extractfile('data_batch_%d' % batch_number)
    array = cPickle.load(fo)
    fo.close()

    x = (array[0] / 255. - .5) * 2
    y = numpy.array(array[1], dtype=numpy.uint8)

    print x.shape

    shared_x = theano.shared(numpy.asarray(x, dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(y, dtype=theano.config.floatX),
                             borrow=True)

    return shared_x, T.cast(shared_y, 'int32')
