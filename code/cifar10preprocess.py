"""
    Left Blank
"""

import cPickle
import tarfile
from matplotlib import pyplot
import numpy

input_file = '../data/cifar-10-python.tar.gz'

tar_file = tarfile.open(input_file, 'r:gz')

fo = tar_file.extractfile('cifar-10-batches-py/data_batch_5')
array = cPickle.load(fo)
fo.close()

x = array['data']
y = numpy.asarray(array['labels'], dtype=numpy.uint8)

xtest = x.reshape(10000, 3, 32, 32)

xtest = xtest[:, :, :, ::-1]

xx = xtest.reshape(10000, 3072)

data = numpy.concatenate((x, xx))

labels = numpy.concatenate((y, y))

fo = open('../data/data_batch_5', 'wb')

cPickle.dump((data, labels), fo, protocol=cPickle.HIGHEST_PROTOCOL)

fo.close()