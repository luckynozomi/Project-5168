"""
    This code puts all the training data together into one file
"""

import cPickle
import tarfile
from matplotlib import pyplot
import numpy

input_file = '../data/cifar-10-python.tar.gz'

tar_file = tarfile.open(input_file, 'r:gz')

fo1 = tar_file.extractfile('cifar-10-batches-py/data_batch_1')
array1 = cPickle.load(fo1)
fo1.close()

fo2 = tar_file.extractfile('cifar-10-batches-py/data_batch_2')
array2 = cPickle.load(fo2)
fo2.close()

fo3 = tar_file.extractfile('cifar-10-batches-py/data_batch_3')
array3 = cPickle.load(fo3)
fo3.close()

fo4 = tar_file.extractfile('cifar-10-batches-py/data_batch_4')
array4 = cPickle.load(fo4)
fo4.close()

x1 = array1['data']
x2 = array2['data']
x3 = array3['data']
x4 = array4['data']
y1 = numpy.asarray(array1['labels'], dtype=numpy.uint8)
y2 = numpy.asarray(array2['labels'], dtype=numpy.uint8)
y3 = numpy.asarray(array3['labels'], dtype=numpy.uint8)
y4 = numpy.asarray(array4['labels'], dtype=numpy.uint8)


data = numpy.concatenate((x1, x2, x3, x4))

labels = numpy.concatenate((y1, y2, y3, y4))

fo = open('../data/data_batch_1', 'wb')

cPickle.dump((data, labels), fo, protocol=cPickle.HIGHEST_PROTOCOL)

fo.close()