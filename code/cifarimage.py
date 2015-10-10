"""
    This python file that shows the image in the cifar10 dataset.

    The cifar-10-python.tar.gz should be stored at ../data/
"""

import cPickle
import tarfile
from matplotlib import pyplot


class CIFARImage(object):
    """ An image form the CIFAR-10 Dataset"""

    def __init__(self, batch=1, number=0):
        """ Get the image in batch=batch, number=number"""
        input_file = '../data/cifar-10-python2.tar.gz'
        tar_file = tarfile.open(input_file, 'r:gz')

        if batch == 6:

            # Test batch in this case
            fo = tar_file.extractfile('test_batch')
        else:

            # Labeled data batch in this case
            fo = tar_file.extractfile('data_batch_%d' % batch)

        array = cPickle.load(fo)
        fo.close()

        #
        self.batch = array[0].reshape(array[0].shape[0], 3, 32, 32)

        self.data = self.batch[number].swapaxes(1, 2).transpose()

        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.label = categories[array[1][number]]

    def show(self):
        """ Show the image"""

        pyplot.imshow(self.data / 255.0)
        pyplot.show()
