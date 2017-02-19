from tensorflow.contrib.learn.python.learn.datasets import mnist

from ..dataset import DatasetBase


class Dataset(DatasetBase):
    name = 'mnist'

    # Attributes.
    width = 28
    height = 28
    channels = 1
    class_count = 10

    def get_data(self):
        if not hasattr(self, '_data'):
            self._data = mnist.read_data_sets('.mnist_data')
        return self._data
