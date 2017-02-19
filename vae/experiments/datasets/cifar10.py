import tarfile
import cPickle as pickle

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets, maybe_download

from ..dataset import DatasetBase, DataWrapper

CIFAR10_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class Dataset(DatasetBase):
    name = 'cifar10'

    # Attributes.
    width = 32
    height = 32
    channels = 3
    class_count = 10

    def get_data(self):
        if not hasattr(self, '_data'):
            self._data = self._load_data()
        return self._data

    def _unpickle(self, fileobj):
        data = pickle.load(fileobj)
        return {
            'x': data['data'].astype(np.float32).reshape(
                    [-1, self.channels, self.width, self.height]
                ).transpose(0, 3, 2, 1).reshape(
                    [-1, self.width * self.height * self.channels]
                ) / 255.,
            'y': np.asarray(data['labels'], dtype=np.uint8),
        }

    def _load_data(self):
        work_directory = '.cifar10_data'
        images_path = maybe_download('cifar-10-python.tar.gz', work_directory, CIFAR10_URL)

        with tarfile.open(images_path, 'r') as images_tar:
            train_data = [
                self._unpickle(images_tar.extractfile('cifar-10-batches-py/data_batch_{}'.format(batch)))
                for batch in range(1, 6)
            ]

            train_data = {
                'x': np.concatenate([d['x'] for d in train_data], axis=0),
                'y': np.concatenate([d['y'] for d in train_data], axis=0),
            }

            test_data = self._unpickle(images_tar.extractfile('cifar-10-batches-py/test_batch'))

        train = DataWrapper(train_data['x'], train_data['y'])
        test = DataWrapper(test_data['x'], test_data['y'])
        validation = DataWrapper(np.asarray([]), np.asarray([]))

        return Datasets(train=train, test=test, validation=validation)
