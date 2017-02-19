import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets, maybe_download
from scipy.io import loadmat

from ..dataset import DatasetBase, DataWrapper


class Dataset(DatasetBase):
    name = 'svhn'

    # Attributes.
    width = 32
    height = 32
    channels = 3
    class_count = 10

    def get_data(self):
        if not hasattr(self, '_data'):
            self._data = self._load_data()
        return self._data

    def _transform(self, dataset):
        x = dataset['X'].astype(np.float32).transpose(3, 0, 1, 2).reshape(
            [-1, self.width * self.height * self.channels]
        ) / 255.
        y = dataset['y'].reshape([-1])
        # Replace label 10 with label 0 as it represents the digit "0".
        y[y == 10] = 0
        return x, y

    def _load_data(self):
        work_directory = '.svhn_data'
        train_path = maybe_download('train_32x32.mat', work_directory,
                                    'http://ufldl.stanford.edu/housenumbers/train_32x32.mat')

        test_path = maybe_download('test_32x32.mat', work_directory,
                                   'http://ufldl.stanford.edu/housenumbers/test_32x32.mat')

        train = DataWrapper(*self._transform(loadmat(train_path)))
        test = DataWrapper(*self._transform(loadmat(test_path)))
        validation = DataWrapper(np.asarray([]), np.asarray([]))

        return Datasets(train=train, test=test, validation=validation)
