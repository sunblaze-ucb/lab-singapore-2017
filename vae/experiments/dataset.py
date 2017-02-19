import numpy as np


class DatasetBase(object):
    """Abstract dataset."""
    name = None

    # Dataset attributes.
    width = None
    height = None
    channels = None
    class_count = None

    def __init__(self, options):
        assert self.name, 'Each dataset must define a name attribute.'

        self.options = options

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        pass

    def get_data(self):
        """Load and return data from the dataset."""
        raise NotImplementedError


class DataWrapper(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0]

        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
