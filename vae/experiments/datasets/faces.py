from __future__ import print_function

import os
import zipfile

import tqdm
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets, maybe_download
from scipy.misc import imread, imresize

from ..dataset import DatasetBase, DataWrapper

FACES_IMAGES_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1'
FACES_LABELS_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1'


class Dataset(DatasetBase):
    name = 'faces'

    # Attributes.
    width = 64
    height = 64
    channels = 3
    class_count = 10

    @classmethod
    def add_options(cls, parser):
        """Augment program arguments."""
        super(Dataset, cls).add_options(parser)

        parser.add_argument('--dataset-random-labels', action='store_true')

    def get_data(self):
        if not hasattr(self, '_data'):
            self._data = self._load_data()
        return self._data

    def _load_data(self):
        work_directory = '.faces_data'
        images_path = maybe_download('img_align_celeba.zip', work_directory, FACES_IMAGES_URL)
        labels_path = maybe_download('list_attr_celeba.txt', work_directory, FACES_LABELS_URL)

        # Load labels.
        image_count = 0
        attributes = []
        attributes_classes = ['Male', 'Young', 'Smiling', 'Attractive']
        label_map = {}
        with open(labels_path, 'r') as labels_file:
            for line_no, line in enumerate(labels_file):
                if line_no == 0:
                    # Parse example count.
                    image_count = int(line)
                    continue
                elif line_no == 1:
                    # Parse header.
                    attributes = line.split()
                    continue

                # Parse line and determine class label.
                line = line.split()
                if self.options.dataset_random_labels:
                    label = (line_no - 2) % self.class_count
                else:
                    label = 0
                    for index, attribute in enumerate(attributes_classes):
                        value = int(line[attributes.index(attribute) + 1])
                        if value == 1:
                            label += 2**index

                    if label > 9:
                        continue

                label_map[line[0]] = label

        # Load images.
        images = np.zeros([image_count, self.width * self.height * self.channels], dtype=np.float32)
        labels = np.zeros([image_count], dtype=np.int8)
        with zipfile.ZipFile(images_path, 'r') as images_zip:
            image_infos = images_zip.infolist()
            index = 0
            progress = tqdm.tqdm(total=image_count, leave=False)
            for image_info in image_infos:
                if not image_info.filename.endswith('.jpg'):
                    continue

                label = label_map.get(os.path.basename(image_info.filename), None)
                if label is None:
                    continue

                with images_zip.open(image_info) as image_file:
                    image = imread(image_file).astype(np.float32)

                    # Resize image to target dimensions.
                    h, w = image.shape[:2]
                    image = imresize(image, [int((float(h) / w) * self.width), self.width])
                    j = int(round((image.shape[0] - self.height) / 2.))
                    image = image[j:j + self.height, :, :]
                    image = image / 255.

                    images[index, :] = image.flatten()
                    labels[index] = label
                    index += 1
                    progress.update()

            image_count = index + 1
            images = images[:image_count]
            labels = labels[:image_count]
            progress.close()

        print('Image count:', index)
        print('Values: min={} max={} mean={}'.format(np.min(images), np.max(images), np.mean(images)))

        print('Class distribution:')
        for label, count in zip(*np.unique(labels, return_counts=True)):
            print('  {}: {}'.format(label, count))

        train = DataWrapper(images, labels)
        test = DataWrapper(images[:1000], labels[:1000])
        validation = DataWrapper(np.asarray([]), np.asarray([]))

        return Datasets(train=train, test=test, validation=validation)
