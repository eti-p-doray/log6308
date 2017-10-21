import csv
import datetime
import logging
import numpy

from tensorflow.contrib.learn.python.learn.datasets import base


class DataSet(object):
    def __init__(self, users, movies, dates, ratings):
        pass

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        pass

def read_data_sets(train_dir,
                   validation_size=5000,
                   seed=None):
    #train = DataSet(train_images, train_labels, **options)
    #validation = DataSet(validation_images, validation_labels, **options)
    #test = DataSet(test_images, test_labels, **options)
    #return base.Datasets(train=train, validation=validation, test=test)
    pass