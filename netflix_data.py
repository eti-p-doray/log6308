import collections
import csv
import datetime
import itertools
import logging
import numpy
import os

import utility

MOVIE_COUNT = 17770
USER_COUNT = 480189
MAX_USER_ID = 2649429
NUM_RATING = 100480507

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class DataSet(object):
    @staticmethod
    def fromfile(filename):
        data = numpy.load(filename)
        return DataSet(data[0,:], data[1,:], data[2,:], data[3,:])

    def __init__(self, movie_ids, user_ids, dates, ratings):
        self.movie_ids_ = numpy.asarray(movie_ids)
        self.user_ids_ = numpy.asarray(user_ids)
        self.dates_ = numpy.asarray(dates)
        self.ratings_ = numpy.asarray(ratings)

    @property
    def num_examples(self):
        return len(self.ratings_)

    @property
    def movie_ids(self):
        return self.movie_ids_

    @property
    def user_ids(self):
        return self.user_ids_

    @property
    def dates(self):
        return self.dates_

    @property
    def ratings(self):
        return self.ratings_

    def iter_batch(self, batch_size, shuffle=True):
        while True:
            indices = numpy.random.choice(self.num_examples, batch_size)
            movie_ids = self.movie_ids_[indices]
            user_ids = self.user_ids_[indices]
            dates = self.dates_[indices]
            ratings = self.ratings_[indices]
            yield DataSet(movie_ids, user_ids, dates, ratings)

    def split(self, validation_size, test_size):
        assert validation_size + test_size <= self.num_examples
        train_size = self.num_examples - validation_size - test_size
        perm = numpy.arange(self.num_examples)
        numpy.random.shuffle(perm)

        def genenerate_datasets():
            for i,j in utility.pairwise(itertools.accumulate([0, train_size, validation_size, test_size, 1.0])):
                indices = perm[i:j]
                movie_ids = self.movie_ids_[indices]
                user_ids = self.user_ids_[indices]
                dates = self.dates_[indices]
                ratings = self.ratings_[indices]
                yield DataSet(movie_ids, user_ids, dates, ratings)

        datasets = genenerate_datasets()
        train = next(datasets)
        validation = next(datasets)
        test = next(datasets)
        return Datasets(train=train, validation=validation, test=test)

    def save(self, filename):
        numpy.save(filename, numpy.vstack((self.movie_ids_, self.user_ids_, self.dates_, self.ratings_)))


def read_data_sets(train_dir):
    epoch = datetime.datetime.utcfromtimestamp(0)

    ratings = numpy.zeros(NUM_RATING, dtype=numpy.int8)
    movieids = numpy.zeros(NUM_RATING, dtype=numpy.int16)
    userids = numpy.zeros(NUM_RATING, dtype=numpy.int32)
    dates = numpy.zeros(NUM_RATING, dtype=numpy.int32)

    k = 0
    for filename in os.listdir(train_dir):
        if filename.endswith(".txt"):
            logging.info(filename)
            with open(os.path.join(train_dir, filename), 'r') as csvfile:
                data = csvfile.readlines()
                movieid = int(data.pop(0)[:-2])
                numratings = len(data)
                movieids[k:k + numratings] = movieid-1

                for j in range(numratings):
                    userid, stars, date = data[j][:-1].split(',')
                    userids[k] = int(userid)
                    ratings[k] = int(stars)
                    dates[k] = (datetime.datetime(int(date[0:4]), int(date[5:7]), int(date[8:10])) -
                                epoch).total_seconds()
                    k = k + 1

    users, userids = numpy.unique(userids, return_inverse=True)
    return DataSet(movieids, userids, dates, ratings), users


def export_movie_titles(input, output):
    with open(input, 'r', encoding='latin-1') as inputfile, open(output, 'w+') as outputfile:
        reader = csv.reader(inputfile, delimiter=',')
        writer = csv.writer(outputfile, delimiter='\t')
        writer.writerow(['Id', 'Year', 'Title'])
        for row in reader:
            writer.writerow(row)

#export_movie_titles('nf_prize_dataset/movie_titles.txt', 'nf_prize_dataset/movie_titles.tsv')

#logging.basicConfig(level=logging.DEBUG)
#nf_prize, users = read_data_sets("nf_prize_dataset/training_set")
#nf_prize.save("nf_prize_dataset/nf_prize.npy")
#numpy.save("nf_prize_dataset/users.npy", users)
#print(nf_prize.num_examples)
