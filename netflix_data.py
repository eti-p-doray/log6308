import datetime
import logging
import numpy
import os

from tensorflow.contrib.learn.python.learn.datasets import base

import utility

MOVIE_COUNT = 17771
USER_COUNT = 2649430
NUM_RATING = 100480507


class DataSet(object):
    @staticmethod
    def fromfile(filename):
        data = numpy.load(filename)
        return DataSet(data[0,:], data[1,:], data[2,:], data[3,:])

    def __init__(self, movies, users, dates, ratings):
        self.movies_ = numpy.asarray(movies)
        self.users_ = numpy.asarray(users)
        self.dates_ = numpy.asarray(dates)
        self.ratings_ = numpy.asarray(ratings)

    @property
    def num_examples(self):
        return len(self.ratings_)

    @property
    def users(self):
        return self.users_

    @property
    def movies(self):
        return self.movies_

    @property
    def dates(self):
        return self.dates_

    @property
    def ratings(self):
        return self.ratings_

    def iter_batch(self, batch_size, shuffle=True):
        perm = numpy.arange(self.num_examples)
        numpy.random.shuffle(perm)
        for i in range(0, len(perm), batch_size):
            indices = perm[i:min(i+batch_size, len(perm))]
            movies = self.movies_[indices]
            users = self.users_[indices]
            dates = self.dates_[indices]
            ratings = self.ratings_[indices]
            yield DataSet(movies, users, dates, ratings)

    def save(self, filename):
        numpy.save(filename, numpy.vstack((self.movies_, self.users_, self.dates_, self.ratings_)))


def read_data_sets(train_dir) -> DataSet:
    epoch = datetime.datetime.utcfromtimestamp(0)

    ratings = numpy.zeros(NUM_RATING, dtype=numpy.int8)
    movieids = numpy.zeros(NUM_RATING, dtype=numpy.int16)
    userids = numpy.zeros(NUM_RATING, dtype=numpy.int32)
    dates = numpy.zeros(NUM_RATING, dtype=numpy.int32)

    k = 0
    for filename in os.listdir(train_dir):
        if filename.endswith(".txt"):
            print(filename)
            with open(os.path.join(train_dir, filename), 'r') as csvfile:
                data = csvfile.readlines()
                movieid = int(data.pop(0)[:-2])
                numratings = len(data)
                movieids[k:k + numratings] = movieid

                for j in range(numratings):
                    userid, stars, date = data[j][:-1].split(',')
                    userids[k] = int(userid)
                    ratings[k] = int(stars)
                    dates[k] = (datetime.datetime(int(date[0:4]), int(date[5:7]), int(date[8:10])) -
                                epoch).total_seconds()
                    k = k + 1

    return DataSet(movieids, userids, dates, ratings)

#nf_prize = read_data_sets("nf_prize_dataset/training_set")
#nf_prize.save("nf_prize_dataset/nf_prize.npy")
#print(nf_prize.num_examples)

#nf_prize = DataSet.fromfile("nf_prize_dataset/nf_prize.npy")
#print(nf_prize.num_examples)
#for x in nf_prize.iter_batch(100):
#    print(x.movies_)
#    break