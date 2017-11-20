import collections
import csv
import datetime
import itertools
import logging
import math
import numpy
import os
import sys

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
        return DataSet(data['movie_ids'], data['user_ids'], data['dates'], data['ratings'], data['user_map'])

    def __init__(self, movie_ids, user_ids, dates, ratings, user_map):
        self.movie_ids_ = numpy.asarray(movie_ids)
        self.user_ids_ = numpy.asarray(user_ids)
        self.dates_ = numpy.asarray(dates)
        self.ratings_ = numpy.asarray(ratings)
        self.user_map_ = user_map

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

    @property
    def user_map(self):
        return self.user_map_

    def iter_batch(self, batch_size, shuffle=True):
        while True:
            indices = numpy.random.choice(self.num_examples, batch_size)
            movie_ids = self.movie_ids_[indices]
            user_ids = self.user_ids_[indices]
            dates = self.dates_[indices]
            ratings = self.ratings_[indices]
            yield DataSet(movie_ids, user_ids, dates, ratings, self.user_map)

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
                yield DataSet(movie_ids, user_ids, dates, ratings, self.user_map)

        datasets = genenerate_datasets()
        train = next(datasets)
        validation = next(datasets)
        test = next(datasets)
        return Datasets(train=train, validation=validation, test=test)

    def save(self, filename):
        numpy.savez(filename,
                    movie_ids = self.movie_ids_,
                    user_ids = self.user_ids_,
                    dates = self.dates_,
                    ratings = self.ratings_,
                    user_map = self.user_map_)

    def find_dense_subset(self, n_users, n_movies):
        user_ratings = numpy.zeros(USER_COUNT)
        movie_ratings = numpy.zeros(MOVIE_COUNT)
        for user, movie in zip(self.user_ids_, self.movie_ids_):
            user_ratings[user] += 1
            movie_ratings[movie] += 1
        user_subset = set(numpy.argpartition(user_ratings, -n_users)[-n_users:])
        movie_subset = set(numpy.argpartition(movie_ratings, -n_movies)[-n_movies:])

        n_couples = 0
        for user, movie in zip(self.user_ids_, self.movie_ids_):
            if user in user_subset and movie in movie_subset:
                n_couples += 1

        ratings = numpy.zeros(n_couples, dtype=numpy.int8)
        movie_ids = numpy.zeros(n_couples, dtype=numpy.int16)
        user_ids = numpy.zeros(n_couples, dtype=numpy.int32)
        dates = numpy.zeros(n_couples, dtype=numpy.int32)
        j = 0
        for i in range(self.num_examples):
            if self.user_ids_[i] in user_subset and self.movie_ids[i] in movie_subset:
                ratings[j] = self.ratings_[i]
                movie_ids[j] = self.movie_ids_[i]
                user_ids[j] = self.user_ids_[i]
                dates[j] = self.dates_[i]
                j += 1

        return DataSet(movie_ids, user_ids, dates, ratings, self.user_map)

    def split_probe_subset(self, probe_file_path):
        couples = set()

        with open(probe_file_path, 'r') as probe_file:
            last_movie_id = None
            n_couples = 0
            for line in probe_file:
                if line: #not empty
                    idx = line.find(':')
                    if idx > 0: #movie id
                        last_movie_id = int(line[0:idx]) - 1
                    else: #user id
                        couples.add((last_movie_id, int(line)))
                        n_couples = n_couples + 1


        training_size = 0
        probe_size = 0
        for i, movie_id in enumerate(self.movie_ids_):
            user_id = self.user_ids[i]
            if (movie_id, self.user_map[user_id]) in couples:
                probe_size += 1
            else:
                training_size += 1

        test_ratings = numpy.zeros(probe_size, dtype=numpy.int8)
        test_movieids = numpy.zeros(probe_size, dtype=numpy.int16)
        test_userids = numpy.zeros(probe_size, dtype=numpy.int32)
        test_dates = numpy.zeros(probe_size, dtype=numpy.int32)

        training_ratings = numpy.zeros(training_size, dtype=numpy.int8)
        training_movieids = numpy.zeros(training_size, dtype=numpy.int16)
        training_userids = numpy.zeros(training_size, dtype=numpy.int32)
        training_dates = numpy.zeros(training_size, dtype=numpy.int32)

        j = 0
        k = 0
        for i, movie_id in enumerate(self.movie_ids_):
            user_id = self.user_ids_[i]
            if (movie_id, self.user_map[user_id]) in couples:
                test_userids[k] = user_id
                test_movieids[k] = movie_id
                test_ratings[k] = self.ratings_[i]
                test_dates[k] = self.dates_[i]
                k = k + 1
            else:
                training_userids[j] = user_id
                training_movieids[j] = movie_id
                training_ratings[j] = self.ratings_[i]
                training_dates[j] = self.dates_[i]
                j = j + 1

        ## Returns Test_set, Training_set, both complementary
        test_dataset = DataSet(test_movieids, test_userids, test_dates, test_ratings, self.user_map)
        training_dataset = DataSet(training_movieids, training_userids, training_dates, training_ratings, self.user_map)
        return test_dataset, training_dataset


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
    return DataSet(movieids, userids, dates, ratings, users)


def export_movie_titles(input, output):
    with open(input, 'r', encoding='latin-1') as inputfile, open(output, 'w+') as outputfile:
        reader = csv.reader(inputfile, delimiter=',')
        writer = csv.writer(outputfile, delimiter='\t')
        writer.writerow(['Id', 'Year', 'Title'])
        for row in reader:
            writer.writerow(row)

def preprocess_word_nt(movie_titles):
    nt = {}
    for movie_title in movie_titles:
        for word in movie_title:
            if word in nt:
                nt[word] = nt[word] + 1
            else:
                nt[word] = 1
    return nt

def preprocess_title_embeddings(model, movie_titles, nt, ncol):
    embeddings = numpy.zeros((len(movie_titles), ncol), dtype=numpy.float32)
    for i, movie_title in enumerate(movie_titles):
        embedding = numpy.zeros(ncol)
        length = 0
        for word in movie_title:
            if word in model.wv.vocab:
                word_vector = model[word][0:ncol]
                tfidf = math.log10(len(movie_titles) / nt[word])
                embedding = embedding + tfidf * word_vector
                length = length + 1
        if length > 0:
            embedding = embedding / length
        embeddings[i] = embedding

    return embeddings

def preprocess_get_titles(gen_movie_titles):
    if gen_movie_titles:
        title_array = numpy.zeros(MOVIE_COUNT, dtype=numpy.unicode_)
        i = 0
        with open('nf_prize_dataset/movie_titles.tsv') as f:
            first = True
            for line in f:
                if first: #ignore header line
                    first = False
                    continue
                words = line.split()
                words = words[2:] #remove movie id and year
                title = " ".join(words)
                title_array[i] = title
                i = i + 1

        numpy.save('nf_prize_dataset/nf_titles.npy', title_array)

        logging.debug('Titles generated and saved')

    else:

        title_array = numpy.load('nf_prize_dataset/nf_titles.npy')
        logging.debug('Title loaded from existing')
    return title_array

def main(argv):
    logging.basicConfig(level=logging.DEBUG)

    if not os.path.exists("nf_prize_dataset/nf_prize.npz"):
        nf_prize = read_data_sets("nf_prize_dataset/training_set")
        nf_prize.save("nf_prize_dataset/nf_prize.npz")
    else:
        nf_prize = DataSet.fromfile("nf_prize_dataset/nf_prize.npz")
    nf_test, nf_training = nf_prize.split_probe_subset("nf_prize_dataset/probe.txt")
    nf_test.save("nf_prize_dataset/nf_probe.npz")
    nf_training.save("nf_prize_dataset/nf_training.npz")
    small_nf_prize = nf_prize.find_dense_subset(5000, 250)
    small_nf_prize.save("nf_prize_dataset/small_nf_prize.npz")
    small_nf_test, small_nf_training = small_nf_prize.split_probe_subset("nf_prize_dataset/probe.txt")
    small_nf_test.save("nf_prize_dataset/small_nf_probel.npz")
    small_nf_training.save("nf_prize_dataset/small_nf_training.npz")

if __name__ == "__main__":
    main(sys.argv[1:])
