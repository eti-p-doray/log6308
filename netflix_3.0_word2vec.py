import logging
import sys
import numpy
from gensim.models.keyedvectors import KeyedVectors
from math import log10

from netflix_utils import NetflixUtils
import netflix_data

import tensorflow as tf

BATCH_SIZE = 512
DEFAULT_N_ITER = int(40 * netflix_data.NUM_RATING / BATCH_SIZE)
REGULARIZATION_FACTOR = 0.5
LEARNING_SPEED = 0.5
MODEL_NAME = "netflix_3.0_word2vec"

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
    embeddings = numpy.zeros((len(movie_titles), ncol))
    for i, movie_title in enumerate(movie_titles):
        embedding = numpy.zeros((ncol))
        length = 0
        for word in movie_title:
            if word in model.wv.vocab:
                word_vector = model[word][0:ncol]
                tfidf = log10(len(movie_titles) / nt[word])
                embedding = embedding + tfidf * word_vector
                length = length + 1
        embedding = embedding / length
        embeddings[i, :] = embedding

    return embeddings

def preprocess_get_titles(utils):
    if utils.args.gen_movie_titles:
        title_array = numpy.zeros(netflix_data.MOVIE_COUNT, dtype=numpy.unicode_)
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
    return title_array.tolist()


def main(argv):
    ############################################################################
    # Initilizing the utility object for common tasks among our models
    utils = NetflixUtils(MODEL_NAME, DEFAULT_N_ITER)
    utils.parse_args(argv)
    utils.load_data()

    ############################################################################
    ## Description of the TensorFlow model.
    #Constants
    user_embedding_size = 60

    utils.init_tensorflow()

    movie_titles = preprocess_get_titles(utils)

    model = KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    title_embedding_value = preprocess_title_embeddings(model, movie_titles, preprocess_word_nt(movie_titles), user_embedding_size)

    logging.debug('done with preprocess')

    title_embeddings = tf.get_variable("title_embeddings", initializer=tf.constant(numpy.asarray(title_embedding_value), dtype=numpy.float32), trainable=False)
    embedded_titles = tf.gather(title_embeddings, utils.movie_ids)

    logging.debug("title embedding initialized")


    #Trainable embeddings for users. Tensor format n_user x user_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    user_embeddings = tf.get_variable("user_embeddings", initializer=tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.1), trainable=True)
    embedded_users = tf.gather(user_embeddings, utils.user_ids) #Loads embeddings of currently treated users.

    #Trainable embeddings for movies. Tensor format n_movie x movie_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    #movie_embeddings = tf.get_variable("movie_embeddings", initializer=tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.1), trainable=True)
    #embedded_movies = tf.gather(movie_embeddings, utils.movie_ids)#Loads embeddings of currently treated movies.

    # Biases
    user_bias = tf.get_variable("user_bias", initializer=tf.zeros([netflix_data.USER_COUNT]))
    movie_bias = tf.get_variable("movie_bias", initializer=tf.zeros([netflix_data.MOVIE_COUNT]))
    B = numpy.mean(utils.training_set.ratings)

    # Layer description
    Y = tf.reduce_sum(tf.multiply(embedded_titles, embedded_users), 1) + tf.gather(user_bias, utils.user_ids) + tf.gather(movie_bias, utils.movie_ids) + B

    #Error calculation
    mse = tf.reduce_mean(tf.square(tf.cast(utils.ratings, tf.float32) - Y))
    # Loss function to minimize
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.reduce_sum(tf.square(embedded_users), 1)) +
           tf.reduce_mean(tf.square(tf.gather(user_bias, utils.user_ids))) +
           tf.reduce_mean(tf.square(tf.gather(movie_bias, utils.movie_ids)))))

   # Metric : check if correct prediction and calculate accuracy.
    correct_prediction = tf.equal(tf.round(Y), tf.cast(utils.ratings, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training setup
    train_step = tf.train.AdadeltaOptimizer(LEARNING_SPEED).minimize(loss, global_step=utils.global_step)

    ############################################################################
    ## Session Initialization and restoration
    logging.debug('Initializing model')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    utils.restore_existing_checkpoint(sess)
    utils.setup_projector(title_embeddings.name)


    ############################################################################
    ## Training loop.
    train_data_update_freq = 20
    test_data_update_freq = 1000
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
