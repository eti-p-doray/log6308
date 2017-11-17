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
MODEL_NAME = "netflix_0.0_latent"

def preprocess_word_nt(movie_titles):
    nt = {}
    for movie_title in movie_titles:
        for word in movie_title:
            if word in nt:
                nt[word] = nt[word] + 1
            else:
                nt[word] = 1
    return nt

def preprocess_title_embeddings(model, movie_titles, nt):
    embeddings = []
    for movie_title in movie_titles:
        embedding = []
        length = 0
        for word in movie_title:
            if word in model.wv.vocab:
                word_vector = model[word].tolist()
                tfidf = log10(len(movie_titles) / nt[word])
                embedding = embedding + [x * tfidf for x in word_vector]
                length = length + 1
        embedding = [x / length for x in embedding]
        embeddings.append(embedding)

    return embeddings

def preprocess_get_titles(utils):
    if utils.args.gen_movie_titles:
        all_titles = {}
        with open('nf_prize_dataset/movie_titles.tsv') as f:
            first = True
            for line in f:
                if first: #ignore header line
                    first = False
                    continue
                words = line.split()
                movie_id = words.pop(0)
                words.pop(0) #remove year
                title = " ".join(words)
                all_titles[int(movie_id)] = title

        training_movie_titles = []
        test_movie_titles = []
        for movie_id in utils.training_set.movie_ids:
            # + 1 because we offseted it by minus one on npy creation
            training_movie_titles.append(all_titles[movie_id + 1])
        training_array = numpy.array(training_movie_titles, dtype=numpy.string_)
        numpy.save(utils.args.input + '.titles.npy', training_array)
        for movie_id in utils.test_set.movie_ids:
            # + 1 because we offseted it by minus one on npy creation
            test_movie_titles.append(all_titles[movie_id + 1])
        test_array = numpy.array(test_movie_titles, dtype=numpy.string_)
        numpy.save(utils.args.test_set + '.titles.npy', test_array)

        logging.debug('Titles generated and saved')

    else:
        training_array = numpy.load(utils.args.input + '.titles.npy')
        test_array = numpy.load(utils.args.test_set + '.titles.npy')
        logging.debug('Title loaded from existing')
    return training_array.tolist() + test_array.tolist()


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
    movie_embedding_size = 60

    utils.init_tensorflow()

    movie_titles = preprocess_get_titles(utils)

    model = KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True)
    title_embedding_value = preprocess_title_embeddings(model, movie_titles, preprocess_word_nt(movie_titles))
    title_embeddings = tf.constant(numpy.asarray(title_embedding_value))
    #embedded_titles = tf.gather(title_embeddings, utils.movie_ids)
    logging.debug('done with preprocess')


    #Trainable embeddings for users. Tensor format n_user x user_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    user_embeddings = tf.get_variable("user_embeddings", initializer=tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.1), trainable=True)
    embedded_users = tf.gather(user_embeddings, utils.user_ids) #Loads embeddings of currently treated users.

    #Trainable embeddings for movies. Tensor format n_movie x movie_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    movie_embeddings = tf.get_variable("movie_embeddings", initializer=tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.1), trainable=True)
    embedded_movies = tf.gather(movie_embeddings, utils.movie_ids)#Loads embeddings of currently treated movies.

    # Biases
    user_bias = tf.get_variable("user_bias", initializer=tf.zeros([netflix_data.USER_COUNT]))
    movie_bias = tf.get_variable("movie_bias", initializer=tf.zeros([netflix_data.MOVIE_COUNT]))
    B = numpy.mean(utils.training_set.ratings)

    # Layer description
    Y = tf.reduce_sum(tf.multiply(embedded_movies, embedded_users), 1) + tf.gather(user_bias, utils.user_ids) + tf.gather(movie_bias, utils.movie_ids) + B

    #Error calculation
    mse = tf.reduce_mean(tf.square(tf.cast(utils.ratings, tf.float32) - Y))
    rmse = tf.sqrt(mse)
    # Loss function to minimize
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.reduce_sum(tf.square(embedded_users), 1)) +
           tf.reduce_mean(tf.reduce_sum(tf.square(embedded_movies), 1)) +
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
    utils.setup_projector(movie_embeddings.name)


    ############################################################################
    ## Training loop.
    train_data_update_freq = 20
    test_data_update_freq = 1000
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, rmse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
