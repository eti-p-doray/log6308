import logging
import sys
import numpy

from netflix_utils import NetflixUtils
import netflix_data

import tensorflow as tf

BATCH_SIZE = 512
DEFAULT_N_ITER = int(40 * netflix_data.NUM_RATING / BATCH_SIZE)
REGULARIZATION_FACTOR = 0.5
LEARNING_SPEED = 0.5
MODEL_NAME = "netflix_0.1_latent"


def linear_bezier(t, p1, p2):
    t_ = t[:, tf.newaxis]
    return (1 - t_) * p1 + t_ * p2


def cubic_bezier(t, p1, p2, p3):
    t_ = t[:,tf.newaxis]
    return (1-t_) * ((1-t_) * p1 + t_*p2) + t_ * ((1-t_)*p2+t_*p3)


def main(argv):
    utils = NetflixUtils(MODEL_NAME, DEFAULT_N_ITER)
    utils.parse_args(argv)
    utils.load_data()

    user_embedding_size = 20
    movie_embedding_size = 20

    utils.init_tensorflow()
    #Trainable embeddings for users. Tensor format n_user x user_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    user_embeddings_initializer = tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.1)
    user_embeddings1 = tf.get_variable("user_embeddings1",
                                      initializer=user_embeddings_initializer, trainable=True)
    user_embeddings2 = tf.get_variable("user_embeddings2",
                                      initializer=user_embeddings_initializer, trainable=True)
    user_embeddings3 = tf.get_variable("user_embeddings3",
                                      initializer=user_embeddings_initializer, trainable=True)
    embedded_users1 = tf.gather(user_embeddings1, utils.user_ids)  # Loads embeddings of currently treated users.
    embedded_users2 = tf.gather(user_embeddings2, utils.user_ids)  # Loads embeddings of currently treated users.
    embedded_users3 = tf.gather(user_embeddings3, utils.user_ids)  # Loads embeddings of currently treated users.

    #Trainable embeddings for movies. Tensor format n_movie x movie_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    movie_embeddings = tf.get_variable("movie_embeddings", initializer=tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.1), trainable=True)
    embedded_movies = tf.gather(movie_embeddings, utils.movie_ids)#Loads embeddings of currently treated movies.

    user_bias1 = tf.get_variable("user_bias1", initializer=tf.zeros([netflix_data.USER_COUNT]))
    user_bias2 = tf.get_variable("user_bias2", initializer=tf.zeros([netflix_data.USER_COUNT]))
    user_bias3 = tf.get_variable("user_bias3", initializer=tf.zeros([netflix_data.USER_COUNT]))
    movie_bias = tf.get_variable("movie_bias", initializer=tf.zeros([netflix_data.MOVIE_COUNT]))
    gathered_user_bias1 = tf.gather(user_bias1, utils.user_ids)
    gathered_user_bias2 = tf.gather(user_bias2, utils.user_ids)
    gathered_user_bias3 = tf.gather(user_bias3, utils.user_ids)
    gathered_movie_bias = tf.gather(movie_bias, utils.movie_ids)

    min_date = numpy.min(utils.training_set.dates)
    max_date = numpy.max(utils.training_set.dates)
    normalized_dates = tf.cast(utils.dates - min_date, tf.float32) / (max_date - min_date)

    user_bias_interpolated = cubic_bezier(normalized_dates,
                                          gathered_user_bias1[:,tf.newaxis],
                                          gathered_user_bias2[:,tf.newaxis],
                                          gathered_user_bias3[:,tf.newaxis])
    embedded_users_interpolated = cubic_bezier(normalized_dates, embedded_users1, embedded_users2, embedded_users3)

    B = numpy.mean(utils.training_set.ratings)
    Y = tf.reduce_sum(tf.multiply(embedded_movies, embedded_users_interpolated), 1) + user_bias_interpolated + gathered_movie_bias + B

    mse = tf.reduce_mean(tf.square(tf.cast(utils.ratings, tf.float32) - Y))
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.reduce_sum(tf.square(embedded_users_interpolated), 1)) +
           tf.reduce_mean(tf.reduce_sum(tf.square(embedded_movies), 1)) +
           tf.reduce_mean(tf.square(user_bias_interpolated)) +
           tf.reduce_mean(tf.square(gathered_movie_bias))))

    correct_prediction = tf.equal(tf.round(Y), tf.cast(utils.ratings, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(LEARNING_SPEED).minimize(loss, global_step=utils.global_step)

    # init
    logging.debug('Initializing model')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    utils.restore_existing_checkpoint(sess)
    utils.setup_projector(movie_embeddings.name)

    train_data_update_freq = 20
    test_data_update_freq = 100000
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
