import logging
import sys

from netflix_utils import NetflixUtils
import netflix_data

import tensorflow as tf

BATCH_SIZE = 512
DEFAULT_N_ITER = int(40 * netflix_data.NUM_RATING / BATCH_SIZE)
REGULARIZATION_FACTOR = 0.1
LEARNING_SPEED = 0.5
MODEL_NAME = "netflix_2.0_kernel"


def main(argv):
    utils = NetflixUtils(MODEL_NAME, DEFAULT_N_ITER)
    utils.parse_args(argv)
    utils.load_data()

    user_embedding_size = 40
    movie_embedding_size = 40

    utils.init_tensorflow()

    user_embeddings = tf.get_variable("user_embeddings",
                                      initializer=tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size],
                                                                      stddev=0.1))
    user_covariance = tf.get_variable("user_bias", initializer=tf.ones([netflix_data.USER_COUNT, user_embedding_size]))
    embedded_users = tf.gather(user_embeddings, utils.user_ids)
    user_covariance_gathered = tf.gather(user_covariance, utils.user_ids)

    movie_embeddings = tf.get_variable("movie_embeddings",
                                       initializer=tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size],
                                                                       stddev=0.1))
    movie_covariance = tf.get_variable("movie_bias", initializer=tf.ones([netflix_data.MOVIE_COUNT, movie_embedding_size]))
    embedded_movies = tf.gather(movie_embeddings, utils.movie_ids)
    movies_covariance_gathered = tf.gather(movie_covariance, utils.movie_ids)

    B = tf.Variable(tf.zeros([1]))
    delta = embedded_users - embedded_movies
    Y = 4 * tf.sigmoid(
        -tf.reduce_sum(tf.divide(delta ** 2, user_covariance_gathered**2 + movies_covariance_gathered**2), axis=1) + B) + 1

    mse = tf.reduce_mean(tf.square(tf.cast(utils.ratings, tf.float32) - Y))
    loss = (mse + REGULARIZATION_FACTOR * (
        tf.reduce_mean(tf.reduce_sum(tf.square(embedded_users), 1)) +
        tf.reduce_mean(tf.reduce_sum(tf.square(embedded_movies), 1))))

    correct_prediction = tf.equal(tf.round(Y), tf.cast(utils.ratings, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf.train.AdadeltaOptimizer(LEARNING_SPEED).minimize(loss, global_step=utils.global_step)

    # init
    logging.debug('Initializing model')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    utils.restore_existing_checkpoint(sess)
    utils.setup_projector(movie_embeddings.name)

    train_data_update_freq = 20
    test_data_update_freq = 1000
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])

