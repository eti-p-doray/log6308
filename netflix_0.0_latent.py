import logging
import sys

from netflix_utils import NetflixUtils
import netflix_data

import tensorflow as tf

DEFAULT_N_ITER = 500000
BATCH_SIZE = 512
REGULARIZATION_FACTOR = 0.1
LEARNING_SPEED = 0.2
MODEL_NAME = "netflix_0.0_latent"


def main(argv):
    utils = NetflixUtils(MODEL_NAME, DEFAULT_N_ITER)
    utils.parse_args(argv)

    user_embedding_size = 60
    movie_embedding_size = 60

    utils.init_tensorflow(user_embedding_size, movie_embedding_size, True)

    user_bias = tf.get_variable("user_bias", initializer=tf.zeros([netflix_data.USER_COUNT]))
    movie_bias = tf.get_variable("movie_bias", initializer=tf.zeros([netflix_data.MOVIE_COUNT]))

    B = tf.Variable(tf.zeros([1]))
    Y = tf.reduce_sum(tf.multiply(utils.embedded_movies, utils.embedded_users), 1) + tf.gather(user_bias, utils.user_ids) + tf.gather(movie_bias, utils.movie_ids) + B

    mse = tf.reduce_mean(tf.square(tf.cast(utils.ratings, tf.float32) - Y))
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.reduce_sum(tf.square(utils.embedded_users), 1)) +
           tf.reduce_mean(tf.reduce_sum(tf.square(utils.embedded_movies), 1)) +
           tf.reduce_mean(tf.square(tf.gather(user_bias, utils.user_ids))) +
           tf.reduce_mean(tf.square(tf.gather(movie_bias, utils.movie_ids)))))

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
    utils.setup_projector()

    train_data_update_freq = 20
    test_data_update_freq = 100
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
