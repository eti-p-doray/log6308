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
MODEL_NAME = "netflix_0.0_latent"


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

    utils.train_model(sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
