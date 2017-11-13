import logging
import sys

from netflix_utils import NetflixUtils
import netflix_data

import tensorflow as tf


# Constants
DEFAULT_N_ITER = 500000
BATCH_SIZE = 512
REGULARIZATION_FACTOR = 0.1
LEARNING_SPEED = 0.5
MODEL_NAME = "netflix_1.0_collaborative"

#Main Script
def main(argv):
    utils = NetflixUtils(MODEL_NAME, DEFAULT_N_ITER)

    utils.parse_args(argv)

    ############################################################################
    ## Description of the TensorFlow model.
    #Constants
    user_embedding_size = 20
    movie_embedding_size = 60

    utils.init_tensorflow()

    #Trainable embeddings for users. Tensor format n_user x user_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    user_embeddings = tf.get_variable("user_embeddings", initializer=tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.01), trainable=True)
    embedded_users = tf.gather(user_embeddings, utils.user_ids) #Loads embeddings of currently treated users.

    #Trainable embeddings for movies. Tensor format n_movie x movie_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    movie_embeddings = tf.get_variable("movie_embeddings", initializer=tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.01), trainable=True)
    embedded_movies = tf.gather(movie_embeddings, utils.movie_ids)#Loads embeddings of currently treated movies.

    #Combine embeddings together along their 2nd dimension.
    #This should result into a tensor with user_embedding_size + movie_embedding_size embedddings
    X = tf.concat((embedded_users, embedded_movies), 1)

    ## Weights and Biases, trainable
    W1 = tf.Variable(tf.truncated_normal([user_embedding_size + movie_embedding_size, 64], stddev=0.1))
    B1 = tf.Variable(tf.zeros([64]))
    W2 = tf.Variable(tf.truncated_normal([64, 64], stddev=0.1))
    B2 = tf.Variable(tf.zeros([64]))
    W3 = tf.Variable(tf.truncated_normal([64, 64], stddev=0.1))
    B3 = tf.Variable(tf.zeros([64]))
    W4 = tf.Variable(tf.truncated_normal([64, 1], stddev=0.1))
    B4 = tf.Variable(tf.zeros([1]))

    ## Layers, combined using Sigmoid function
    Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    Y = tf.matmul(Y3, W4) + B4

    mse = tf.reduce_mean(tf.square(tf.cast(utils.ratings, tf.float32) - Y)) #Error calculation
    rmse = tf.sqrt(mse)
    # Loss function to minimize
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.square(embedded_users)) +
           tf.reduce_mean(tf.square(embedded_movies))))

    # Metric : check if correct prediction and calculate accuracy.
    correct_prediction = tf.equal(tf.round(Y), tf.cast(utils.ratings, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(LEARNING_SPEED).minimize(loss, global_step=utils.global_step)

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
    test_data_update_freq = 100
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, rmse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
