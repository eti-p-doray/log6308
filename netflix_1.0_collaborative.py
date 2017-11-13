import logging
import sys

from netflix_utils import NetflixUtils

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

    utils.init_tensorflow(user_embedding_size, movie_embedding_size, True)

    #Combine embeddings together along their 2nd dimension.
    #This should result into a tensor with user_embedding_size + movie_embedding_size embedddings
    X = tf.concat((utils.embedded_users, utils.embedded_movies), 1)

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
    # Loss function to minimize
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.square(utils.embedded_users)) +
           tf.reduce_mean(tf.square(utils.embedded_movies))))

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
    utils.setup_projector()

    ############################################################################
    ## Training loop.

    train_data_update_freq = 20
    test_data_update_freq = 100
    sess_save_freq = 5000

    utils.train_model(sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, BATCH_SIZE)

if __name__ == "__main__":
    main(sys.argv[1:])
