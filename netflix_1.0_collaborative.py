import argparse
import logging
import os
import sys

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import netflix_data

# Constants
DEFAULT_N_ITER = 500000
BATCH_SIZE = 512
REGULARIZATION_FACTOR = 0.1
LEARNING_SPEED = 0.5
LOG_DIR = "log/netflix_1.0_collaborative"
MODEL_NAME = "netflix_1.0_collaborative"

#Main Script
def main(argv):
    ############################################################################
    ## Building a parser to interpret the arguments given with the script.
    ## Using -h with the script will show its description.
    parser = argparse.ArgumentParser(description="""
    Simple matrix factorization algorithm to find latent variables that represents netflix prize data.
    """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', default = 'nf_prize_dataset/nf_prize.npy',
                        help='Netflix input data input file.')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Saved session checkpoint, -1 for latest')
    parser.add_argument('--n_iter', type=int, default=DEFAULT_N_ITER,
                        help='Total number of iterations')

    # Configuring Log severity for future printing
    logging.basicConfig(level=logging.DEBUG)

    args = parser.parse_args() # Use the parser to get the arguments
    n_iter = args.n_iter
    logging.info("Number of iterations: " + str(n_iter))


    ############################################################################
    ## Description of the TensorFlow model.

    #Constants
    user_embedding_size = 20
    movie_embedding_size = 60

    #Placeholders for data to be treated
    user_ids = tf.placeholder(tf.int32, shape=[None])
    movie_ids = tf.placeholder(tf.int32, shape=[None])
    ratings = tf.placeholder(tf.int8, shape=[None])

    #Global step(iteration) identifier. Will be incremented, not trained.
    global_step = tf.get_variable('global_step', initializer=0, trainable=False)

    #Trainable embeddings for users. Tensor format n_user x user_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    user_embeddings = tf.get_variable("user_embeddings", initializer=tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.01), trainable=True)
    embedded_users = tf.gather(user_embeddings, user_ids) #Loads embeddings of currently treated users.

    #Trainable embeddings for movies. Tensor format n_movie x movie_embedding_size.
    #Initialized randomly accorded to a truncated normal distribution
    movie_embeddings = tf.get_variable("movie_embeddings", initializer=tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.01), trainable=True)
    embedded_movies = tf.gather(movie_embeddings, movie_ids)#Loads embeddings of currently treated movies.

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

    mse = tf.reduce_mean(tf.square(tf.cast(ratings, tf.float32) - Y)) #Error calculation
    # Loss function to minimize
    loss = (mse + REGULARIZATION_FACTOR*(
           tf.reduce_mean(tf.square(embedded_users)) +
           tf.reduce_mean(tf.square(embedded_movies))))

    # Metric : check if correct prediction and calculate accuracy.
    correct_prediction = tf.equal(tf.round(Y), tf.cast(ratings, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    train_step = tf.train.GradientDescentOptimizer(LEARNING_SPEED).minimize(loss, global_step=global_step)

    ############################################################################
    ## Session Initialization and restoration
    logging.debug('Initializing model')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver() #To be able to save and restore variable status

    # If a checkpoint was given in argument to the script, we will import the
    # variables' status as it was at that checkpoint. NB : -1 implies latest checkpoint.
    if args.checkpoint is not None and os.path.exists(os.path.join(LOG_DIR, 'checkpoint')):
        if args.checkpoint == -1: #latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        else: #Specified checkpoint
            saver.restore(sess, os.path.join(LOG_DIR, MODEL_NAME+".ckpt-"+str(args.checkpoint)))
        logging.debug('Model restored to step ' + str(global_step.eval(sess)))

    ############################################################################
    ## Visualisation configuration for TensorBoard

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = movie_embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = '../movie_titles.tsv'

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    ############################################################################
    ## Data loading in batches of 5000
    logging.debug('Loading netflix data')
    data = netflix_data.DataSet.fromfile("nf_prize_dataset/nf_prize.npy").split(0, 5000)
    train_batch_iter = data.train.iter_batch(BATCH_SIZE)

    ############################################################################
    ## Training loop.

    train_data_update_freq = 20
    test_data_update_freq = 100
    sess_save_freq = 5000

    logging.debug('Training model')
    while global_step.eval(sess) < n_iter:
        batch = next(train_batch_iter) # next batch of data to train on.

        # the backpropagation training step
        sess.run(train_step, feed_dict={user_ids: batch.user_ids, movie_ids: batch.movie_ids, ratings: batch.ratings})
        i = global_step.eval(sess)

        # compute training values for visualisation.
        if i % train_data_update_freq == 0:
            a, m, l = sess.run([accuracy, tf.sqrt(mse), loss],
                                     feed_dict={user_ids: batch.user_ids, movie_ids: batch.movie_ids, ratings: batch.ratings})
            logging.info(str(i) + ": accuracy:" + str(a) + " loss: " + str(l) + " rmse: " + str(m))

        # compute test values for visualisation
        if i % test_data_update_freq == 0:
            a, m, l = sess.run([accuracy, tf.sqrt(mse), loss],
                               feed_dict={user_ids: data.test.user_ids, movie_ids: data.test.movie_ids,
                                          ratings: data.test.ratings})
            logging.info(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(
                l) + " rmse: " + str(m))

        # Saving training progress
        if i % sess_save_freq == 0:
            logging.debug('Saving model')
            saver.save(sess, os.path.join(LOG_DIR, MODEL_NAME+".ckpt"), global_step=i)

if __name__ == "__main__":
    main(sys.argv[1:])
