import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import logging

import netflix_data

N_ITER = 100000
BATCH_SIZE = 512
REGULARIZATION_FACTOR = 0.01
LEARNING_SPEED = 0.5
LOG_DIR = "log"

logging.basicConfig(level=logging.DEBUG)

user_embedding_size = 40
movie_embedding_size = 40

user_ids = tf.placeholder(tf.int32, shape=[None])
movie_ids = tf.placeholder(tf.int32, shape=[None])
ratings = tf.placeholder(tf.int8, shape=[None])

user_embeddings = tf.Variable(tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.1), name="user_embeddings")
user_bias = tf.Variable(tf.zeros([netflix_data.USER_COUNT]))
embedded_users = tf.gather(user_embeddings, user_ids)

movie_embeddings = tf.Variable(tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.1), name="movie_embeddings")
movie_bias = tf.Variable(tf.zeros([netflix_data.MOVIE_COUNT]))
embedded_movies = tf.gather(movie_embeddings, movie_ids)

B = tf.Variable(tf.zeros([1]))
Y = tf.reduce_sum(tf.multiply(embedded_movies, embedded_users), 1) + tf.gather(user_bias, user_ids) + tf.gather(movie_bias, movie_ids) + B

mse = tf.reduce_mean(tf.square(tf.cast(ratings, tf.float32) - Y))
loss = (mse + REGULARIZATION_FACTOR*(
       tf.reduce_mean(tf.square(embedded_users)) +
       tf.reduce_mean(tf.square(embedded_movies)) +
       tf.reduce_mean(tf.square(tf.gather(user_bias, user_ids))) +
       tf.reduce_mean(tf.square(tf.gather(movie_bias, movie_ids)))))

correct_prediction = tf.equal(tf.round(Y), tf.cast(ratings, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(LEARNING_SPEED).minimize(loss)

# init
logging.debug('Initializing model')
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = movie_embeddings.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = 'movie_titles.tsv'

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)

logging.debug('Loading netflix data')
data = netflix_data.DataSet.fromfile("nf_prize_dataset/nf_prize.npy").split(0, 5000)
train_batch_iter = data.train.iter_batch(BATCH_SIZE)

train_data_update_freq = 20
test_data_update_freq = 100
sess_save_freq = 5000

logging.debug('Training model')
for i in range(0, N_ITER):
    batch = next(train_batch_iter)

    # the backpropagation training step
    sess.run(train_step, feed_dict={user_ids: batch.users, movie_ids: batch.movies, ratings: batch.ratings})

    # compute training values for visualisation
    if i % train_data_update_freq == 0:
        a, m, l = sess.run([accuracy, tf.sqrt(mse), loss],
                                 feed_dict={user_ids: batch.users, movie_ids: batch.movies, ratings: batch.ratings})
        logging.info(str(i) + ": accuracy:" + str(a) + " loss: " + str(l) + " rmse: " + str(m))

    # compute test values for visualisation
    if i % test_data_update_freq == 0:
        a, m, l = sess.run([accuracy, tf.sqrt(mse), loss],
                           feed_dict={user_ids: data.test.users, movie_ids: data.test.movies,
                                      ratings: data.test.ratings})
        logging.info(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a) + " test loss: " + str(
            l) + " rmse: " + str(m))

    if i % sess_save_freq == 0:
        logging.debug('Saving model')
        saver.save(sess, os.path.join(LOG_DIR, "netflix_0.0_latent.ckpt"), global_step=i)


