import tensorflow as tf

import netflix_data
import visu

REGULARIZATION_FACTOR = 0.1
LEARNING_SPEED = 0.1
BATCH_SIZE = 512

user_embedding_size = 15
movie_embedding_size = 15

user_ids = tf.placeholder(tf.int32, shape=[None])
movie_ids = tf.placeholder(tf.int32, shape=[None])
ratings = tf.placeholder(tf.int8, shape=[None])

user_embeddings = tf.Variable(tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.1))
user_bias = tf.Variable(tf.zeros([netflix_data.USER_COUNT]))
embedded_users = tf.gather(user_embeddings, user_ids)

movie_embeddings = tf.Variable(tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.1))
movie_bias = tf.Variable(tf.zeros([netflix_data.MOVIE_COUNT]))
embedded_movies = tf.gather(movie_embeddings, movie_ids)

X = tf.concat((embedded_users, embedded_movies), 1)

B = tf.Variable(tf.zeros([1]))

Y = tf.matmul(embedded_movies, tf.transpose(embedded_users)) + tf.gather(user_bias, user_ids) + tf.gather(movie_bias, movie_ids) + B

all_embedded_users = tf.concat([tf.reshape(embedded_users, [-1])], 0)
all_embedded_movies  = tf.concat([tf.reshape(embedded_movies, [-1])], 0)

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

datavis = visu.NetflixDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data = netflix_data.DataSet.fromfile("nf_prize_dataset/nf_prize.npy")
train_batch_iter = train_data.iter_batch(BATCH_SIZE)


def training_step(i, update_test_data, update_train_data):
    batch = next(train_batch_iter)

    # compute training values for visualisation
    if update_train_data:
        a, m, l, w, b = sess.run([accuracy, tf.sqrt(mse), loss, all_embedded_users, all_embedded_movies],
                              feed_dict={user_ids: batch.users, movie_ids: batch.movies, ratings: batch.ratings})
        datavis.append_training_curves_data(i, a, l)
        datavis.append_data_histograms(i, w, b)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(l) + " rmse: " + str(m))

    # compute test values for visualisation
    """if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(
            c))"""

    # the backpropagation training step
    sess.run(train_step, feed_dict={user_ids: batch.users, movie_ids: batch.movies, ratings: batch.ratings})

datavis.animate(training_step, iterations=100000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)