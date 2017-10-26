import tensorflow as tf
import matplotlib.pyplot as plt

import netflix_data

user_embedding_size = 15
movie_embedding_size = 10

user_ids = tf.placeholder(tf.int32, shape=[None])
movie_ids = tf.placeholder(tf.int32, shape=[None])
ratings = tf.placeholder(tf.int8, shape=[None])

user_embeddings = tf.Variable(tf.truncated_normal([netflix_data.USER_COUNT, user_embedding_size], stddev=0.1))
embedded_users = tf.gather(user_embeddings, user_ids)

movie_embeddings = tf.Variable(tf.truncated_normal([netflix_data.MOVIE_COUNT, movie_embedding_size], stddev=0.1))
embedded_movies = tf.gather(movie_embeddings, movie_ids)

X = tf.concat((embedded_users, embedded_movies), 1)

W1 = tf.Variable(tf.truncated_normal([user_embedding_size + movie_embedding_size, 20], stddev=0.1))
B1 = tf.Variable(tf.zeros([20]))
W2 = tf.Variable(tf.truncated_normal([20, 10], stddev=0.1))
B2 = tf.Variable(tf.zeros([10]))
W3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
B3 = tf.Variable(tf.zeros([1]))

Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y = tf.matmul(Y2, W3) + B3

mse = tf.reduce_mean(tf.square(tf.cast(ratings, tf.float32) - Y))
loss = (mse +
       0.0001 * tf.reduce_mean(tf.square(embedded_users)) +
       0.0001 * tf.reduce_mean(tf.square(embedded_movies)))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(19.20,10.80), dpi=70)
plt.gcf().canvas.set_window_title("Netflix prize")
fig.set_facecolor('#FFFFFF')

ax1.set_title("Accuracy", y=1.02)
ax2.set_title("Cross entropy loss", y=1.02)
ax3.set_title("Training digits", y=1.02)
ax4.set_title("Weights", y=1.02)
ax5.set_title("Biases", y=1.02)
ax6.set_title("Test digits", y=1.02)

#plt.show()

train_data = netflix_data.DataSet.fromfile("nf_prize_dataset/nf_prize.npy")
for batch in train_data.iter_batch(500):
    #print(batch.ratings)
    # the backpropagation training step
    sess.run(train_step, feed_dict={user_ids: batch.users, movie_ids: batch.movies, ratings: batch.ratings})

    mse_value, Y_value = sess.run([mse, Y], feed_dict={user_ids: batch.users, movie_ids: batch.movies, ratings: batch.ratings})
    print(mse_value)
    print(Y_value)
    #print(batch.ratings)