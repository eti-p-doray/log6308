import argparse
import logging
import os
import csv

import numpy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import netflix_data
import utility

class NetflixUtils(object):

    ### Logical order of calls :
    ### 1. parse_args
    ### 2. init_tensorflow
    ### 3. the caller should declare the ts layers and continue on ts order of
    ###    operation until sess.run(init)
    ### 4. (optional)restore_existing_checkpoint
    ### 5. (optional)setup_projector
    ### 6. train_model


    def __init__(self, model_name, default_n_iter):
        self.args_ = None
        self.model_name_ = model_name
        self.default_n_iter_ = default_n_iter
        self.user_ids_ = None
        self.movie_ids_ = None
        self.dates_ = None
        self.ratings_ = None
        self.global_step_ = None
        self.saver_ = None
        self.training_set_ = None
        self.test_set_ = None
        self.first_write = True

    @property
    def args(self):
        return self.args_

    @property
    def model_name(self):
        return self.model_name_

    @property
    def default_n_iter(self):
        return self.default_n_iter_

    @property
    def user_ids(self):
        return self.user_ids_

    @property
    def movie_ids(self):
        return self.movie_ids_

    @property
    def dates(self):
        return self.dates_

    @property
    def ratings(self):
        return self.ratings_

    @property
    def global_step(self):
        return self.global_step_

    @property
    def training_set(self):
        return self.training_set_

    @property
    def test_set(self):
        return self.test_set_

    @property
    def saver(self):
        return self.saver_

    def parse_args(self, argv):
        ############################################################################
        ## Building a parser to interpret the arguments given with the script.
        ## Using -h with the script will show its description.
        parser = argparse.ArgumentParser(description="""
        Simple matrix factorization algorithm to find latent variables that represents netflix prize data.
        """, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-i', '--input', default = 'nf_prize_dataset/nf_probe_training.npy',
                            help='Netflix input data input file.')
        parser.add_argument('-t', '--test_set', default = 'nf_prize_dataset/nf_probe.npy',
                            help='Netflix test set input file.')
        parser.add_argument('--checkpoint', type=int, default=None,
                            help='Saved session checkpoint, -1 for latest.')
        parser.add_argument('--n_iter', type=int, default=self.default_n_iter_,
                            help='Total number of iterations.')
        parser.add_argument('--logdir', default="log/" + self.model_name_,
                            help='Directory where logs should be written.')
        parser.add_argument('--log_perf', action='store_true',
                            help='Enables performance logging.')
        parser.add_argument('--gen_movie_titles', action='store_true',
                            help='Enables movie title list generation for each datasets')

        # Configuring Log severity for future printing
        logging.basicConfig(level=logging.DEBUG)

        self.args_ = parser.parse_args(argv) # Use the parser to get the arguments
        logging.info("Number of iterations: " + str(self.args_.n_iter))

    def init_tensorflow(self):
        #Placeholders for data to be treated
        self.user_ids_ = tf.placeholder(tf.int32, shape=[None])
        self.movie_ids_ = tf.placeholder(tf.int32, shape=[None])
        self.dates_ = tf.placeholder(tf.int32, shape=[None])
        self.ratings_ = tf.placeholder(tf.int8, shape=[None])

        #Global step(iteration) identifier. Will be incremented, not trained.
        self.global_step_ = tf.get_variable('global_step', initializer=0, trainable=False)

    def restore_existing_checkpoint(self, sess):
        self.saver_ = tf.train.Saver() #To be able to save and restore variable status
        # If a checkpoint was given in argument to the script, we will import the
        # variables' status as it was at that checkpoint. NB : -1 implies latest checkpoint.
        if self.args_.checkpoint is not None and os.path.exists(os.path.join(self.args_.logdir, 'checkpoint')):
            if self.args_.checkpoint == -1:#latest checkpoint
                self.saver_.restore(sess, tf.train.latest_checkpoint(self.args_.logdir))
            else:#Specified checkpoint
                self.saver_.restore(sess, os.path.join(self.args_.logdir, self.model_name_+".ckpt-"+str(self.args_.checkpoint)))
            self.first_write = False
            logging.debug('Model restored to step ' + str(self.global_step_.eval(sess)))

    def setup_projector(self, movie_tensor_name):
        ############################################################################
        ## Visualisation configuration for TensorBoard

        # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = movie_tensor_name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = '../movie_titles.tsv'

        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(self.args_.logdir)

        # The next line writes a projector_config.pbtxt in the logdir.
        # TensorBoard will read this file during startup.
        projector.visualize_embeddings(summary_writer, config)

    def load_data(self):
        logging.debug('Loading netflix data')
        self.training_set_ = netflix_data.DataSet.fromfile(self.args_.input)
        self.test_set_ = netflix_data.DataSet.fromfile(self.args_.test_set)

    def save_perfo(self, values, is_test, clear = False):
        if self.args.log_perf:
            mode = 'w' if (clear) else 'a'
            filename = 'perf_test.csv' if (is_test) else 'perf_train.csv'
            with open(os.path.join(self.args_.logdir, filename), mode, newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                if (clear):
                    csvwriter.writerow(("i", "Accuracy", "RMSE", "Loss"))
                for value in values:
                    csvwriter.writerow(value)

    def train_model(self, sess, train_step, accuracy, mse, loss,
                    train_data_update_freq, test_data_update_freq,
                    sess_save_freq, batch_size):

        train_batch_iter = self.training_set.iter_batch(batch_size)
        logging.debug('Training model')
        training_results = []
        test_results = []
        while self.global_step_.eval(sess) < self.args_.n_iter:
            batch = next(train_batch_iter) # next batch of data to train on.

            # the backpropagation training step
            sess.run(train_step, feed_dict={self.user_ids_: batch.user_ids,
                                            self.movie_ids_: batch.movie_ids,
                                            self.dates_ : batch.dates,
                                            self.ratings_: batch.ratings})
            i = self.global_step_.eval(sess)

            # compute training values for visualisation.
            if i % train_data_update_freq == 0:
                a, e, l = sess.run([accuracy, mse, loss],
                                         feed_dict={self.user_ids_: batch.user_ids,
                                                    self.movie_ids_: batch.movie_ids,
                                                    self.dates_ : batch.dates,
                                                    self.ratings_: batch.ratings})
                logging.info(str(i) + ": accuracy:" + str(a) + " loss: " + str(l) + " rmse: " + str(numpy.sqrt(e)))
                training_results.append((i, a, numpy.sqrt(e), l))

            # compute test values for visualisation
            if i % test_data_update_freq == 0:
                a_total = 0
                e_total = 0
                l_total = 0
                for (first, last) in utility.split_range(0, self.test_set.ratings.shape[0], batch_size):
                    chunk_size = last - first
                    a, e, l = sess.run([accuracy, mse, loss],
                                       feed_dict={self.user_ids_: self.test_set.user_ids[first:last],
                                                  self.movie_ids_: self.test_set.movie_ids[first:last],
                                                  self.dates_: self.test_set.dates[first:last],
                                                  self.ratings_: self.test_set.ratings[first:last]})
                    a_total = a_total + a * chunk_size
                    e_total = e_total + e * chunk_size
                    l_total = l_total + l * chunk_size
                a_total = a_total / self.test_set.ratings.shape[0]
                e_total = e_total / self.test_set.ratings.shape[0]
                l_total = l_total / self.test_set.ratings.shape[0]
                logging.info(str(i) + ": ********* epoch " + str(i) + " ********* test accuracy:" + str(a_total) + " test loss: " + str(
                    l_total) + " rmse: " + str(numpy.sqrt(e_total)))
                test_results.append((i, a_total, numpy.sqrt(e_total), l_total))

            # Saving training progress
            if i % sess_save_freq == 0:
                logging.debug('Saving model')
                self.saver_.save(sess, os.path.join(self.args_.logdir, self.model_name_+".ckpt"), global_step=i)
                self.save_perfo(training_results, False, self.first_write)
                training_results = []
                self.save_perfo(test_results, True, self.first_write)
                test_results = []
                self.first_write = False
