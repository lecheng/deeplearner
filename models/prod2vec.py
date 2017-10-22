import tensorflow as tf
import numpy as np
import math
import os
from logger import EmptyLogger
from data.prod2vec import generate_batch, set_data

class Prod2Vec(object):
    def __init__(self, config, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = EmptyLogger()
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self._data()
        self._build()

    def _data(self):
        self.data_p, self.data_u, self.product_to_desc, self.reverse_dict_p, self.reverse_dict_u \
            , self.user_products = set_data(self.config.data_file)

    def _build(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.config.batch_size, 2])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.config.batch_size, 1])
        self.valid_dataset_p = tf.placeholder(tf.int32, shape=[None])
        self.valid_dataset_u = tf.placeholder(tf.int32, shape=[None])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings_p = tf.Variable(
                tf.random_uniform([self.config.max_items_num, self.config.embedding_size_p], -1.0, 1.0))
            embed_p = tf.nn.embedding_lookup(embeddings_p, self.train_inputs[:,0])
            embeddings_u = tf.Variable(
                tf.random_uniform([self.config.max_users_num, self.config.embedding_size_u], -1.0, 1.0))
            embed_u = tf.nn.embedding_lookup(embeddings_u, self.train_inputs[:,1])
            embed = tf.concat([embed_u,embed_p], 1)
            # self.logger.info('embed_p shape {0}'.format(embed_p.shape))
            # self.logger.info('embed_u shape {0}'.format(embed_u.shape))
            # self.logger.info('embed shape {0}'.format(embed.shape))

            embedding_size = self.config.embedding_size_p + self.config.embedding_size_u
            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([self.config.max_items_num, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.config.max_items_num]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=self.train_labels,
                           inputs=embed,
                           num_sampled=self.config.num_sampled,
                           num_classes=self.config.max_items_num))

        # Construct the SGD optimizer using a learning rate of 1.0.
        # self.train_op = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        self.train_op = tf.train.MomentumOptimizer(self.config.learning_rate, 0.9).minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm_p = tf.sqrt(tf.reduce_sum(tf.square(embeddings_p), 1, keep_dims=True))
        self.normalized_embeddings_p = embeddings_p / norm_p
        norm_u = tf.sqrt(tf.reduce_sum(tf.square(embeddings_u), 1, keep_dims=True))
        self.normalized_embeddings_u = embeddings_u / norm_u

        valid_embeddings_p = tf.nn.embedding_lookup(
            self.normalized_embeddings_p, self.valid_dataset_p)
        self.similarity_p = tf.matmul(
            valid_embeddings_p, self.normalized_embeddings_p, transpose_b=True)
        valid_embeddings_u = tf.nn.embedding_lookup(
            self.normalized_embeddings_u, self.valid_dataset_u)
        self.similarity_u = tf.matmul(
            valid_embeddings_u, self.normalized_embeddings_u, transpose_b=True)

    def train(self, sess):
        if not os.path.exists(os.path.dirname(self.config.checkpoints_dir)):
            os.mkdir(os.path.dirname(self.config.checkpoints_dir))
        if not os.path.exists(self.config.checkpoints_dir):
            os.mkdir(self.config.checkpoints_dir)

        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        start_step = 0
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            self.logger.info('restore from checkpoint {0}'.format(checkpoint))
            start_step += int(checkpoint.split('-')[-1])
        self.logger.info('start training...')
        step = start_step
        average_loss = 0
        try:
            for step in range(start_step, self.config.steps):
                batch_inputs, batch_labels = generate_batch(
                    self.config.batch_size, self.config.num_skips, self.config.skip_window, self.data_p, self.data_u)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    self.logger.info('Average loss at step {0} : {1}'.format(step, average_loss))
                    average_loss = 0
            self.final_embeddings_p = self.normalized_embeddings_p
            self.final_embeddings_u = self.normalized_embeddings_u
            saver.save(sess, self.config.model_dir, global_step=step)
        except KeyboardInterrupt:
            self.logger.error('Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, self.config.model_dir, global_step=step)
            self.logger.info('Last epoch were saved, next time will start from epoch {0}.'.format(step))

    def evaluate(self, sess):
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        sim_p, sim_u = sess.run([self.similarity_p, self.similarity_u],
                                feed_dict={
                                    self.valid_dataset_p: self.valid_examples,
                                    self.valid_dataset_u: self.valid_examples
                                })
        # print('reversed_dict_p: {0}'.format(self.reverse_dict_p))
        # print('reversed_dict_u: {0}'.format(self.reverse_dict_u))
        for i in xrange(self.valid_size):
            valid_product = self.reverse_dict_p[self.valid_examples[i]]
            valid_user = self.reverse_dict_u[self.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest_p = (-sim_p[i, :]).argsort()[1:top_k + 1]
            nearest_u = (-sim_u[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest products to %s:\n' % self.product_to_desc[valid_product]
            for k in xrange(top_k):
                # print('nearest{0}_p:{1}'.format(k,nearest_p[k]))
                # print('nearest{0}_p:{1}'.format(k, self.reverse_dict_p[nearest_p[k]]))
                close_word = self.product_to_desc[self.reverse_dict_p[nearest_p[k]]]
                log_str = '%s %s,\n' % (log_str, close_word)
            self.logger.info(log_str)

            log_str = 'Nearest users to {0}[{1}]:\n'.format(valid_user, self.user_products[valid_user])
            for k in xrange(top_k):
                # print('nearest{0}_u:{1}'.format(k, nearest_u[k]))
                close_user = self.reverse_dict_u[nearest_u[k]]
                # print('nearest{0}_u:{1}'.format(k, close_user))
                log_str = '%s %s( %s ),\n' % (log_str, close_user, self.user_products[close_user])
            self.logger.info(log_str)
        self.logger.info('finished predicting...')