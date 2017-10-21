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
        self.data, self.product_to_desc, self.reverse_dictionary = set_data(self.config.data_file)

    def _build(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.config.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.config.batch_size, 1])
        self.valid_dataset = tf.placeholder(tf.int32, shape=[None])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([self.config.max_items_num, self.config.embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([self.config.max_items_num, self.config.embedding_size],
                                    stddev=1.0 / math.sqrt(self.config.embedding_size)))
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
        self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        self.normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(
            valid_embeddings, self.normalized_embeddings, transpose_b=True)

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
                    self.config.batch_size, self.config.num_skips, self.config.skip_window, self.data)
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
            self.final_embeddings = self.normalized_embeddings
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
        sim = sess.run(self.similarity, feed_dict={self.valid_dataset: self.valid_examples})
        for i in xrange(self.valid_size):
            valid_word = self.reverse_dictionary[self.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % self.product_to_desc[valid_word]
            for k in xrange(top_k):
                close_word = self.product_to_desc[self.reverse_dictionary[nearest[k]]]
                log_str = '%s %s,\n' % (log_str, close_word)
            self.logger.info(log_str)
        self.logger.info('finished predicting...')