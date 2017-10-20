'''
    Matirx factorization method for recommendation system
    R = P * Q.transpose
    shape of R: (U, D)
    shape of P: (U, K)
    shape of Q: (D, K)
    U: number of users
    D: number of items
    K: number of features
'''
import os
import tensorflow as tf
from logger import EmptyLogger


class MatrixFactorization(object):
    def __init__(self, config, K=10, logger=None):
        '''
        :param config: configuration object
        :param U: number of users
        :param D: number of items
        :param K: number of features
        :param logger: logger object
        '''
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = EmptyLogger()
        U = self.config.max_users_num
        D = self.config.max_items_num
        # self.P = tf.get_variable('P', [U, K], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # self.Q = tf.get_variable('Q', [D, K], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.P = tf.Variable(initial_value=tf.truncated_normal([U, K]), name='users')
        self.Q = tf.Variable(initial_value=tf.truncated_normal([D, K]), name='items')
        self.users_indices = tf.placeholder(tf.int32, [None])
        self.items_indices = tf.placeholder(tf.int32, [None])
        self.rates = tf.placeholder(tf.float32, [None])
        self._build()

    def _build(self):
        # mask = tf.not_equal(self.R, tf.constant(0.0), 'mask')
        # self.logits_matrix = tf.matmul(self.P, self.Q, transpose_b=True)
        # self.loss = tf.reduce_sum(tf.pow(tf.boolean_mask(self.R, mask) - tf.boolean_mask(self.logits_matrix, mask), 2))
        # self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

        # # ensure the elements in P and Q are non-negative
        # self.P = self.P.assign(tf.maximum(tf.zeros_like(self.P, tf.float32), self.P))
        # self.Q = self.Q.assign(tf.maximum(tf.zeros_like(self.Q, tf.float32), self.Q))
        self.logits_matrix = tf.matmul(self.P, self.Q, transpose_b=True)
        self.logits_matrix_flatten = tf.reshape(self.logits_matrix, [-1])
        self.R = tf.gather(self.logits_matrix_flatten, self.users_indices *tf.shape(self.logits_matrix)[1] + self.items_indices)
        self.loss = tf.reduce_sum(tf.abs(self.R - self.rates))
        lr = tf.constant(self.config.learning_rate, name='learning_rate')
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 10000, self.config.decay_rate, staircase=True, name='learning_rate')
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess, users, items, rates):
        if not os.path.exists(os.path.dirname(self.config.checkpoints_dir)):
            os.mkdir(os.path.dirname(self.config.checkpoints_dir))
        if not os.path.exists(self.config.checkpoints_dir):
            os.mkdir(self.config.checkpoints_dir)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        start_epoch = 0
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            self.logger.info('restore from checkpoint {0}'.format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        self.logger.info('start training...')
        epoch = start_epoch
        try:
            for i in range(start_epoch, self.config.epochs):
                epoch = i
                loss, _, R = sess.run([self.loss, self.train_op, self.R], feed_dict={self.users_indices: users,
                                        self.items_indices: items, self.rates: rates})
                self.logger.info('iteration: {0}, training loss: {1}'.format(epoch, loss))
                self.logger.info('R shape: {0} R: {1}'.format(R.shape, R))
                self.logger.info('rates shape: {0} rates: {1}'.format(rates.shape, rates))
                # if epoch % 6 == 0:
                #     saver.save(sess, self.config.model_dir, global_step= epoch)
        except KeyboardInterrupt:
            self.logger.error('Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, self.config.model_dir, global_step=epoch)
            self.logger.info('Last epoch were saved, next time will start from epoch {0}.'.format(epoch))

    def evaluate(self, sess, users, items, rates):
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)
        loss = sess.run([self.loss],
                        feed_dict={
                        self.users_indices: users,
                        self.items_indices: items,
                        self.rates: rates})
        self.logger.info('evaluation loss: {0}'.format(loss))

    def predict(self, sess, users, items):
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)
        predict = sess.run([self.R],
                        feed_dict={
                        self.users_indices: users,
                        self.items_indices: items})
        self.logger.info('finished predicting...')
        return predict
