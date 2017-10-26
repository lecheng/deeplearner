import os
import tensorflow as tf
import numpy as np
from logger import EmptyLogger
from data.classification import thucnews

class TextClassificationCNN(object):

    def __init__(self, config, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = EmptyLogger()
        self.input_x = tf.placeholder(tf.int32, [None, self.config.text_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_train = True
        self._data_process()

    def _data_process(self):
        self.x_train, self.y_train, self.x_test, self.y_test,\
            self.x_val, self.y_val, self.words = thucnews.preocess_file(self.config.data_dir)

    def _build(self):

        with tf.device('/cpu:0'):
            embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_size],
                                -1.0, 1.0), name='embedding')
            embeded = tf.nn.embedding_lookup(embedding, self.input_x)

        conv = tf.layers.conv1d(embeded, self.config.filters_num, self.config.kernel_size, activation=tf.nn.relu, name='conv')
        self.logger.info('conv shape {0}'.format(conv.shape))
        gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        self.logger.info('gmp shape {0}'.format(gmp.shape))

        fc1 = tf.contrib.layers.fully_connected(gmp, self.config.hidden_dim)
        if self.is_train:
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
        self.logger.info('fc1 shape {0}'.format(fc1.shape))

        self.logits = tf.contrib.layers.fully_connected(fc1, self.config.class_num, activation_fn=None)
        self.logger.info('logits shape {0}'.format(self.logits.shape))
        self.predict_y = tf.argmax(self.logits,1,output_type=tf.int32)
        self.logger.info('predict_y shape {0}'.format(self.predict_y))
        output_onehot = tf.one_hot(self.input_y, self.config.class_num)
        self.logger.info('one hot shape {0}'.format(output_onehot.shape))

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=output_onehot)
        self.total_loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.minimize(self.total_loss)
        correct_pred = tf.equal(self.input_y, self.predict_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, sess):
        self.is_train = True
        self._build()
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
        try:
            for epoch in range(start_epoch, self.config.epochs):
                iterations = len(self.x_train) // self.config.batch_size
                self.logger.info('total iterations: {0}'.format(iterations))
                random_indices = np.random.permutation(np.arange(len(self.x_train)))
                shuffled_x_train = np.array(self.x_train)[random_indices]
                shuffled_y_train = np.array(self.y_train)[random_indices]
                for i in range(iterations):
                    start_index = i * self.config.batch_size
                    end_index = (i + 1) * self.config.batch_size
                    x_train = shuffled_x_train[start_index:end_index]
                    y_train = shuffled_y_train[start_index:end_index]
                    loss, _ = sess.run([
                        self.total_loss,
                        self.train_op
                    ], feed_dict = {self.input_x: x_train, self.input_y: y_train, self.keep_prob: self.config.keep_prob})
                    self.logger.info('Epoch: {0}, iteration: {1}, training loss: {2}'.format(epoch, i, loss))
                if (epoch+1) % 6 == 0:
                    saver.save(sess, self.config.model_dir, global_step= epoch)
        except KeyboardInterrupt:
            self.logger.error('Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, self.config.model_dir, global_step = epoch)
            self.logger.info('Last epoch were saved, next time will start from epoch {0}.'.format(epoch))

    def evaluate(self, sess):
        self.is_train = False
        self._build()
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)
        loss, accuracy = sess.run([self.total_loss, self.accuracy],
                        feed_dict={
                            self.input_x: self.x_test,
                            self.input_y: self.y_test})
        self.logger.info('evaluation loss: {0}, accuracy: {1}'.format(loss, accuracy))

    def predict(self,sess, input_x):
        self.is_train = False
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)
        predict_y = sess.run(self.predict_y,feed_dict={self.input_x: input_x})
        self.logger.info('prediction finished...')

