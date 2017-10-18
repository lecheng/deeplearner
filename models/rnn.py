# coding=utf-8
import os
import tensorflow as tf
import numpy as np

from data.data import process_poem
from utils.utils import to_word
from logger import EmptyLogger

class TextClassificationRNN(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        pass

    def _build(self):
        pass

    def _train(self):
        pass

    def _evaluate(self):
        pass

class PoemRNN(object):
    def __init__(self, config, model_type, logger=None):
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = EmptyLogger()
        self.model_type = model_type
        self.endpoints = {}
        self._data_process()

    def _data_process(self):
        self.logger.info('loading corpus from {0}'.format(self.config.data_file))
        self.poems_vector, self.word_to_index, self.vocabulary = process_poem(self.config.data_file)
        self.vocab_size = len(self.vocabulary)

    def _build(self):
        cell_type = None
        self.input_data = tf.placeholder(tf.int32, [self.config.batch_size, None], 'input')
        self.output_data = tf.placeholder(tf.int32, [self.config.batch_size, None], 'output')
        if self.model_type == 'rnn':
            cell_type = tf.contrib.rnn.BasicRNNCell
        elif self.model_type == 'gru':
            cell_type = tf.contrib.rnn.GRUCell
        elif self.model_type == 'lstm':
            cell_type = tf.contrib.rnn.BasicLSTMCell
        cell = cell_type(self.config.cell_size, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * self.config.num_layers, state_is_tuple=True)

        # [batch_size, cell_size]
        initial_state = cell.zero_state(self.config.batch_size, tf.float32)
        self.logger.info('initial state {0}'.format(initial_state))
        self.logger.info('initial state shape {0}'.format(len(initial_state)))

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
                [self.vocab_size + 1, self.config.cell_size], -1.0, 1.0
            ))
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # [batch_size, max_time, cell_size] = [64, ?, 128]
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state= initial_state)
        self.logger.info('inputs shape {0}'.format(inputs.shape)) # [batch, max_time, cell_size] = [64, ?, 128]
        self.logger.info('outputs shape {0}'.format(outputs.shape)) # [batch, max_time, cell_size] = [64, ?, 128]
        self.logger.info('last state {0}'.format(last_state)) # tuple of two LSTMStateTuple( shape=(batch, cell_size))
        self.logger.info('last state shape {0}'.format(len(last_state))) # 2
        output = tf.reshape(outputs, [-1, self.config.cell_size])
        self.logger.info('output shape {0}'.format(output.shape)) # [batch * max_time, cell_size] = [?, 128]

        weights = tf.Variable(tf.truncated_normal([self.config.cell_size, self.vocab_size + 1]))
        self.logger.info('weights shape {0}'.format(weights.shape)) # [cell_size, vocab_size] = [128, 6111]
        bias = tf.Variable(tf.zeros(shape=[self.vocab_size + 1]))
        self.logger.info('bias shape {0}'.format(bias.shape)) # [vocab_size] = [6111]
        logits = tf.nn.bias_add(tf.matmul(output, weights), bias = bias)
        self.logger.info('logits shape {0}'.format(logits.shape)) # [batch * max_time, vocab_size] = [?, 6111]

        # [?, vocab_size+1]
        labels = tf.one_hot(tf.reshape(self.output_data, [-1]), depth=self.vocab_size+1)
        # [batch * max_time, vocab_size] = [?, 6111 ]
        self.logger.info('labels shape {0}'.format(labels.shape))

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        self.logger.info('loss shape {0}'.format(loss.shape)) # [batch * max_time] = [?]
        prediction = tf.nn.softmax(logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(total_loss)
        self.endpoints['initial_state'] = initial_state
        self.endpoints['output'] = output
        self.endpoints['train_op'] = train_op
        self.endpoints['total_loss'] = total_loss
        self.endpoints['loss'] = loss
        self.endpoints['last_state'] = last_state
        self.endpoints['prediction'] = prediction

    def train(self, sess):
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
                iterations = len(self.poems_vector) // self.config.batch_size
                self.logger.info('total iterations: {0}'.format(iterations))
                for i in range(iterations):
                    start_index = i * self.config.batch_size
                    end_index = (i+1) * self.config.batch_size
                    batches = self.poems_vector[start_index:end_index]

                    # get the max length of poems in the batch
                    max_length = max(map(len, batches))

                    # fill in a matrix of shape [batch_size, max_length] with index of space
                    x_data = np.full((self.config.batch_size, max_length), self.word_to_index[' '], np.int32)

                    # fill in each poem to matrix
                    for row in range(self.config.batch_size):
                        x_data[row, :len(batches[row])] = batches[row]

                    # get label based on x
                    y_data = np.copy(x_data)
                    y_data[:, :-1] = x_data[:,1:]
                    loss, _, _ = sess.run([
                        self.endpoints['total_loss'],
                        self.endpoints['last_state'],
                        self.endpoints['train_op']
                    ], feed_dict = {self.input_data: x_data, self.output_data: y_data})
                    self.logger.info('Epoch: {0}, iteration: {1}, training loss: {2}'.format(epoch, i, loss))
                if epoch % 6 == 0:
                    saver.save(sess, self.config.model_dir, global_step= epoch)
        except KeyboardInterrupt:
            self.logger.error('Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, self.config.model_dir, global_step=epoch)
            self.logger.info('Last epoch were saved, next time will start from epoch {0}.'.format(epoch))

    def predict(self, sess, begin_word):
        self.config.batch_size = 1
        self._build()
        start_token = 'B'
        end_token = 'E'
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(self.config.checkpoints_dir)
        self.logger.info('restore from checkpoint {0}'.format(checkpoint))
        saver.restore(sess, checkpoint)

        x = np.array([list(map(self.word_to_index.get, start_token))])
        [predict, last_state] = sess.run([self.endpoints['prediction'], self.endpoints['last_state']],
                                         feed_dict={self.input_data: x})
        self.logger.info('predict shape {0}'.format(predict.shape))
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, self.vocabulary)
        poem = ''
        while word != end_token:
            poem += word
            x = np.zeros((1,1))
            x[0,0] = self.word_to_index[word]
            [predict, last_state] = sess.run([self.endpoints['prediction'], self.endpoints['last_state']],
                                             feed_dict={self.input_data: x, self.endpoints['initial_state']:last_state})
            word = to_word(predict, self.vocabulary)
        return poem

class LyricsRNN(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        pass

    def _build(self):
        pass

    def _train(self):
        pass

    def _evaluate(self):
        pass