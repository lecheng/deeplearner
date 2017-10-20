# coding=utf-8
import tensorflow as tf
import pandas as pd

from models.rnn import *
from models.mf import *
from utils.utils import *
from settings.config import *
import numpy as np

def run_poem(is_train=True):
    model = PoemRNN(PoemRNNConfig, model_type='lstm')
    with tf.Session() as sess:
        if is_train:
            model.train(sess)
        else:
            input = raw_input('输入藏头诗首字：').decode('utf-8')
            poem = model.predict(sess,input)
            print_poem(poem)

def run_mf():
    df = pd.read_csv(MatrixFactorizationConfig.data_file, sep='\t', names=['user','item','rate','time'])
    print('total length: {0}'.format(len(df)))
    msk = np.random.rand(len(df)) < 0.7
    df_train = df[msk]
    df_val = df[~msk]
    print('train length: {0}'.format(len(df_train)))
    users_indices_train = [x-1 for x in df_train.user.values]
    items_indices_train = [x-1 for x in df_train.item.values]
    rates_train = df_train.rate.values

    users_indices_val = [x - 1 for x in df_val.user.values]
    items_indices_val = [x - 1 for x in df_val.item.values]
    rates_val = df_train.rate.values

    model = MatrixFactorization(MatrixFactorizationConfig)
    with tf.Session() as sess:
        model.train(sess, users_indices_train, items_indices_train, rates_train)
        model.evaluate(sess, users_indices_val, items_indices_val, rates_val)
        predict = model.predict(sess, users_indices_val, items_indices_val)
        for i in range(len(users_indices_val)):
            print('User: {0}, Item: {1}, Original rate: {2}, Predicted rate: {3}'\
                  .format(users_indices_val[i], items_indices_val[i], rates_val[i], predict[i]))
if __name__ == '__main__':
    # run_poem(is_train=True)
    run_mf()