# coding=utf-8
import tensorflow as tf

from models.rnn import *
from utils.utils import *
from settings.config import *

def run_poem(is_train=True):
    model = PoemRNN(PoemRNNConfig, model_type='lstm')
    with tf.Session() as sess:
        if is_train:
            model.train(sess)
        else:
            input = raw_input('输入藏头诗首字：').decode('utf-8')
            poem = model.predict(sess,input)
            print_poem(poem)

if __name__ == '__main__':
    run_poem(is_train=True)