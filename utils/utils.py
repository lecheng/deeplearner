# coding=utf-8
import numpy as np

def print_poem(poem):
    poem_sentences = poem.split(u'。')
    for s in poem_sentences:
        if s!= '' and len(s) > 10:
            print(s + u'。')

def to_word(predict, vocabs):
    #print(predict.shape)
    #print(np.argmax(predict))
    t = np.cumsum(predict)
    #print(t.shape)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    #print(sample)
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]