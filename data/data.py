# -*- coding: utf-8 -*-
import pandas
from collections import Counter

def process_poem(file_path):
    '''
    :param file_path: path of data file
    :return: 
    '''
    poems = []
    start_token = 'B'
    end_token = 'E'
    with open(file_path, 'r') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 15 or len(content) > 79*3:
                    continue

                content = start_token + content + end_token
                poems.append(content.decode('utf-8'))
            except ValueError as e:
                print(e)
        f.close()

    # sort poems by length
    poems = sorted(poems, key=lambda l: len(line))

    all_words = []
    for poem in poems:
        all_words += [word for word in poem]

    # count each character in poems
    counter = Counter(all_words)
    print(len(counter))
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    # count_pairs = [('a':1),('b':2)]
    # zip(*count_pairs) = [('a','b'),(1,2)]

    # add space to vocabulary
    words = words + (' ',)

    # map word to index
    word_to_index = dict(zip(words, range(len(words))))

    # get poems vector (each word in poems was indicated as index)
    poems_vector = [list(map(lambda word: word_to_index.get(word, len(words)), poem)) for poem in poems]
    print(len(poems_vector))
    return poems_vector, word_to_index, words



if __name__ == '__main__':
    process_poem('poems/poems.txt')