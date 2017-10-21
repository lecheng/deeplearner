import pandas as pd
import collections
import random
import numpy as np
import itertools

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

def build_dataset_by_df(file_path):
    """Process file inputs into a dataset."""
    data = pd.read_csv(file_path, sep=',')
    print(data.head())

    user_ids = data['CustomerID'].unique()
    prod_ids = data['StockCode'].unique()
    user_num = len(user_ids)
    product_num = len(prod_ids)
    product_lists_by_user = dict(data.groupby('CustomerID')['StockCode'].apply(list)).values()
    product_to_desc = dict(data.groupby('StockCode')['Description'].apply(set))
    dictionary = {}
    for id in prod_ids:
        dictionary[id] = len(dictionary)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return product_lists_by_user, product_to_desc, dictionary, reversed_dictionary, user_num, product_num

def get_data(product_lists, skip_window):
    '''
    :param dict_: (np.array([[]]))
    :return: list of input pairs
    '''
    data = []
    for product_list in product_lists:
        pass


# data_row = 0
# data_col = 0
data_index = 0

def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def set_data(path):
    file_path = path
    data = pd.read_csv(file_path, sep=',')
    user_ids = data['CustomerID'].unique()
    prod_ids = data['StockCode'].unique()
    user_num = len(user_ids)
    product_num = len(prod_ids)
    product_lists_by_user = np.array(dict(data.groupby('CustomerID')['StockCode'].apply(list)).values())
    product_to_desc = dict(data.groupby('StockCode')['Description'].apply(set))

    products = list(itertools.chain(*product_lists_by_user))
    data, count, dictionary, reverse_dictionary = build_dataset(products,
                                                                len(products))
    del products # Hint to reduce memory.
    return data, product_to_desc, reverse_dictionary

if __name__ == '__main__':
    data, _, reverse_dictionary = set_data('prod2vec/data.csv')
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
            '->', labels[i, 0], reverse_dictionary[labels[i, 0]])