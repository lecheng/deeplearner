import pandas as pd
import collections
import random
import numpy as np
import itertools

def build_dataset(product_lists, user_ids):
    """Process raw inputs into a dataset."""
    products = reduce(lambda a,b:a+b, product_lists)
    n_products = len(products)
    count = []
    count.extend(collections.Counter(products).most_common(n_products - 1))
    dict_p = dict()
    dict_u = dict()
    for u in user_ids:
        dict_u[u] = len(dict_u)
    for p, _ in count:
        dict_p[p] = len(dict_p)
    data_p = list()
    data_u = list()
    for p in products:
        data_p.append(dict_p[p])
    for i, product_list in enumerate(product_lists):
        data_u.extend([i] * len(product_list))
    reversed_dict_p = dict(zip(dict_p.values(), dict_p.keys()))
    reversed_dict_u = dict(zip(dict_u.values(), dict_u.keys()))
    return data_p, data_u, count, dict_p, reversed_dict_p, reversed_dict_u

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

def generate_batch(batch_size, num_skips, skip_window, data_p, data_u):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, 2), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer_p = collections.deque(maxlen=span)
    buffer_u = collections.deque(maxlen=span)
    if data_index + span > len(data_p):
        data_index = 0
    buffer_p.extend(data_p[data_index:data_index + span])
    buffer_u.extend(data_u[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            # print([buffer_p[skip_window], buffer_u[skip_window]])
            batch[i * num_skips + j,:] = [buffer_p[skip_window], buffer_u[skip_window]]
            labels[i * num_skips + j, 0] = buffer_p[target]
        if data_index == len(data_p):
            for i in range(0, span):
                buffer_p.append(data_p[i])
                buffer_u.append(data_u[i])
            data_index = span
        else:
            buffer_p.append(data_p[data_index])
            buffer_u.append(data_u[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data_p) - span) % len(data_p)
    return batch, labels

def set_data(path):
    file_path = path
    data = pd.read_csv(file_path, sep=',').dropna(axis=0, how='any', subset=['CustomerID'])
    user_ids = data['CustomerID'].unique()
    prod_ids = data['StockCode'].unique()
    user_num = len(user_ids)
    product_num = len(prod_ids)
    # print(user_num)
    # print(product_num)
    product_lists_by_user = np.array(dict(data.groupby('CustomerID')['StockCode'].apply(list)).values())
    product_to_desc = dict(data.groupby('StockCode')['Description'].apply(set))
    user_products = dict(data.groupby('CustomerID')['Description'].apply(list))

    data_p, data_u, _, _, reverse_dict_p, reverse_dict_u = build_dataset(product_lists_by_user, user_ids)
    del product_lists_by_user # Hint to reduce memory.
    return data_p, data_u, product_to_desc, reverse_dict_p, reverse_dict_u, user_products

if __name__ == '__main__':
    data_p, data_u, _, reverse_dict_p, reverse_dict_u, _ = set_data('prod2vec/data.csv')
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data_p=data_p, data_u=data_u)
    for i in range(8):
        print('product: ', batch[i][0], reverse_dict_p[batch[i][0]], 'user: ', batch[i][1], reverse_dict_u[batch[i][1]],
            '->', labels[i, 0], reverse_dict_p[labels[i, 0]])