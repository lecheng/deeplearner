# coding=utf-8
import io, os
import tensorflow.contrib.keras as kr
from collections import Counter

class THUCNews(object):
    def _read_category(self):
        categories = [u'体育',u'财经',u'房产',u'家居',u'教育',u'科技',u'时尚',u'时政',u'游戏',u'娱乐',u'社会',u'彩票',u'星座',u'股票']
        cat_to_id = dict(zip(categories,range(len(categories))))
        return categories, cat_to_id

    def _read_file(self, filename):
        """
        get label and content from file
        :param filename: (String) file name
        :return: (List of List of String)content list, (List of String)label list
        """
        contents = []
        labels = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                try:
                    label, content = line.strip().split('\t')
                    contents.append(list(content.decode('utf-8')))
                    labels.append(label.decode('utf-8'))
                except:
                    pass
        return contents, labels

    def _read_original_file(self, filename):
        """read content of a file and compressed into one line"""
        with io.open(filename, 'r') as f:
            return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

    def _read_vocab(self, filename):
        """
        :param filename: file name
        :return: words vocaburary and word to id dictionary
        """
        words = list(map(lambda line: line.strip(),
                         io.open(filename, 'r', encoding='utf-8').readlines()))
        word_to_id = dict(zip(words, range(len(words))))

        return words, word_to_id

    def build_vocab(self, filename, vocab_size=5000):
        """
        build words vocaburary 
        :param filename: 
        :param vocab_size: size of vocab
        :return: 
        """
        data, _ = self._read_file(filename)

        all_data = []
        for content in data:
            all_data.extend(content)

        counter = Counter(all_data)
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<PAD>'] + list(words)

        io.open('classification/cnews/vocab_cnews.txt', 'w').write('\n'.join(words))

    def save_file(self, dirname):
        """
        Merge all the files in thucnews folders into single file for train, validation and test
        """
        f_train = io.open('classification/cnews/cnews.train.txt', 'w', encoding='utf-8')
        f_test = io.open('classification/cnews/cnews.test.txt', 'w', encoding='utf-8')
        f_val = io.open('classification/cnews/cnews.val.txt', 'w', encoding='utf-8')
        for category in os.listdir(dirname):
            cat_dir = os.path.join(dirname, category)
            if not os.path.isdir(cat_dir):
                continue
            files = os.listdir(cat_dir)
            count = 0
            for cur_file in files:
                filename = os.path.join(cat_dir, cur_file)
                content = self._read_original_file(filename)
                if count < 5000:
                    f_train.write(category.decode('utf-8') + '\t' + content + '\n')
                elif count < 6000:
                    f_test.write(category.decode('utf-8') + '\t' + content + '\n')
                else:
                    f_val.write(category.decode('utf-8') + '\t' + content + '\n')
                count += 1

            print('Finished:', category)

        f_train.close()
        f_test.close()
        f_val.close()

    def to_words(self, content, words):
        """
        transfer id content into text
        :param content: id version of content
        :param words: id to word dictionary
        :return: text content
        """
        return ''.join(words[x] for x in content)

    def _file_to_ids(self, filename, word_to_id, max_length=100):
        """
        transfer word and label to id list
        :param filename: (String) file name
        :param word_to_id: (Dict) word to id dictionary
        :param max_length: (Int) max length of file
        :return: (List of List of Int)id list, (List of Int)label id list
        """
        _, cat_to_id = self._read_category()
        contents, labels = self._read_file(filename)

        data_id = []
        label_id = []
        for i in range(len(contents)):
            data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
            label_id.append(cat_to_id[labels[i]])

        # limited text into fixed length
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
        return x_pad, label_id

    def preocess_file(self, data_path='classification/cnews/', seq_length=100):
        """
        :param data_path: data file path
        :param seq_length: max length of file
        :return: 
        """
        words, word_to_id = self._read_vocab(os.path.join(data_path,
                                                     'vocab_cnews.txt'))
        x_train, y_train = self._file_to_ids(os.path.join(data_path,
                                                     'cnews.train.txt'), word_to_id, seq_length)
        x_test, y_test = self._file_to_ids(os.path.join(data_path,
                                                   'cnews.test.txt'), word_to_id, seq_length)
        x_val, y_val = self._file_to_ids(os.path.join(data_path,
                                                 'cnews.val.txt'), word_to_id, seq_length)
        return x_train, y_train, x_test, y_test, x_val, y_val, words

thucnews = THUCNews()
if __name__ == '__main__':
    thucnews = THUCNews()
    if not os.path.exists('classification/cnews/vocab_cnews.txt'):
        thucnews.build_vocab('classification/cnews/cnews.train.txt')
    x_train, y_train, x_test, y_test, x_val, y_val, words = thucnews.preocess_file()
    print(words)