# Configuration classes for the projects
import os

class BasicConfig(object):
    """
    basic configuration for the project
    """
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../')
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
    MODEL_ROOT = os.path.join(PROJECT_ROOT, 'models')
    LOG_ROOT = os.path.join(PROJECT_ROOT, 'logs')
    SETTING_ROOT = os.path.join(PROJECT_ROOT, 'settings')
    UTIL_ROOT = os.path.join(PROJECT_ROOT, 'utils')
    TF_MODELS_ROOT = os.path.join(PROJECT_ROOT, 'tf_models')
    CHECKPOINTS_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_LEVLE = 2
    # LOG_LEVEL:
    # |-- 0 - Only record error messages
    # |-- 1 - Record errors and warnings
    # |-- 2 - All available messages are recorded

class TCCNNConfig(object):
    """
    configuration for text classification cnn model
    """
    learning_rate = 1e-3
    embedding_size = 128
    class_num = 14
    vocab_size = 7000
    epochs = 12
    filters_num = 256
    kernel_size = 5
    hidden_dim = 128
    text_length = 100
    batch_size = 128
    keep_prob = 0.7
    data_dir = os.path.join(BasicConfig.DATA_ROOT, 'classification/cnews')
    checkpoints_dir = os.path.join(BasicConfig.CHECKPOINTS_ROOT, 'classification')
    model_dir = os.path.join(checkpoints_dir, 'classification')

    pass

class TCRNNConfig(object):
    """
    configuration for text classification rnn model
    """
    pass

class PoemRNNConfig(BasicConfig):
    """
    configuration for poem rnn model
    """
    learning_rate = 1e-2
    cell_size = 128
    num_layers = 2
    batch_size = 64
    epochs = 12
    data_file = os.path.join(BasicConfig.DATA_ROOT, 'poems/poems.txt')
    checkpoints_dir = os.path.join(BasicConfig.CHECKPOINTS_ROOT, 'poems')
    model_dir = os.path.join(checkpoints_dir, 'poems')

class LyricsRNNConfig(object):
    """
    configuration for lyrics rnn model
    """
    pass

class MatrixFactorizationConfig(object):
    """
    configuration for matrix factorization model
    """
    learning_rate = 2e-2
    epochs = 1000
    max_users_num = 943
    max_items_num = 1682
    decay_rate = 1.0
    data_file = os.path.join(BasicConfig.DATA_ROOT, 'mf/u.data')
    checkpoints_dir = os.path.join(BasicConfig.CHECKPOINTS_ROOT, 'mf')
    model_dir = os.path.join(checkpoints_dir, 'mf')

class ProductToVectorConfig(object):
    """
    configuration for prod2vec
    """
    learning_rate = 2e-2
    steps = 100001
    max_users_num = 4372
    max_items_num = 3684
    batch_size = 128
    num_sampled = 64        # Number of negative examples to sample.
    embedding_size_p = 128    # Dimension of the product embedding vector.
    embedding_size_u = 128  # Dimension of the user embedding vector.
    skip_window = 1         # How many words to consider left and right.
    num_skips = 2           # How many times to reuse an input to generate a label.
    data_file = os.path.join(BasicConfig.DATA_ROOT, 'prod2vec/data.csv')
    checkpoints_dir = os.path.join(BasicConfig.CHECKPOINTS_ROOT, 'prod2vec')
    model_dir = os.path.join(checkpoints_dir, 'prod2vec')