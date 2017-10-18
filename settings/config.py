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

class TCCNNConfig(object):
    """
    configuration for text classification cnn model
    """
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