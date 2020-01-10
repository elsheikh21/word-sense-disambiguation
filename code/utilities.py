import json
import logging
import os
import pickle
import random
import warnings
import zipfile

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.backend import set_session
from tensorflow.keras.utils import get_file
from tqdm import tqdm


def initialize_logger():
    """
    Customize the logger, and fixes seed
    """
    np.random.seed(0)
    random.seed(0)
    tf.set_random_seed(0)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def configure_tf():
    warnings.filterwarnings('ignore', category=FutureWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # Allowing TF to automatically choose an existing and
    # supported device to run the operations in case the specified one doesn't exist
    config.allow_soft_placement = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)


def configure_workspace():
    """
    Initialize the logger and makes seed constant,
    read the configuration file and load its variables,
    Configure Tensorflow on the GPU or CPU based on config File
    :return: config_params {dict}
    """
    initialize_logger()

    # Load our config file
    config_file_path = os.path.join(os.getcwd(), "config.yaml")
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    # run on CPU, change in Config file.
    if config_params["USE_CPU"]:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        configure_tf()

    return config_params


def save_pickle(save_to, save_what):
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    with open(load_from, 'rb') as f:
        return pickle.load(f)


def save_json(save_to, save_what):
    with open(save_to, 'w+') as json_file:
        json.dump(save_what, json_file)


def freq_and_min(data_x, data_y):
    min_len = len(min(data_x, key=len))
    min_len_y = len(min(data_y, key=len))
    count_x, count_y = 0, 0
    for x, y in zip(data_x, data_y):
        if min_len == len(x):
            count_x += 1
        if min_len_y == len(y):
            count_y += 1
    return min_len, min_len_y, count_x, count_y


def freq_and_max(data_x, data_y):
    max_len = len(max(data_x, key=len))
    max_len_y = len(max(data_y, key=len))
    count_x, count_y = 0, 0
    for x, y in zip(data_x, data_y):
        if max_len == len(x):
            count_x += 1
        if max_len_y == len(y):
            count_y += 1
    return max_len, max_len_y, count_x, count_y


def dataset_summarize(dataset):
    train_x, train_y = dataset.get('train_x'), dataset.get('train_y')
    dev_x, dev_y = dataset.get('test_x'), dataset.get('test_y')
    tokenizer = dataset.get('tokenizer')
    del dataset

    ender = f'\n{"_____" * 10}\n'

    print('\nTraining Data Summary: \n')
    print(
        f'Length of train_x {len(train_x)}. \nLength of train_y {len(train_y)}.')

    min_len, min_len_y, count_x, count_y = freq_and_min(train_x, train_y)

    print(f'\nMin of train_x: {min_len}. \nMin of train_y: {min_len_y}.')
    print(
        f'Frequency of train_x with min_len: {count_x}. \nFrequency of train_y with min_len: {count_y}.')

    max_len, max_len_y, count_x, count_y = freq_and_max(train_x, train_y)

    print(f'\nMax of train_x: {max_len}. \nMax of train_y: {max_len_y}.')
    print(
        f'Frequency of train_x with max_len: {count_x}. \nFrequency of train_y with max_len: {count_y}.',
        end=ender)

    print('\nDevelopment Data Summary: \n')
    print(
        f'Length of dev_x {len(dev_x)}. \nLength of dev_y {len(dev_y)}.')

    min_len, min_len_y, count_x, count_y = freq_and_min(dev_x, dev_y)

    print(f'\nMin of dev_x: {min_len}. \nMin of dev_y: {min_len_y}.')
    print(
        f'Number of dev_x with min_len: {count_x}. \nNumber of dev_y with min_len: {count_y}.')

    max_len, max_len_y, count_x, count_y = freq_and_max(dev_x, dev_y)

    print(f'\nMax of dev_x: {max_len}. \nMax of dev_y: {max_len_y}.')
    print(
        f'Number of dev_x with max_len: {count_x}. \nNumber of dev_y with max_len: {count_y}.', end=ender)

    word_tokens = [
        word for word in tokenizer.word_index if not word.startswith('wn:')]
    sense_tokens = [
        word for word in tokenizer.word_index if word.startswith('wn:')]
    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    print(
        f'''\nNumber of word tokens: {len(word_tokens)}.
             \nNumber of sense tokens: {len(sense_tokens)}.
             \nTotal Size of vocab: {output_size}.''', end=ender)


def build_dict(file_name, save_to=None):
    """
    Builds and saves dictionary from text file
    This dictionary contains all the senses of all words
    """
    if save_to is not None and os.path.exists(save_to) and os.path.getsize(save_to) > 0:
        file_dict = load_pickle(save_to)
        logging.info("Dictionary is loaded")
    else:
        file_dict = dict()
        with open(file_name, mode='r') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc='Building dictionary'):
                synset_id, synset = line.split()[0], line.split()[1]
                file_dict[synset_id] = synset
        logging.info("Dictionary is built")
        if save_to is not None:
            save_pickle(save_to, file_dict)
        logging.info("Dictionary is saved")
    return file_dict


def download_unzip_dataset(download_from, download_to, clean_after=True):
    """Downloads the dataset and removes the zip file after unzipping it"""
    file_path = get_file(fname=download_to, origin=download_from)
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        for file in tqdm(iterable=zip_file.namelist(),
                         total=len(zip_file.namelist()),
                         desc=f'Unzipping {file_path}'):
            zip_file.extract(member=file)
    if clean_after:
        os.remove(file_path)


def build_bn2wn_dict(file_path, save_to=None):
    if save_to is None:
        save_to = [None, None]
    if save_to[0] is not None and os.path.exists(save_to[0]) and os.path.getsize(save_to[0]) > 0 and \
            save_to[1] is not None and os.path.exists(save_to[1]) and os.path.getsize(save_to[1]) > 0:
        babelnet_wordnet = load_pickle(save_to[0])
        wordnet_babelnet = load_pickle(save_to[1])
        logging.info("Dictionary is loaded")
        return babelnet_wordnet, wordnet_babelnet

    babelnet_wordnet = dict()
    with open(file_path, mode='r') as file:
        lines = file.read().splitlines()
        for line in tqdm(lines, desc='Building BN_WN mapping dict'):
            bn, wn = line.split('\t')
            babelnet_wordnet[bn] = wn

    wordnet_babelnet = dict([[v, k] for k, v in babelnet_wordnet.items()])

    if save_to[0] is not None and save_to[1] is not None:
        save_pickle(save_to[0], babelnet_wordnet)
        save_pickle(save_to[1], wordnet_babelnet)
    logging.info("Dictionary is saved")

    return babelnet_wordnet, wordnet_babelnet


def build_bn2lex_dict(file_path, save_to=None):
    if save_to is None:
        save_to = [None, None]
    if save_to[0] is not None and os.path.exists(save_to[0]) and os.path.getsize(save_to[0]) > 0 and \
            save_to[1] is not None and os.path.exists(save_to[1]) and os.path.getsize(save_to[1]) > 0:
        babelnet_lex = load_pickle(save_to[0])
        lex_babelnet = load_pickle(save_to[1])
        logging.info("Dictionary is loaded")
        return babelnet_lex, lex_babelnet

    babelnet_lex = dict()
    with open(file_path, mode='r') as file:
        lines = file.read().splitlines()
        for line in tqdm(lines, desc='Building bn_lex mapping dict'):
            bn, lex = line.split('\t')
            babelnet_lex[bn] = lex

    lex_babelnet = dict([[v, k] for k, v in babelnet_lex.items()])

    if save_to[0] is not None and save_to[1] is not None:
        save_pickle(save_to[0], babelnet_lex)
        save_pickle(save_to[1], lex_babelnet)
    logging.info("Dictionaries are saved")

    return babelnet_lex, lex_babelnet


def build_bn2dom_dict(file_path, save_to=None):
    if save_to is None:
        save_to = [None, None]
    if save_to[0] is not None and os.path.exists(save_to[0]) and os.path.getsize(save_to[0]) > 0 and \
            save_to[1] is not None and os.path.exists(save_to[1]) and os.path.getsize(save_to[1]) > 0:
        babelnet_dom = load_pickle(save_to[0])
        dom_babelnet = load_pickle(save_to[1])
        logging.info("Dictionary is loaded")
        return babelnet_dom, dom_babelnet

    babelnet_dom = dict()
    with open(file_path, mode='r') as file:
        lines = file.read().splitlines()
        for line in tqdm(lines, desc='Building bn_dom mapping dict'):
            split = line.split('\t')
            bn, dom = split[0], split[1]
            babelnet_dom[bn] = dom

    dom_babelnet = dict([[v, k] for k, v in babelnet_dom.items()])

    if save_to[0] is not None and save_to[1] is not None:
        save_pickle(save_to[0], babelnet_dom)
        save_pickle(save_to[1], dom_babelnet)
    logging.info("Dictionaries are saved")

    return babelnet_dom, dom_babelnet


def split_datasets(semcor_omsti):
    """
    Takes the SemCor+Omsti dataset and extract from it Omsti dataset,
    by reading all the file contents and then finding last </corpus>,
    and then including everything from the index of '<corpus lang="en" source="mun">'
    tag and then writing all of the list elements in between to another file
    in same directory

    :param semcor_omsti: str
    :return: None
    """
    with open(semcor_omsti, "r") as inputFile:
        file_content = inputFile.read().splitlines()
        first_line = file_content[0]
        corpus_endings = list(filter(lambda i: file_content[i] == "</corpus>", range(len(file_content))))

        omsti_corpus_idx = file_content.index('<corpus lang="en" source="mun">')
        omsti_corpus_end_idx = corpus_endings[1] + 1
        omsti_corpus = file_content[omsti_corpus_idx:omsti_corpus_end_idx]

    with open(semcor_omsti.replace('semcor+omsti', 'omsti'), 'w+') as omsti_file:
        omsti_file.write(f'{first_line}\n')
        for line in tqdm(omsti_corpus, desc='writing omsti file'):
            omsti_file.write(f'{line}\n')
    logging.info('DONE WITH OMSTI')


if __name__ == '__main__':
    semcor_omsti = os.path.join(os.getcwd(), 'data', 'training', 'WSD_Training_Corpora',
                                'SemCor+OMSTI', 'semcor+omsti.data.xml')
    split_datasets(semcor_omsti)

    file_path = os.path.join(os.getcwd(), 'resources', 'babelnet2wordnet.tsv')
    babelnet_wordnet, wordnet_babelnet = build_bn2wn_dict(file_path)

    file_path_ = os.path.join(os.getcwd(), 'resources', 'babelnet2lexnames.tsv')
    babelnet_lex, lex_babelnet = build_bn2lex_dict(file_path_)
