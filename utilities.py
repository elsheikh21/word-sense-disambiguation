import json
import logging
import os
import pickle
import random
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.preprocessing.text import Tokenizer
import zipfile
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
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)


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
    dev_x, dev_y = dataset.get('dev_x'), dataset.get('dev_y')
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


def load_raw_data(save_to_paths):
    sentences_list, labeled_sentences_list, masks_builder = [], [], []
    if (save_to_paths is not None and
            os.path.exists(save_to_paths[0]) and
            os.path.getsize(save_to_paths[0]) > 0 and
            os.path.exists(save_to_paths[1]) and
            os.path.getsize(save_to_paths[1]) > 0 and
            os.path.exists(save_to_paths[2]) and
            os.path.getsize(save_to_paths[2]) > 0):
        sentences_list = load_pickle(save_to_paths[0])
        labeled_sentences_list = load_pickle(save_to_paths[1])
        masks_builder = load_pickle(save_to_paths[2])
        logging.info("Parsed Dataset is loaded")
    return sentences_list, labeled_sentences_list, masks_builder


def save_raw_data(save_to_paths, sentences_list, labeled_sentences_list, masks_builder):
    if save_to_paths is not None:
        save_x_to, save_y_to = save_to_paths[0], save_to_paths[1]
        save_mask_to = save_to_paths[2]
        save_pickle(save_x_to, sentences_list)
        save_pickle(save_y_to, labeled_sentences_list)
        save_pickle(save_mask_to[2], masks_builder)
        logging.info("Saved the dataset")


def load_processed_data(save_data, save_tokenizer):
    data_x, data_y, tokenizer = None, None, None
    if (save_data[0] is not None
            and os.path.exists(save_data[0])
            and os.path.getsize(save_data[0]) > 0):
        data_x = load_pickle(save_data[0])
        logging.info("data_x is loaded")
    if (save_data[1] is not None
            and os.path.exists(save_data[1])
            and os.path.getsize(save_data[1]) > 0):
        data_y = load_pickle(save_data[1])
        logging.info("data_y is loaded")

    if (save_tokenizer is not None
            and os.path.exists(save_tokenizer)
            and os.path.getsize(save_tokenizer) > 0):
        tokenizer = load_pickle(save_tokenizer)
        logging.info("Tokenizer is loaded")
    else:
        filters = '!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\'\t'
        tokenizer = Tokenizer(filters=filters, oov_token='<OOV>', lower=True)

    return data_x, data_y, tokenizer


def save_processed_data(tokenizer, save_tokenizer,
                        data_x, data_y, save_data):
    if save_tokenizer is not None:
        save_pickle(save_tokenizer, tokenizer)
        logging.info("Tokenizer Saved")

    if save_data is not None:
        save_pickle(save_data[0], data_x)
        save_pickle(save_data[1], data_y)
        logging.info("Processed Data is Saved")

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

