import logging
import os

import numpy as np
import yaml
from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from utilities import (initialize_logger, load_pickle, save_pickle,
                       dataset_summarize,
                       build_dict, download_unzip_dataset)


def parse_dataset(file_name, gold_dict, config_path, save_to_paths=None):
    """
    Starts with reading xml file, only sentence tags, iterates over children
    of sentences' tags.
    If it word-format(wf) or instance we add lemma,
    if instance we add it in form of lemma_synsetID.

    :param save_to_paths:
    :param file_name: string points to path of xml file
    :param gold_dict: dict contains all senses as per xml file
    :return sentences: list of strings contains all data unlabeled
    :return sentences_labeled: list of strings contains all data labeled
    """
    # TODO: ADD POS & LEX Sentences

    sentences_list, labeled_sentences_list, masks_builder = [], [], []
    config_file = open(config_path)
    config_params = yaml.load(config_file)
    batch_size = config_params['batch_size']

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
    else:
        # read file contents in terms of sentences
        context = iterparse(file_name, tag="sentence")
        # iterating over the sentences
        for _, elements in tqdm(context, desc="Parsing corpus"):
            sentence, sentence_labeled = [], []
            mask_builder = []
            for elem in list(elements.iter()):
                if elem is not None:
                    if ((elem.tag == 'wf' or elem.tag == 'instance') and
                            elem.text is not None):
                        elem_lemma = elem.attrib['lemma']
                        sentence.append(elem_lemma)
                        sentence_labeled.append(elem_lemma)
                        if elem.tag == 'wf':
                            mask_builder.append([elem_lemma])
                    if elem.tag == 'instance' and elem.text is not None:
                        elem_id = elem.attrib['id']
                        elem_lemma = elem.attrib['lemma']
                        sense_key = str(gold_dict.get(elem_id))
                        if sense_key is not None:
                            synset = wn.lemma_from_key(sense_key).synset()
                            synset_id = f"wn:{str(synset.offset()).zfill(8)}{synset.pos()}"
                            sentence_labeled[-1] = f'{synset_id}'
                        if elem_lemma:
                            mask_builder.append([elem_lemma, synset_id])
            # if the sentence is not empty
            if len(sentence) and len(sentence_labeled) and len(mask_builder):
                sentences_list.append(sentence)
                labeled_sentences_list.append(sentence_labeled)
                masks_builder.append(mask_builder)
            elements.clear()
        logging.info("Parsed the dataset")

        while len(sentences_list) % batch_size != 0:
            sentences_list.append(sentences_list[0])
            labeled_sentences_list.append(labeled_sentences_list[0])
            masks_builder.append(masks_builder[0])


    if save_to_paths is not None:
        save_x_to, save_y_to = save_to_paths[0], save_to_paths[1]
        save_mask_to = save_to_paths[2]
        save_pickle(save_x_to, sentences_list)
        save_pickle(save_y_to, labeled_sentences_list)
        save_pickle(save_mask_to, masks_builder)
        logging.info("Saved the dataset")

    return sentences_list, labeled_sentences_list, masks_builder


def process_dataset(data_x, data_y, save_tokenizer=None,
                    save_data=None, elmo=False):
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
        tokenizer = Tokenizer(oov_token='<OOV>')
        tokenizer.fit_on_texts(data_x)
        tokenizer.fit_on_texts(data_y)
        tokenizer.word_index.update({'<PAD>': 0})
        tokenizer.index_word.update({0: '<PAD>'})

        if save_tokenizer is not None:
            save_pickle(save_tokenizer, tokenizer)
            logging.info("Tokenizer Saved")

        data_x = data_x if elmo else tokenizer.texts_to_sequences(data_x)
        data_y = tokenizer.texts_to_sequences(data_y)

        if save_data is not None:
            save_pickle(save_data[0], data_x)
            save_pickle(save_data[1], data_y)
            logging.info("Processed Data is Saved")

    return data_x, data_y


def load_dataset(summarize=False, elmo=False):
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    resources_path = os.path.join(cwd, 'resources')
    config_path = os.path.join(cwd, 'config.yaml')

    try:
        # Building the gold dictionary for training set
        file_path = os.path.join(
            data_path, 'training', 'WSD_Training_Corpora',
            'SemCor', 'semcor.gold.key.txt')
        save_to = os.path.join(resources_path, 'gold_dict.pkl')
        gold_dict = build_dict(file_path, save_to)

        # parsing the dataset & save it
        path = os.path.join(data_path, 'training',
                            'WSD_Training_Corpora',
                            'SemCor', 'semcor.data.xml')
        save_data = [os.path.join(resources_path, 'train_x.pkl'),
                     os.path.join(resources_path, 'train_y.pkl'),
                     os.path.join(resources_path, 'train_mask.pkl')]
        (data_x, data_y, mask_builder) = parse_dataset(
            path, gold_dict, config_path, save_to_paths=save_data)
    except FileNotFoundError:
        download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip'
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)

    save_tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
    train_x, train_y = process_dataset(data_x, data_y,
                                       save_tokenizer=save_tokenizer,
                                       save_data=save_data, elmo=elmo)

    try:
        # Building the gold dictionary for dev set
        eval_path = os.path.join(
            data_path, 'evaluation',
            'WSD_Unified_Evaluation_Datasets', 'ALL',
            'ALL.data.xml')

        eval_gold = os.path.join(
            data_path, 'evaluation',
            'WSD_Unified_Evaluation_Datasets', 'ALL',
            'ALL.gold.key.txt')

        # Parsing the gold dict
        save_eval_to = os.path.join(resources_path, 'eval_dict.pkl')
        eval_dict = build_dict(eval_gold, save_eval_to)

        # Parsing the dev dataset
        save_data = [os.path.join(resources_path, 'test_x.pkl'),
                     os.path.join(resources_path, 'test_y.pkl'),
                     os.path.join(resources_path, 'test_mask.pkl')]
        (data_x, data_y, _) = parse_dataset(eval_path, eval_dict,
                                            config_path,
                                            save_to_paths=save_data)
    except FileNotFoundError:
        download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip'
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)

    test_x, test_y = process_dataset(data_x, data_y,
                                     save_tokenizer=save_tokenizer,
                                     save_data=save_data, elmo=elmo)
    tokenizer = load_pickle(save_tokenizer)
    word_tokens = [
        word for word in tokenizer.word_index if not word.startswith('wn:')]
    sense_tokens = [
        word for word in tokenizer.word_index if word.startswith('wn:')]

    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    dataset = {
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'tokenizer': load_pickle(save_tokenizer),
        'vocabulary_size': vocabulary_size,
        'output_size': output_size,
        'mask_builder': mask_builder
    }

    if summarize:
        dataset_summarize(dataset)

    return dataset


def create_mask(mask_builder, mask_shape, tokenizer, output_size):
    mask_x = np.full(shape=(mask_shape[0],
                            mask_shape[1],
                            output_size),
                     fill_value=-np.inf)

    # Get the word2idx dictionary from tokenizer class
    word2idx = tokenizer.word_index
    # for every sentence in the mask builder
    for sentence in range(len(mask_builder)):
        # for every word in every sentence
        for word_arr in range(len(mask_builder[sentence])):
            # get the word from the tuple passed (lemma)
            word = mask_builder[sentence][word_arr][0]
            # get its index, and set it to 0.0
            word_idx = word2idx.get(word)
            mask_x[sentence][word_arr][word_idx] = 0.0
            # if the word is of instance type, get candidate synsets' indices
            # and set them to zero.
            if len(mask_builder[sentence][word_arr]) == 2:
                candidate_synsets_idx = get_candidate_senses(word, word2idx)
                for idx in candidate_synsets_idx:
                    mask_x[sentence][word_arr][idx] = 0.0

    return mask_x


def val_generator(data_x, data_y, batch_size, output_size,
                  use_elmo):
    start = 0
    while True:
        end = start + batch_size
        data_x_, data_y_ = data_x[start:end], data_y[start:end]
        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(np.array(data_x_), padding='post',
                                value=pad_val, maxlen=max_len, dtype=object).tolist()

        data_y_ = pad_sequences(np.array(data_y_), padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)


        _data_x = data_x_ if not use_elmo else np.array([' '.join(x) for x in data_x_],
                                                        dtype=object)


        yield _data_x, _data_y

        if start + batch_size > len(data_x):
            start = 0
        else:
            start += batch_size


def train_generator(data_x, data_y, batch_size, output_size,
                    use_elmo, mask_builder, tokenizer, shuffle=False):
    start = 0
    while True:
        end = start + batch_size
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        mask_builder = np.array(mask_builder)
        if shuffle:
            permutation = np.random.permutation(len(data_x))
            data_x_, data_y_ = data_x[permutation[start:end]], data_y[permutation[start:end]]
            mask_builder_ = mask_builder[permutation[start:end]]

        else:
            data_x_, data_y_ = data_x[start:end], data_y[start:end]
            mask_builder_ = mask_builder[start:end]

        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(data_x_, padding='post',
                                value=pad_val, maxlen=max_len, dtype=object).tolist()

        data_y_ = pad_sequences(data_y_, padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        mask_x = create_mask(mask_builder_,
                             [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size)

        if use_elmo:
            _data_x = np.array([' '.join(x) for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = data_x_

        mask_x = pad_sequences(np.array(mask_x), padding='post', value=0, maxlen=max_len)

        yield [_data_x, mask_x], _data_y

        if start + batch_size >= len(data_x):
            start = 0
            if shuffle:
                permutation = np.random.permutation(len(data_x))
        else:
            start += batch_size


def get_candidate_senses(word, word2idx):
    candidates = [f'wn:{str(synset.offset()).zfill(8)}{synset.pos()}'
                  for synset in wn.synsets(word)]

    return [word2idx.get(candidate, None) for candidate in candidates
            if word2idx.get(candidate, None)]


if __name__ == "__main__":
    initialize_logger()
    dataset = load_dataset(summarize=True, elmo=False)
