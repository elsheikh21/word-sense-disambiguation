import logging
import os

import re

import numpy as np
import yaml
from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from utilities import (initialize_logger, load_pickle, save_pickle,
                       dataset_summarize, build_bn2wn_dict,
                       build_bn2lex_dict, build_dict,
                       download_unzip_dataset)


def parse_dataset(file_name, gold_dict, config_path, wordnet_babelnet, babelnet_lex, save_to_paths=None):
    """
    Starts with reading xml file, only sentence tags, iterates over children
    of sentences' tags.
    If it word-format(wf) or instance we add lemma,
    if instance we add it in form of lemma_synsetID.

    :param file_name: string points to path of xml file
    :param gold_dict: dict contains all senses as per xml file
    :param config_path:
    :param wordnet_babelnet:
    :param babelnet_lex:
    :param save_to_paths:
    :return sentences: list of strings contains all data unlabeled
    :return sentences_labeled: list of strings contains all data labeled
    """
    sentences_list, labeled_sentences_list, masks_builder = [], [], []
    pos_labeled_list, lex_labeled_list = [], []

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
        if 'train' in save_to_paths[1]:
            pos_labeled_list = load_pickle(save_to_paths[1].replace('train_y', 'train_pos_y'))
            lex_labeled_list = load_pickle(save_to_paths[1].replace('train_y', 'train_lex_y'))
        else:
            pos_labeled_list = load_pickle(save_to_paths[1].replace('test_y', 'test_pos_y'))
            lex_labeled_list = load_pickle(save_to_paths[1].replace('test_y', 'test_lex_y'))
        logging.info("Parsed Dataset is loaded")
    else:
        # read file contents in terms of sentences
        context = iterparse(file_name, tag="sentence")
        # iterating over the sentences
        for _, elements in tqdm(context, desc="Parsing corpus"):
            sentence, sentence_labeled = [], []
            mask_builder = []
            sentence_pos_labeled, sentence_lex_labeled = [], []
            for elem in list(elements.iter()):
                if elem is not None:
                    if ((elem.tag == 'wf' or elem.tag == 'instance') and
                            elem.text is not None):
                        elem_lemma = elem.attrib['lemma']
                        elem_pos = elem.attrib['pos']
                        sentence.append(elem_lemma)
                        sentence_labeled.append(elem_lemma)
                        sentence_pos_labeled.append(elem_pos)
                        sentence_lex_labeled.append(elem_lemma)
                        if elem.tag == 'wf':
                            mask_builder.append([elem_lemma])
                    if elem.tag == 'instance' and elem.text is not None:
                        elem_id, elem_lemma = elem.attrib['id'], elem.attrib['lemma']
                        sense_key, synset_id = str(gold_dict.get(elem_id)), None
                        if sense_key is not None:
                            synset = wn.lemma_from_key(sense_key).synset()
                            synset_id = f"wn:{str(synset.offset()).zfill(8)}{synset.pos()}"
                            sentence_labeled[-1] = f'{synset_id}'
                            if wordnet_babelnet.get(synset_id) is None:
                                sentence_lex_labeled[-1] = 'factotum'
                            else:
                                sentence_lex_labeled[-1] = str(babelnet_lex[wordnet_babelnet[synset_id]])
                        if elem_lemma:
                            mask_builder.append([elem_lemma, synset_id])
            if len(sentence) and len(sentence_labeled) and len(mask_builder)\
                    and len(sentence_pos_labeled)\
                    and len(sentence_lex_labeled):
                # if the sentence is not empty
                sentences_list.append(sentence)
                labeled_sentences_list.append(sentence_labeled)
                masks_builder.append(mask_builder)
                pos_labeled_list.append(sentence_pos_labeled)
                lex_labeled_list.append(sentence_lex_labeled)
            elements.clear()
        logging.info("Parsed the dataset")

        '''
        # TODO: WE CAN DISCARD SENTENCES OF 1 WORD LEN, BUT I NEED TO KNOW 
         HOW LOW WORDS COUNT/SENTENCE IS ENOUGH TO GRASB THE CONCEPT 
        '''
        while len(sentences_list) % batch_size != 0:
            sentences_list.append(sentences_list[0])
            labeled_sentences_list.append(labeled_sentences_list[0])
            masks_builder.append(masks_builder[0])
            pos_labeled_list.append(pos_labeled_list[0])
            lex_labeled_list.append(lex_labeled_list[0])

    if save_to_paths is not None:
        save_x_to, save_y_to = save_to_paths[0], save_to_paths[1]
        save_mask_to = save_to_paths[2]
        if 'train' in save_to_paths[1]:
            save_pos_to = save_to_paths[1].replace('train_y', 'train_pos_y')
            save_lex_to = save_to_paths[1].replace('train_y', 'train_lex_y')
        else:
            save_pos_to = save_to_paths[1].replace('test_y', 'test_pos_y')
            save_lex_to = save_to_paths[1].replace('test_y', 'test_lex_y')

        save_pickle(save_x_to, sentences_list)
        save_pickle(save_y_to, labeled_sentences_list)
        save_pickle(save_pos_to, pos_labeled_list)
        save_pickle(save_lex_to, lex_labeled_list)
        save_pickle(save_mask_to, masks_builder)
        logging.info("Saved the dataset")

    return sentences_list, labeled_sentences_list, masks_builder, pos_labeled_list, lex_labeled_list


def process_dataset(data_x, data_y, pos_labeled_list,
                    lex_labeled_list, save_tokenizer=None,
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
        if 'train' in save_data[1]:
            pos_labeled_list = load_pickle(save_data[1].replace('train_y', 'train_pos_y'))
            lex_labeled_list = load_pickle(save_data[1].replace('train_y', 'train_lex_y'))
        else:
            pos_labeled_list = load_pickle(save_data[1].replace('test_y', 'test_pos_y'))
            lex_labeled_list = load_pickle(save_data[1].replace('test_y', 'test_lex_y'))
        logging.info("data_y is loaded")

    if (save_tokenizer is not None
            and os.path.exists(save_tokenizer)
            and os.path.getsize(save_tokenizer) > 0):
        tokenizer = load_pickle(save_tokenizer)
        pos_tokenizer = load_pickle(save_tokenizer.replace('tokenizer', 'pos_tokenizer'))
        lex_tokenizer = load_pickle(save_tokenizer.replace('tokenizer', 'lex_tokenizer'))
        try:
            # If text already converted to sequences then skip this part
            data_x = data_x if elmo else tokenizer.texts_to_sequences(data_x)
            data_y = tokenizer.texts_to_sequences(data_y)
            pos_labeled_list = pos_tokenizer.texts_to_sequences(pos_labeled_list)
            lex_labeled_list = lex_tokenizer.texts_to_sequences(lex_labeled_list)
        except AttributeError:
            pass
        logging.info("Tokenizers are loaded")
    else:
        tokenizer = Tokenizer(oov_token='<OOV>')
        tokenizer.fit_on_texts(data_x)
        tokenizer.fit_on_texts(data_y)
        tokenizer.word_index.update({'<PAD>': 0})
        tokenizer.index_word.update({0: '<PAD>'})

        pos_tokenizer = Tokenizer(oov_token='<OOV>')
        pos_tokenizer.fit_on_texts(pos_labeled_list)
        pos_tokenizer.word_index.update({'<PAD>': 0})
        pos_tokenizer.index_word.update({0: '<PAD>'})

        lex_tokenizer = Tokenizer(oov_token='factotum')
        lex_tokenizer.fit_on_texts(lex_labeled_list)
        lex_tokenizer.word_index.update({'<PAD>': 0})
        lex_tokenizer.index_word.update({0: '<PAD>'})

        if save_tokenizer is not None:
            save_pickle(save_tokenizer, tokenizer)
            save_pickle(save_tokenizer.replace('tokenizer', 'pos_tokenizer'), pos_tokenizer)
            save_pickle(save_tokenizer.replace('tokenizer', 'lex_tokenizer'), lex_tokenizer)
            logging.info("Tokenizers are Saved")

        data_x = data_x if elmo else tokenizer.texts_to_sequences(data_x)
        data_y = tokenizer.texts_to_sequences(data_y)
        pos_labeled_list = pos_tokenizer.texts_to_sequences(pos_labeled_list)
        lex_labeled_list = lex_tokenizer.texts_to_sequences(lex_labeled_list)

        if save_data is not None:
            save_pickle(save_data[0], data_x)
            save_pickle(save_data[1], data_y)
            if 'train' in save_data[1]:
                save_pickle(save_data[1].replace('train_y', 'train_pos_y'), pos_labeled_list)
                save_pickle(save_data[1].replace('train_y', 'train_lex_y'), lex_labeled_list)
            else:
                save_pickle(save_data[1].replace('test_y', 'test_pos_y'), pos_labeled_list)
                save_pickle(save_data[1].replace('test_y', 'test_lex_y'), lex_labeled_list)
            logging.info("Processed Data is Saved")

    return data_x, data_y, pos_labeled_list, lex_labeled_list


def load_dataset(summarize=False, elmo=False):
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    resources_path = os.path.join(cwd, 'resources')
    config_path = os.path.join(cwd, 'config.yaml')
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    bn2lex_path = os.path.join(resources_path, 'babelnet2lexnames.tsv')

    bn2wn_save_path = bn2wn_path.replace('tsv', 'pkl')
    wn2bn_save_path = bn2wn_path.replace('babelnet2wordnet.tsv', 'wordnet2babelnet.pkl')
    _, wordnet_babelnet = build_bn2wn_dict(bn2wn_path, save_to=[bn2wn_save_path, wn2bn_save_path])

    bn2lex_save_path = bn2lex_path.replace('tsv', 'pkl')
    lex2bn_save_path = bn2lex_path.replace('babelnet2lexnames.tsv', 'lexnames2babelnet.pkl')
    babelnet_lex, _ = build_bn2lex_dict(bn2lex_path, save_to=[bn2lex_save_path, lex2bn_save_path])

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
        (data_x, data_y, mask_builder,
         pos_labeled_list_, lex_labeled_list_) = parse_dataset(
            path, gold_dict, config_path, wordnet_babelnet,
            babelnet_lex, save_to_paths=save_data)
    except FileNotFoundError:
        download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip'
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)

    save_tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
    (train_x, train_y,
     pos_labeled_list,
     lex_labeled_list) = process_dataset(data_x, data_y,
                                         pos_labeled_list_, lex_labeled_list_,
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
        (data_x, data_y, _,
         dev_pos_labeled_list, dev_lex_labeled_list) = parse_dataset(eval_path, eval_dict,
                                                                     config_path, wordnet_babelnet,
                                                                     babelnet_lex, save_to_paths=save_data)
    except FileNotFoundError:
        download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip'
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)

    (test_x, test_y,
     pos_labeled_list_test,
     lex_labeled_list_test) = process_dataset(data_x, data_y,
                                              dev_pos_labeled_list, dev_lex_labeled_list,
                                              save_tokenizer=save_tokenizer,
                                              save_data=save_data, elmo=elmo)
    tokenizer = load_pickle(save_tokenizer)
    word_tokens = [
        word for word in tokenizer.word_index if not word.startswith('wn:')]
    sense_tokens = [
        word for word in tokenizer.word_index if word.startswith('wn:')]

    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    pos_tokenizer = load_pickle(save_tokenizer.replace('tokenizer', 'pos_tokenizer'))
    pos_vocab_size = len(pos_tokenizer.word_index.items())

    lex_tokenizer = load_pickle(save_tokenizer.replace('tokenizer', 'lex_tokenizer'))
    lex_vocab_size = len(lex_tokenizer.word_index.items())
    # lex_word_regex = '[a-z]+(?:\\.)[a-z]+$'
    # lex_names = [word for word in lex_tokenizer.word_index.keys() if re.match(lex_word_regex, word) and len(word) > 3]
    # lex_names.extend(['<OOV>', '<PAD>', 'factotum'])
    # lex_vocab_size = len(lex_names)

    dataset = {
        'train_x': train_x,
        'train_y': train_y,
        'pos_labeled_list': pos_labeled_list,
        'lex_labeled_list': lex_labeled_list,
        'test_x': test_x,
        'test_y': test_y,
        'pos_labeled_list_test': pos_labeled_list_test,
        'lex_labeled_list_test': lex_labeled_list_test,
        'tokenizer': load_pickle(save_tokenizer),
        'vocabulary_size': vocabulary_size,
        'output_size': output_size,
        'mask_builder': mask_builder,
        'pos_vocab_size': pos_vocab_size,
        'lex_vocab_size': lex_vocab_size
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

# TODO: COUPLE OF FIXES
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


def multitask_train_generator(data_x, data_y, pos_y, lex_y,
                              batch_size, output_size, use_elmo,
                              mask_builder, tokenizer, shuffle=False):
    start = 0
    while True:
        end = start + batch_size
        data_x, data_y = np.array(data_x), np.array(data_y)
        pos_y, lex_y = np.array(pos_y), np.array(lex_y)
        mask_builder = np.array(mask_builder)

        if shuffle:
            permutation = np.random.permutation(len(data_x))
            data_x_, data_y_ = data_x[permutation[start:end]], data_y[permutation[start:end]]
            pos_y_, lex_y_ = pos_y[permutation[start:end]], lex_y[permutation[start:end]]
            mask_builder_ = mask_builder[permutation[start:end]]

        else:
            data_x_, data_y_ = data_x[start:end], data_y[start:end]
            pos_y_, lex_y_ = pos_y[start:end], lex_y[start:end]
            mask_builder_ = mask_builder[start:end]

        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(data_x_, padding='post',
                                value=pad_val, maxlen=max_len, dtype=object).tolist()

        data_y_ = pad_sequences(data_y_, padding='post', value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        pos_y_ = pad_sequences(pos_y_, padding='post', value=0, maxlen=max_len, dtype=object)
        _pos_y = np.expand_dims(pos_y_, axis=-1)

        lex_y_ = pad_sequences(lex_y_, padding='post', value=0, maxlen=max_len, dtype=object)
        _lex_y = np.expand_dims(lex_y_, axis=-1)

        mask_x = create_mask(mask_builder_,
                             [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size)

        if use_elmo:
            _data_x = np.array([' '.join(x) for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = data_x_

        mask_x = pad_sequences(np.array(mask_x), padding='post', value=0, maxlen=max_len)

        yield [_data_x, mask_x], [_data_y, _pos_y, _lex_y]

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
