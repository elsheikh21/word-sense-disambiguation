import logging
import os

import numpy as np
from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from utilities import (initialize_logger, load_pickle, save_pickle,
                       freq_and_min, freq_and_max, dataset_summarize,
                       load_raw_data, save_raw_data, save_processed_data,
                       load_processed_data, build_dict, download_unzip_dataset)


def parse_dataset(file_name, gold_dict, save_to_paths=None):
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
    sentences_list, labeled_sentences_list, masks_builder = load_raw_data(save_to_paths)

    filters = '!"#$%&()*+,-./;<=>?@[\\]^_`{|}~\'\t'
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
                        _elem_lemma = elem_lemma.translate(
                            str.maketrans('', '', filters))
                        if _elem_lemma:
                            mask_builder.append([_elem_lemma])
                if elem.tag == 'instance' and elem.text is not None:
                    elem_id = elem.attrib['id']
                    elem_lemma = elem.attrib['lemma']
                    _elem_lemma = elem_lemma.translate(
                        str.maketrans('', '', filters))
                    sense_key = str(gold_dict.get(elem_id))
                    if sense_key is not None:
                        synset = wn.lemma_from_key(sense_key).synset()
                        synset_id = f"wn:{str(synset.offset()).zfill(8)}{synset.pos()}"
                        sentence_labeled[-1] = f'{synset_id}'
                    if _elem_lemma:
                        mask_builder.append([_elem_lemma, synset_id])
        # if the sentence is not empty
        if len(sentence) and len(sentence_labeled) and len(mask_builder):
            sentences_list.append(sentence)
            labeled_sentences_list.append(sentence_labeled)
            masks_builder.append(mask_builder)
        elements.clear()
    logging.info("Parsed the dataset")

    save_raw_data(save_to_paths, sentences_list, labeled_sentences_list, masks_builder)

    return sentences_list, labeled_sentences_list, masks_builder


def process_dataset(data_x, data_y, save_tokenizer=None,
                    save_data=None, elmo=False):
    tokenizer, data_x, data_y = load_processed_data(save_data, save_tokenizer)

    tokenized_data_x = tokenizer.fit_on_texts(data_x)
    tokenized_data_y = tokenizer.fit_on_texts(data_y)

    save_processed_data(tokenizer, save_tokenizer,
                        data_x, data_y, save_data)

    data_x = data_x if elmo else tokenizer.texts_to_sequences(tokenized_data_x)
    data_y = tokenizer.texts_to_sequences(tokenized_data_y)

    return data_x, data_y


def read_training_data(data_path, resources_path, elmo, download_from):
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
        (x_data, y_data, mask_builder) = parse_dataset(path, gold_dict, save_to_paths=save_data)
        save_tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
        train_x, train_y = process_dataset(x_data, y_data,
                                           save_tokenizer=save_tokenizer,
                                           save_data=save_data, elmo=elmo)
        return train_x, train_y, mask_builder
    except FileNotFoundError:
        download_to = os.path.join(
            os.getcwd(), 'data', 'training', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)
        read_training_data(data_path, resources_path, elmo)


def read_testing_data(data_path, resources_path, download_from, elmo):
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
        save_data = [os.path.join(resources_path, 'dev_x.pkl'),
                     os.path.join(resources_path, 'dev_y.pkl'),
                     os.path.join(resources_path, 'dev_mask.pkl')]
        (data_x, data_y, _) = parse_dataset(eval_path, eval_dict,
                                               save_to_paths=save_data)
        save_tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
        dev_x, dev_y = process_dataset(data_x, data_y,
                                       save_tokenizer=save_tokenizer,
                                       save_data=save_data, elmo=elmo)
        return dev_x, dev_y
    except FileNotFoundError:
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)
        read_testing_data(data_path, resources_path, download_from, elmo)


def load_dataset(summarize=False, elmo=False):
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    resources_path = os.path.join(cwd, 'resources')

    download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip'
    train_x, train_y, mask_builder = read_training_data(data_path, resources_path, elmo, download_from)

    download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip'
    dev_x, dev_y = read_testing_data(data_path, resources_path, elmo, download_from)

    save_tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
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
        'dev_x': dev_x,
        'dev_y': dev_y,
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

        pad_val = '<PAD>' if use_elmo else 0.
        data_x_ = pad_sequences(np.array(data_x_), padding='post',
                                value=pad_val, maxlen=max_len, dtype=object)

        data_y_ = pad_sequences(np.array(data_y_), padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        _data_x = data_x_ if not use_elmo else np.array([' '.join(x) for x in data_x_],
                                                        dtype=object)

        yield [_data_x], _data_y

        if start + batch_size > len(data_x):
            start = 0
        else:
            start += batch_size


def train_generator(data_x, data_y, batch_size, output_size,
                    use_elmo, mask_builder, tokenizer):
    start = 0
    while True:
        end = start + batch_size
        data_x_, data_y_ = data_x[start:end], data_y[start:end]
        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0.
        data_x_ = pad_sequences(np.array(data_x_), padding='post',
                                value=pad_val, maxlen=max_len, dtype=object)

        data_y_ = pad_sequences(np.array(data_y_), padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        mask_builder_ = mask_builder[start:end]
        mask_x = create_mask(mask_builder_,
                             [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size)
        _data_x = data_x_ if not use_elmo else np.array([' '.join(x) for x in data_x_],
                                                        dtype=object)
        yield [_data_x, mask_x], _data_y

        if start + batch_size > len(data_x):
            start = 0
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
