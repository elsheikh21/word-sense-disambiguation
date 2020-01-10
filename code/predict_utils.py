import os

import numpy as np
from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from parsing_dataset import create_mask


def build_gold(file_name):
    file_dict = dict()
    with open(file_name, mode='r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc='Building dictionary'):
            synset_id, synset = line.split()[0], line.split()[1]
            file_dict[synset_id] = synset
    return file_dict


def parse_test(input_path, tokenizer, gold_dict, batch_size, elmo=False):
    context = iterparse(input_path, tag="sentence")
    sentences_list, masks_list = [], []
    # labeled_sentences_list = []
    # iterating over the sentences
    for _, elements in tqdm(context, desc="Parsing corpus"):
        sentence, mask_builder = [], []
        # sentence_labeled = []
        for elem in list(elements.iter()):
            if elem is not None:
                if elem.tag == 'wf' and elem.text is not None:
                    elem_lemma = elem.attrib['lemma']
                    sentence.append(elem_lemma)
                    # sentence_labeled.append(elem_lemma)
                    mask_builder.append([elem_lemma])
                if elem.tag == 'instance' and elem.text is not None:
                    elem_lemma = elem.attrib['lemma']
                    elem_id = elem.attrib['id']
                    sentence.append(elem_lemma)
                    # sense_key, synset_id = str(gold_dict.get(elem_id)), None
                    # if sense_key is not None:
                    #     synset = wn.lemma_from_key(sense_key).synset()
                    #     synset_id = f"wn:{str(synset.offset()).zfill(8)}{synset.pos()}"
                    #     sentence_labeled[-1] = f'{synset_id}'
                    mask_builder.append([elem_lemma, elem_id])
        # if the sentence is not empty
        if len(sentence) and len(mask_builder):
            sentences_list.append(sentence)
            # labeled_sentences_list.append(sentence_labeled)
            masks_list.append(mask_builder)
        elements.clear()

    while len(sentences_list) % batch_size != 0:
        sentences_list.append(sentences_list[0])
        # labeled_sentences_list.append(sentence_labeled[0])
        masks_list.append(masks_list[0])

    test_x = sentences_list if elmo else tokenizer.texts_to_sequences(sentences_list)

    return test_x, masks_list


def build_bn2domains_dict(file_path):
    babelnet2domains = dict()
    with open(file_path, mode='r') as file:
        lines = file.read().splitlines()
        for line in tqdm(lines, desc='Building babelnet_domains mapping dict'):
            bn, domain = line.split('\t')
            babelnet2domains[bn] = domain

    domains2babelnet = dict([[v, k] for k, v in babelnet2domains.items()])

    return babelnet2domains, domains2babelnet


def predict_sense(token):
    synsets = wn.synsets(token)
    return token if synsets is None or len(synsets) else f'wn:{str(synsets[0].offset()).zfill(8)}{synsets[0].pos()}'


def test_generator(data_x, batch_size, output_size, use_elmo, mask_builder, tokenizer, use_bert):
    for start in range(0, len(data_x), batch_size):
        end = start + batch_size
        data_x_ = data_x[start:end]
        mask_builder_ = mask_builder[start:end]

        max_len = len(max(data_x_, key=len))
        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(data_x_, padding='post', value=pad_val, maxlen=max_len, dtype=object)

        if use_elmo:
            _data_x = np.array([' '.join(x) for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = data_x_

        mask_x = create_mask(mask_builder_,
                             [data_x_.shape[0], data_x_.shape[1]],
                             tokenizer, output_size, use_bert=use_bert)

        mask_x = pad_sequences(np.array(mask_x), padding='post', value=0, maxlen=max_len)

        yield _data_x, mask_x


if __name__ == '__main__':
    file_path = os.path.join(os.getcwd(), 'resources', 'babelnet2wndomains.tsv')
    babelnet_domains, domains_babelnet = build_bn2domains_dict(file_path)