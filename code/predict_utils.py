import os
import tensorflow.keras.backend as K
from lxml.etree import iterparse
from tqdm import tqdm
from nltk.corpus import wordnet as wn


def build_gold(file_name):
    file_dict = dict()
    with open(file_name, mode='r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc='Building dictionary'):
            synset_id, synset = line.split()[0], line.split()[1]
            file_dict[synset_id] = synset
    return file_dict


def parse_test(input_path, gold_dict, tokenizer):
    context = iterparse(input_path, tag="sentence")
    sentences_list, lbld_sentences_list = [], []
    # iterating over the sentences
    for _, elements in tqdm(context, desc="Parsing corpus"):
        sentence, sentence_labeled = [], []
        for elem in list(elements.iter()):
            if elem is not None:
                if ((elem.tag == 'wf' or elem.tag == 'instance') and
                        elem.text is not None):
                    elem_lemma = elem.attrib['lemma']
                    sentence.append(elem_lemma)
                    sentence_labeled.append(elem_lemma)
                if elem.tag == 'instance' and elem.text is not None:
                    elem_id = elem.attrib['id']
                    elem_lemma = elem.attrib['lemma']
                    sense_key = str(gold_dict.get(elem_id))
                    if sense_key is not None:
                        synset = wn.lemma_from_key(sense_key).synset()
                        synset_id = f"wn:{str(synset.offset()).zfill(8)}{synset.pos()}"
                        sentence_labeled[-1] = f'{synset_id}'
        sentence_, sentence_labeled_ = ' '.join(
            sentence), ' '.join(sentence_labeled)
        # if the sentence is not empty
        if len(sentence_) and len(sentence_labeled_):
            sentences_list.append(sentence_)
            lbld_sentences_list.append(sentence_labeled_)
        elements.clear()
    return sentences_list, lbld_sentences_list


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def build_bn2domains_dict(file_path):
    babelnet2domains = dict()
    with open(file_path, mode='r') as file:
        lines = file.read().splitlines()
        for line in tqdm(lines, desc='Building babelnet_domains mapping dict'):
            bn, domain = line.split('\t')
            babelnet2domains[bn] = domain

    domains2babelnet = dict([[v, k] for k, v in babelnet2domains.items()])

    return babelnet2domains, domains2babelnet


if __name__ == '__main__':
    file_path = os.path.join(os.getcwd(), 'resources', 'babelnet2wndomains.tsv')
    babelnet_domains, domains_babelnet = build_bn2domains_dict(file_path)