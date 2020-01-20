import logging
import os

import numpy as np
import yaml
from lxml.etree import iterparse
from nltk.corpus import wordnet as wn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
except ModuleNotFoundError:
    os.system('pip install langdetect')
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0

from utilities import (build_bn2dom_dict, build_bn2lex_dict, build_bn2wn_dict,
                       build_dict, dataset_summarize, download_unzip_dataset,
                       initialize_logger, load_pickle, save_pickle, get_lemma2synsets)


def parse_dataset(file_name, gold_dict, config_path, wordnet_babelnet,
                  babelnet_lex, babelnet_domain, save_to_paths=None, multilingual=False):
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
    pos_labeled_list, lex_labeled_list, domain_labeled_list = [], [], []

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
            pos_labeled_list = load_pickle(
                save_to_paths[1].replace('train_y', 'train_pos_y'))
            lex_labeled_list = load_pickle(
                save_to_paths[1].replace('train_y', 'train_lex_y'))
        else:
            pos_labeled_list = load_pickle(
                save_to_paths[1].replace('test_y', 'test_pos_y'))
            lex_labeled_list = load_pickle(
                save_to_paths[1].replace('test_y', 'test_lex_y'))
        logging.info("Parsed Dataset is loaded")
    else:
        # read file contents in terms of sentences
        context = iterparse(file_name, tag="sentence")
        # iterating over the sentences
        for _, elements in tqdm(context, desc="Parsing corpus"):
            sentence, sentence_labeled = [], []
            mask_builder = []
            sentence_pos_labeled, sentence_lex_labeled, sentence_domain_labeled = [], [], []
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
                        sentence_domain_labeled.append(elem_lemma)
                        if elem.tag == 'wf':
                            sentence_labeled[-1] = "<WF>"
                            mask_builder.append([elem_lemma])
                    if elem.tag == 'instance' and elem.text is not None:
                        elem_id, elem_lemma = elem.attrib['id'], elem.attrib['lemma']
                        sense_key, synset_id = str(
                            gold_dict.get(elem_id)), None
                        if sense_key is not None:
                            synset = wn.lemma_from_key(sense_key).synset()
                            synset_id = f"wn:{str(synset.offset()).zfill(8)}{synset.pos()}"
                            synset_id_ = wordnet_babelnet.get(synset_id)
                            sentence_labeled[-1] = f'{synset_id_}' if not multilingual else f'{elem_lemma}_{synset_id_}'
                            if synset_id_ is None:
                                sentence_lex_labeled[-1] = 'factotum'
                                sentence_domain_labeled[-1] = 'factotum'
                            else:
                                sentence_lex_labeled[-1] = str(
                                    babelnet_lex[wordnet_babelnet[synset_id]])
                                sentence_domain_labeled[-1] = str(babelnet_domain.get(wordnet_babelnet[synset_id],
                                                                                      'factotum'))
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
                domain_labeled_list.append(sentence_domain_labeled)
            elements.clear()
        logging.info("Parsed the dataset")

        while len(sentences_list) % batch_size != 0:
            sentences_list.append(sentences_list[0])
            labeled_sentences_list.append(labeled_sentences_list[0])
            masks_builder.append(masks_builder[0])
            pos_labeled_list.append(pos_labeled_list[0])
            lex_labeled_list.append(lex_labeled_list[0])
            domain_labeled_list.append(domain_labeled_list[0])

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

    return (sentences_list, labeled_sentences_list, masks_builder, pos_labeled_list,
            lex_labeled_list, domain_labeled_list)


def process_dataset(data_x, data_y, pos_labeled_list,
                    lex_labeled_list, dom_labeled_list,
                    save_tokenizer=None,
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
            pos_labeled_list = load_pickle(
                save_data[1].replace('train_y', 'train_pos_y'))
            lex_labeled_list = load_pickle(
                save_data[1].replace('train_y', 'train_lex_y'))
        else:
            pos_labeled_list = load_pickle(
                save_data[1].replace('test_y', 'test_pos_y'))
            lex_labeled_list = load_pickle(
                save_data[1].replace('test_y', 'test_lex_y'))
        logging.info("data_y is loaded")

    if (save_tokenizer is not None
            and os.path.exists(save_tokenizer)
            and os.path.getsize(save_tokenizer) > 0):
        tokenizer = load_pickle(save_tokenizer)
        pos_tokenizer = load_pickle(
            save_tokenizer.replace('tokenizer', 'pos_tokenizer'))
        lex_tokenizer = load_pickle(
            save_tokenizer.replace('tokenizer', 'lex_tokenizer'))
        dom_tokenizer = load_pickle(
            save_tokenizer.replace('tokenizer', 'dom_tokenizer'))
        try:
            # If text already converted to sequences then skip this part
            data_x = data_x if elmo else tokenizer.texts_to_sequences(data_x)
            data_y = tokenizer.texts_to_sequences(data_y)
            pos_labeled_list = pos_tokenizer.texts_to_sequences(
                pos_labeled_list)
            lex_labeled_list = lex_tokenizer.texts_to_sequences(
                lex_labeled_list)
            dom_labeled_list = dom_tokenizer.texts_to_sequences(
                dom_labeled_list)
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
        # This part had to be hard-coded, however, its generation was done using
        # lex_regex = '/\[a-z]+(?:\.)\[a-z]+/'
        # lex_vocab = list(set(lex_name for word in lex_tokenizer.word_index.keys()
        # for lex_name in word if re.match(lex_regex, lex_name) and len(lex_name) > 3))
        lex_vocab = ['adj.all', 'adj.pert', 'adj.ppl', 'adv.all', 'noun.Tops', 'noun.act',
                     'noun.animal', 'noun.artifact', 'noun.attribute', 'noun.body', 'noun.cognition',
                     'noun.communication', 'noun.event', 'noun.feeling', 'noun.food', 'noun.group', 'noun.location',
                     'noun.motive', 'noun.object', 'noun.person', 'noun.phenomenon', 'noun.plant', 'noun.possession',
                     'noun.process', 'noun.quantity', 'noun.relation', 'noun.shape', 'noun.state', 'noun.substance',
                     'noun.time', 'verb.body', 'verb.change', 'verb.cognition', 'verb.communication',
                     'verb.competition', 'verb.consumption', 'verb.contact', 'verb.creation', 'verb.emotion',
                     'verb.motion', 'verb.perception', 'verb.possession', 'verb.social', 'verb.stative',
                     'verb.weather', 'factotum']
        lex_tokenizer.word_index = (dict((w, i)
                                         for i, w in enumerate(lex_vocab, start=1)))
        lex_tokenizer.word_index.update({'<PAD>': 0})
        lex_tokenizer.index_word = dict(
            (i, w) for w, i in lex_tokenizer.word_index.items())

        dom_tokenizer = Tokenizer(oov_token='factotum')
        dom_tokenizer.fit_on_texts(dom_labeled_list)
        dom_tokenizer.word_index.update({'<PAD>': 0})
        dom_tokenizer.index_word.update({0: '<PAD>'})

        if save_tokenizer is not None:
            save_pickle(save_tokenizer, tokenizer)
            save_pickle(save_tokenizer.replace(
                'tokenizer', 'pos_tokenizer'), pos_tokenizer)
            save_pickle(save_tokenizer.replace(
                'tokenizer', 'lex_tokenizer'), lex_tokenizer)
            save_pickle(save_tokenizer.replace(
                'tokenizer', 'dom_tokenizer'), dom_tokenizer)
            logging.info("Tokenizers are Saved")

        data_x = data_x if elmo else tokenizer.texts_to_sequences(data_x)
        data_y = tokenizer.texts_to_sequences(data_y)
        pos_labeled_list = pos_tokenizer.texts_to_sequences(pos_labeled_list)
        lex_labeled_list = lex_tokenizer.texts_to_sequences(lex_labeled_list)
        dom_labeled_list = dom_tokenizer.texts_to_sequences(dom_labeled_list)

        if save_data is not None:
            save_pickle(save_data[0], data_x)
            save_pickle(save_data[1], data_y)
            if 'train' in save_data[1]:
                save_pickle(save_data[1].replace(
                    'train_y', 'train_pos_y'), pos_labeled_list)
                save_pickle(save_data[1].replace(
                    'train_y', 'train_lex_y'), lex_labeled_list)
                save_pickle(save_data[1].replace(
                    'train_y', 'train_dom_y'), dom_labeled_list)
            else:
                save_pickle(save_data[1].replace(
                    'test_y', 'test_pos_y'), pos_labeled_list)
                save_pickle(save_data[1].replace(
                    'test_y', 'test_lex_y'), lex_labeled_list)
                save_pickle(save_data[1].replace(
                    'test_y', 'test_dom_y'), dom_labeled_list)
            logging.info("Processed Data is Saved")

    return data_x, data_y, pos_labeled_list, lex_labeled_list, dom_labeled_list


def load_dataset(summarize=False, elmo=False, use_omsti=False):
    """
    Parse & Preprocess dataset, returns dataset dict.
    More to know:
    Reads Semcor dataset, and all the mappings required, process
    the data to have all different labels as per our sentences,
    prints out summary for frequency of sentences length, can integrate
    another dataset, and preprocess data for ELMo.
    If training data and or testing data is not there, downloads it,
    unzips then retry parsing and pre-processing
    :param summarize:
    :param elmo:
    :param use_omsti:
    :return: dataset {dict} contains
        -  train_x, lemmas of word format and instance tags
        -  train_y, lemma_sense format for every word of instance tag
        -  pos_labeled_list, lemma_pos format
        -  lex_labeled_list, lemma_lex format
        -  dom_labeled_list, lemma_dom
        -  test_x, same as train
        -  test_y, same as train
        -  pos_labeled_list_test, same as train
        -  lex_labeled_list_test, same as train
        -  tokenizer, tokenizer and the dictionary of the vocabulary
        -  vocabulary_size, num of words in our dataset
        -  output_size, num of senses in our dataset
        -  mask_builder, mask as per training data
        -  dev_mask_builder, mask as per testing data
        -  pos_vocab_size, num of poses in our dataset
        -  lex_vocab_size, num of lexes in our dataset
        -  dom_vocab_size num of domains in our dataset
    }

    """
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    resources_path = os.path.join(cwd, 'resources')
    config_path = os.path.join(cwd, 'config.yaml')
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    bn2lex_path = os.path.join(resources_path, 'babelnet2lexnames.tsv')
    bn2dom_path = os.path.join(resources_path, 'babelnet2wndomains.tsv')

    bn2wn_save_path = bn2wn_path.replace('tsv', 'pkl')
    wn2bn_save_path = bn2wn_path.replace(
        'babelnet2wordnet.tsv', 'wordnet2babelnet.pkl')
    _, wordnet_babelnet = build_bn2wn_dict(
        bn2wn_path, save_to=[bn2wn_save_path, wn2bn_save_path])

    bn2lex_save_path = bn2lex_path.replace('tsv', 'pkl')
    lex2bn_save_path = bn2lex_save_path.replace(
        'babelnet2lexnames.pkl', 'lexnames2babelnet.pkl')
    babelnet_lex, _ = build_bn2lex_dict(
        bn2lex_path, save_to=[bn2lex_save_path, lex2bn_save_path])

    bn2dom_save_path = bn2dom_path.replace(
        'babelnet2wndomains.tsv', 'babelnet2wndomains.pkl')
    dom2bn_save_path = bn2dom_save_path.replace(
        'babelnet2wndomains.pkl', 'wndomains2babelnet.pkl')
    babelnet_domain, _ = build_bn2dom_dict(
        bn2dom_path, save_to=[bn2dom_save_path, dom2bn_save_path])

    try:
        # Building the gold dictionary for training set
        file_path = os.path.join(
            data_path, 'training', 'WSD_Training_Corpora',
            'SemCor+OMSTI', 'semcor+omsti.gold.key.txt')
        save_to = os.path.join(resources_path, 'gold_omsti_dict.pkl')
        gold_dict = build_dict(file_path, save_to)

        path = os.path.join(data_path, 'training',
                            'WSD_Training_Corpora',
                            'SemCor', 'semcor.data.xml')

        save_data = [os.path.join(resources_path, 'train_x.pkl'),
                     os.path.join(resources_path, 'train_y.pkl'),
                     os.path.join(resources_path, 'train_mask.pkl')]

        (data_x, data_y, mask_builder,
         pos_labeled_list_, lex_labeled_list_, dom_labeled_list_) = parse_dataset(
            path, gold_dict, config_path, wordnet_babelnet,
            babelnet_lex, babelnet_domain, save_to_paths=save_data)

        if use_omsti:
            omsti_path = os.path.join(data_path, 'training',
                                      'WSD_Training_Corpora',
                                      'SemCor+OMSTI', 'omsti.data.xml')
            save_data = [os.path.join(resources_path, 'train_omsti_x.pkl'),
                         os.path.join(resources_path, 'train_omsti_y.pkl'),
                         os.path.join(resources_path, 'train_omsti_mask.pkl')]
            (data_x_o, data_y_o, mask_builder_o,
             pos_labeled_list_o_, lex_labeled_list_o_) = parse_dataset(
                omsti_path, gold_dict, config_path, wordnet_babelnet,
                babelnet_lex, save_to_paths=save_data)
            data_x.extend(data_x_o)
            data_y.extend(data_y_o)
            mask_builder.extend(mask_builder_o)
            pos_labeled_list_.extend(pos_labeled_list_o_)
            lex_labeled_list_.extend(lex_labeled_list_o_)
    except FileNotFoundError:
        download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip'
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)
        (data_x, data_y, mask_builder,
         pos_labeled_list_, lex_labeled_list_, dom_labeled_list_) = parse_dataset(
            path, gold_dict, config_path, wordnet_babelnet,
            babelnet_lex, babelnet_domain, save_to_paths=save_data)

    save_tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
    (train_x, train_y, pos_labeled_list,
     lex_labeled_list, dom_labeled_list) = process_dataset(data_x, data_y,
                                                           pos_labeled_list_, lex_labeled_list_,
                                                           dom_labeled_list_, save_tokenizer=save_tokenizer,
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
        (data_x, data_y, dev_mask_builder,
         dev_pos_labeled_list, dev_lex_labeled_list,
         dev_dom_labeled_list) = parse_dataset(eval_path, eval_dict,
                                               config_path, wordnet_babelnet,
                                               babelnet_lex, babelnet_domain,
                                               save_to_paths=save_data)
    except FileNotFoundError:
        download_from = 'http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip'
        download_to = os.path.join(
            os.getcwd(), 'data', 'evaluation', download_from.split('/')[-1])
        download_unzip_dataset(download_from, download_to)

    (test_x, test_y,
     pos_labeled_list_test,
     lex_labeled_list_test, dom_labeled_list_test) = process_dataset(data_x, data_y,
                                                                     dev_pos_labeled_list, dev_lex_labeled_list,
                                                                     dev_dom_labeled_list,
                                                                     save_tokenizer=save_tokenizer,
                                                                     save_data=save_data, elmo=elmo)
    tokenizer = load_pickle(save_tokenizer)
    word_tokens = [
        word for word in tokenizer.word_index if not word.startswith('wn:')]
    sense_tokens = [
        word for word in tokenizer.word_index if word.startswith('wn:')]

    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    pos_tokenizer = load_pickle(
        save_tokenizer.replace('tokenizer', 'pos_tokenizer'))
    pos_vocab_size = len(pos_tokenizer.word_index.items())

    lex_tokenizer = load_pickle(
        save_tokenizer.replace('tokenizer', 'lex_tokenizer'))
    lex_vocab_size = len(lex_tokenizer.word_index.items())

    dom_tokenizer = load_pickle(
        save_tokenizer.replace('tokenizer', 'dom_tokenizer'))
    dom_vocab_size = len(dom_tokenizer.word_index.items())
    '''
    lex_word_regex = '[a-z]+(?:\\.)[a-z]+$'
    lex_names = [word for word in lex_tokenizer.word_index.keys() if re.match(lex_word_regex, word) and len(word) > 3]
    lex_names.extend(['<OOV>', '<PAD>', 'factotum'])
    lex_vocab_size = len(lex_names)
    '''

    dataset = {
        'train_x': train_x,
        'train_y': train_y,
        'pos_labeled_list': pos_labeled_list,
        'lex_labeled_list': lex_labeled_list,
        'dom_labeled_list': dom_labeled_list,
        'test_x': test_x,
        'test_y': test_y,
        'pos_labeled_list_test': pos_labeled_list_test,
        'lex_labeled_list_test': lex_labeled_list_test,
        'tokenizer': load_pickle(save_tokenizer),
        'vocabulary_size': vocabulary_size,
        'output_size': output_size,
        'mask_builder': mask_builder,
        'dev_mask_builder': dev_mask_builder,
        'pos_vocab_size': pos_vocab_size,
        'lex_vocab_size': lex_vocab_size,
        'dom_vocab_size': dom_vocab_size
    }

    if summarize:
        dataset_summarize(dataset)

    return dataset


def create_mask(mask_builder, mask_shape, tokenizer, output_size,
                use_bert=False, lemma_synsets=None, wordnet_babelnet=None):
    mask_x = np.full(shape=(mask_shape[0],
                            mask_shape[1],
                            output_size),
                     fill_value=-np.inf)

    if use_bert:
        word2idx = tokenizer.vocab
    else:
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
            # Adding entropy to the system for numerical stability
            output_size = len(word2idx.items())
            random_numbers = np.random.randint(output_size, size=10)
            for idx in random_numbers:
                mask_x[sentence][word_arr][idx] = 0.0
            # if the word is of instance type, get candidate synsets' indices
            # and set them to zero.
            if len(mask_builder[sentence][word_arr]) == 2:
                candidate_synsets_idx = get_candidate_senses(word, word2idx, lemma_synsets,
                                                             wordnet_babelnet=wordnet_babelnet)
                for idx in candidate_synsets_idx:
                    mask_x[sentence][word_arr][idx] = 0.0

    return mask_x


def val_generator(data_x, data_y, batch_size, output_size,
                  use_elmo, mask_builder, tokenizer, use_bert):
    start = 0
    while True:
        end = start + batch_size
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        mask_builder = np.array(mask_builder)

        data_x_, data_y_ = data_x[start:end], data_y[start:end]
        mask_builder_ = mask_builder[start:end]
        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(np.array(data_x_), padding='post',
                                value=pad_val, maxlen=max_len, dtype=object).tolist()

        data_y_ = pad_sequences(np.array(data_y_), padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        mask_x = create_mask(mask_builder_,
                             [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size, use_bert=use_bert)

        _data_x = data_x_ if not use_elmo else np.array([' '.join(x) for x in data_x_],
                                                        dtype=object)

        yield [_data_x, mask_x], _data_y

        if start + batch_size > len(data_x):
            start = 0
        else:
            start += batch_size


def train_generator(data_x, data_y, batch_size, output_size,
                    use_elmo, mask_builder, tokenizer,
                    use_bert=False, shuffle=False):
    start = 0
    while True:
        end = start + batch_size
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        mask_builder = np.array(mask_builder)
        if shuffle:
            permutation = np.random.permutation(len(data_x))
            data_x_, data_y_ = data_x[permutation[start:end]
                                      ], data_y[permutation[start:end]]
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
                             tokenizer, output_size, use_bert=use_bert)

        if use_elmo:
            _data_x = np.array([' '.join(x) for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = data_x_

        mask_x = pad_sequences(
            np.array(mask_x), padding='post', value=0, maxlen=max_len)

        yield [_data_x, mask_x], _data_y

        if start + batch_size >= len(data_x):
            start = 0
            if shuffle:
                permutation = np.random.permutation(len(data_x))
        else:
            start += batch_size


def multitask_train_generator(data_x, data_y, pos_y, lex_y,
                              batch_size, output_size, use_elmo,
                              mask_builder, tokenizer, use_bert,
                              shuffle=False, lemma_synsets_dict=None,
                              wn2bn=None):
    start = 0
    while True:
        end = start + batch_size
        data_x, data_y = np.array(data_x), np.array(data_y)
        pos_y, lex_y = np.array(pos_y), np.array(lex_y)
        mask_builder = np.array(mask_builder)

        if shuffle:
            permutation = np.random.permutation(len(data_x))
            data_x_, data_y_ = data_x[permutation[start:end]
                                      ], data_y[permutation[start:end]]
            pos_y_, lex_y_ = pos_y[permutation[start:end]
                                   ], lex_y[permutation[start:end]]
            mask_builder_ = mask_builder[permutation[start:end]]

        else:
            data_x_, data_y_ = data_x[start:end], data_y[start:end]
            pos_y_, lex_y_ = pos_y[start:end], lex_y[start:end]
            mask_builder_ = mask_builder[start:end]

        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(data_x_, padding='post',
                                value=pad_val, maxlen=max_len, dtype=object).tolist()

        data_y_ = pad_sequences(data_y_, padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        pos_y_ = pad_sequences(pos_y_, padding='post',
                               value=0, maxlen=max_len, dtype=object)
        _pos_y = np.expand_dims(pos_y_, axis=-1)

        lex_y_ = pad_sequences(lex_y_, padding='post',
                               value=0, maxlen=max_len, dtype=object)
        _lex_y = np.expand_dims(lex_y_, axis=-1)

        mask_x = create_mask(mask_builder_,
                             [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size, use_bert=use_bert,
                             lemma_synsets=lemma_synsets_dict,
                             wordnet_babelnet=wn2bn)

        if use_elmo:
            _data_x = np.array([' '.join(str(x)) for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = np.array(data_x_)

        mask_x = pad_sequences(np.array(mask_x), padding='post', value=0, maxlen=max_len)

        yield [_data_x, mask_x], [_data_y, _pos_y, _lex_y]

        if start + batch_size >= len(data_x):
            start = 0
            if shuffle:
                permutation = np.random.permutation(len(data_x))
        else:
            start += batch_size


def bert_multitask_train_generator(data_x, data_y, lex_y, pos_y, batch_size,
                                   output_size, use_elmo, mask_builder, tokenizer, shuffle=False):
    start = 0
    input_ids, input_masks, segment_ids = data_x
    while True:
        end = start + batch_size
        input_ids = np.array(input_ids)
        input_masks, segment_ids = np.array(input_masks), np.array(segment_ids)
        data_y = np.array(data_y)
        pos_y_, lex_y_ = np.array(pos_y), np.array(lex_y)
        mask_builder = np.array(mask_builder)
        if shuffle:
            permutation = np.random.permutation(len(input_ids))
            data_x_, data_y_ = input_ids[permutation[start:end]
                                         ], data_y[permutation[start:end]]
            input_masks_, segment_ids_ = input_masks[permutation[start:end]
                                                     ], segment_ids[permutation[start:end]]
            mask_builder_ = mask_builder[permutation[start:end]]
            pos_y, lex_y = pos_y[permutation[start:end]
                                 ], lex_y[permutation[start:end]]
        else:
            data_x_, data_y_ = input_ids[start:end], data_y[start:end]
            input_masks_, segment_ids_ = input_masks[start:end], segment_ids[start:end]
            mask_builder_ = mask_builder[start:end]
            pos_y_, lex_y_ = pos_y_[start:end], lex_y_[start:end]

        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(data_x_, padding='post',
                                value=pad_val, maxlen=max_len, dtype=object)

        data_y_ = pad_sequences(data_y_, padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        pos_y_ = pad_sequences(pos_y_, padding='post',
                               value=0, maxlen=max_len, dtype=object)
        _pos_y = np.expand_dims(pos_y_, axis=-1)

        lex_y_ = pad_sequences(lex_y_, padding='post',
                               value=0, maxlen=max_len, dtype=object)
        _lex_y = np.expand_dims(lex_y_, axis=-1)

        mask_x = create_mask(mask_builder_, [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size, use_bert=True)

        if use_elmo:
            _data_x = np.array([' '.join(str(x))
                                for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = data_x_

        mask_x = pad_sequences(
            np.array(mask_x), padding='post', value=0, maxlen=max_len)
        input_masks_ = pad_sequences(
            np.array(input_masks_), padding='post', value=0, maxlen=max_len)
        segment_ids_ = pad_sequences(
            np.array(segment_ids_), padding='post', value=0, maxlen=max_len)

        yield [_data_x, input_masks_, segment_ids_, mask_x], [_data_y, _pos_y, _lex_y]

        if start + batch_size >= len(data_x):
            start = 0
            if shuffle:
                permutation = np.random.permutation(len(data_x))
        else:
            start += batch_size


def bert_train_generator(data_x, data_y, batch_size, output_size,
                         use_elmo, mask_builder, tokenizer, shuffle=False):
    start = 0
    input_ids, input_masks, segment_ids = data_x
    while True:
        end = start + batch_size
        input_ids = np.array(input_ids)
        input_masks, segment_ids = np.array(input_masks), np.array(segment_ids)
        data_y = np.array(data_y)
        mask_builder = np.array(mask_builder)
        if shuffle:
            permutation = np.random.permutation(len(input_ids))
            data_x_, data_y_ = input_ids[permutation[start:end]
                                         ], data_y[permutation[start:end]]
            input_masks_, segment_ids_ = input_masks[permutation[start:end]
                                                     ], segment_ids[permutation[start:end]]
            mask_builder_ = mask_builder[permutation[start:end]]

        else:
            data_x_, data_y_ = input_ids[start:end], data_y[start:end]
            input_masks_, segment_ids_ = input_masks[start:end], segment_ids[start:end]
            mask_builder_ = mask_builder[start:end]

        max_len = len(max(data_x_, key=len))

        pad_val = '<PAD>' if use_elmo else 0
        data_x_ = pad_sequences(data_x_, padding='post',
                                value=pad_val, maxlen=max_len, dtype=object)

        data_y_ = pad_sequences(data_y_, padding='post',
                                value=0, maxlen=max_len, dtype=object)
        _data_y = np.expand_dims(data_y_, axis=-1)

        mask_x = create_mask(mask_builder_,
                             [_data_y.shape[0], _data_y.shape[1]],
                             tokenizer, output_size, use_bert=True)

        if use_elmo:
            _data_x = np.array([' '.join(str(x))
                                for x in data_x_], dtype=object)
            _data_x = np.expand_dims(_data_x, axis=-1)
        else:
            _data_x = data_x_

        mask_x = pad_sequences(
            np.array(mask_x), padding='post', value=0, maxlen=max_len)
        input_masks_ = pad_sequences(
            np.array(input_masks_), padding='post', value=0, maxlen=max_len)
        segment_ids_ = pad_sequences(
            np.array(segment_ids_), padding='post', value=0, maxlen=max_len)

        yield [_data_x, input_masks_, segment_ids_, mask_x], _data_y

        if start + batch_size >= len(data_x):
            start = 0
            if shuffle:
                permutation = np.random.permutation(len(data_x))
        else:
            start += batch_size


def get_candidate_senses(word, word2idx, lemma_synsets=None, wordnet_babelnet=None):
    if not lemma_synsets:
        candidates = [f'wn:{str(synset.offset()).zfill(8)}{synset.pos()}'
                      for synset in wn.synsets(word)]
        return [word2idx.get(candidate, None) for candidate in candidates
                if word2idx.get(candidate, None)]
    else:
        candidates = lemma_synsets.get(word, [])
        try:
            if wordnet_babelnet:
                word_lang = (detect(str(word))).lower()
                # instead of implementing multiple if-else statements, i added them into a dictionary
                # replicating switch cases
                languages = {'fr': 'fra', 'it': 'ita', 'es': 'spa', 'en': 'eng'}
                language = languages.get(word_lang)
                if language:
                    wn_candidates = [f'wn:{str(synset.offset()).zfill(8)}{synset.pos()}'
                                     for synset in wn.synsets(word, lang=language)]
                    bn_candidates = [wordnet_babelnet.get(candidate) for candidate in wn_candidates
                                     if wordnet_babelnet.get(candidate)]
                    candidates.extend(bn_candidates)
        except LangDetectException:
            pass

        return [word2idx.get(f'{word}_{candidate}') for candidate in candidates
                if word2idx.get(f'{word}_{candidate}')]


def parse_onesec(file_name, gold_dict, config_params, babelnet_wordnet, babelnet_lex, babelnet_domain):
    # load configuration file
    batch_size = config_params['batch_size']

    sentences_list, labeled_sentences_list, masks_builder = [], [], []
    pos_labeled_list, dom_labeled_list, lex_labeled_list = [], [], []
    # read file contents in terms of sentences
    context = iterparse(file_name, tag="sentence")
    # iterating over the sentences
    for _, elements in tqdm(context, desc="Parsing corpus"):
        sentence, sentence_labeled, mask_builder = [], [], []
        sentence_pos_labeled, sentence_lex_labeled, sentence_domain_labeled = [], [], []
        for elem in list(elements.iter()):
            if elem is not None:
                if (elem.tag == 'wf' or elem.tag == 'instance') and elem.text is not None:
                    elem_lemma = elem.attrib['lemma']
                    elem_pos = elem.attrib['pos']
                    sentence.append(elem_lemma)
                    sentence_labeled.append(elem_lemma)
                    sentence_pos_labeled.append(elem_pos)
                    sentence_lex_labeled.append(elem_lemma)
                    sentence_domain_labeled.append(elem_lemma)
                    if elem.tag == 'wf':
                        sentence_labeled[-1] = "<WF>"
                        mask_builder.append([elem_lemma])
                if elem.tag == 'instance' and elem.text is not None:
                    elem_lemma = elem.attrib['lemma']
                    elem_id = elem.attrib['id']
                    sense_key = str(gold_dict.get(elem_id))
                    if sense_key is not None:
                        sentence_labeled[-1] = f'{elem_lemma}_{sense_key}'
                        sentence_lex_labeled[-1] = str(babelnet_lex.get(sense_key, 'factotum'))
                        sentence_domain_labeled[-1] = str(babelnet_domain.get(sense_key, 'factotum'))
                    else:
                        sentence_lex_labeled[-1] = 'factotum'
                        sentence_domain_labeled[-1] = 'factotum'
                    if elem_lemma:
                        mask_builder.append([elem_lemma, sense_key])
        # if the sentence is not empty
        if len(sentence) and len(sentence_labeled) \
                and len(mask_builder) and len(sentence_pos_labeled) \
                and len(sentence_lex_labeled) and len(sentence_domain_labeled):
            sentences_list.append(sentence)
            labeled_sentences_list.append(sentence_labeled)
            masks_builder.append(mask_builder)
            pos_labeled_list.append(sentence_pos_labeled)
            lex_labeled_list.append(sentence_lex_labeled)
            dom_labeled_list.append(sentence_domain_labeled)
        elements.clear()
    logging.info("Parsed the dataset")

    while len(sentences_list) % batch_size != 0:
        sentences_list.append(sentences_list[0])
        labeled_sentences_list.append(labeled_sentences_list[0])
        masks_builder.append(masks_builder[0])
        pos_labeled_list.append(pos_labeled_list[0])
        lex_labeled_list.append(lex_labeled_list[0])
        dom_labeled_list.append(dom_labeled_list[0])

    return (sentences_list, labeled_sentences_list,
            masks_builder, pos_labeled_list,
            lex_labeled_list, dom_labeled_list)


def load_onesec():
    config_file_path = os.path.join(os.getcwd(), 'config.yaml')
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    data_x, data_y, masks = [], [], []
    data_pos, data_lex, data_dom = [], [], []

    resources_path = os.path.join(os.getcwd(), 'resources')
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    bn2lex_path = os.path.join(resources_path, 'babelnet2lexnames.tsv')
    bn2dom_path = os.path.join(resources_path, 'babelnet2wndomains.tsv')

    babelnet_wordnet, _ = build_bn2wn_dict(bn2wn_path)
    babelnet_lex, _ = build_bn2lex_dict(bn2lex_path)
    babelnet_domain, _ = build_bn2dom_dict(bn2dom_path)

    es_data_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'ES',
                                'WSD_framework', 'onesec_200_2.0_ES.data.xml')
    keys_file_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'ES', 'WSD_framework',
                                  'onesec_200_2.0_ES.key.txt')
    es_mapping_dict = build_dict(keys_file_path)
    (sentences_list, labeled_sentences_list, masks_builder,
     pos_labeled_list, lex_labeled_list, dom_labeled_list) = parse_onesec(es_data_path, es_mapping_dict,
                                                                          config_params, babelnet_wordnet,
                                                                          babelnet_lex, babelnet_domain)
    data_x.extend(sentences_list[:10_000])
    data_y.extend(labeled_sentences_list[:10_000])
    masks.extend(masks_builder[:10_000])
    data_pos.extend(pos_labeled_list[:10_000])
    data_lex.extend(lex_labeled_list[:10_000])
    data_dom.extend(dom_labeled_list[:10_000])
    logging.info('Parsed the ES dataset')

    de_data_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'DE',
                                'WSD_framework', 'onesec_200_2.0_DE.data.xml')
    keys_file_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'DE', 'WSD_framework',
                                  'onesec_200_2.0_DE.key.txt')

    de_mapping_file = build_dict(keys_file_path)
    (sentences_list, labeled_sentences_list, masks_builder,
     pos_labeled_list, lex_labeled_list, dom_labeled_list) = parse_onesec(de_data_path, de_mapping_file,
                                                                          config_params, babelnet_wordnet,
                                                                          babelnet_lex, babelnet_domain)
    data_x.extend(sentences_list[:10_000])
    data_y.extend(labeled_sentences_list[:10_000])
    masks.extend(masks_builder[:10_000])
    data_pos.extend(pos_labeled_list[:10_000])
    data_lex.extend(lex_labeled_list[:10_000])
    data_dom.extend(dom_labeled_list[:10_000])
    logging.info('Parsed the DE dataset')

    fr_data_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'FR',
                                'WSD_framework', 'onesec_200_2.0_FR.data.xml')
    keys_file_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'FR', 'WSD_framework',
                                  'onesec_200_2.0_FR.key.txt')
    fr_mapping_dict = build_dict(keys_file_path)
    (sentences_list, labeled_sentences_list, masks_builder,
     pos_labeled_list, lex_labeled_list, dom_labeled_list) = parse_onesec(fr_data_path, fr_mapping_dict,
                                                                          config_params, babelnet_wordnet,
                                                                          babelnet_lex, babelnet_domain)
    data_x.extend(sentences_list[:10_000])
    data_y.extend(labeled_sentences_list[:10_000])
    masks.extend(masks_builder[:10_000])
    data_pos.extend(pos_labeled_list[:10_000])
    data_lex.extend(lex_labeled_list[:10_000])
    data_dom.extend(dom_labeled_list[:10_000])
    logging.info('Parsed the FR dataset')

    it_data_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'IT',
                                'WSD_framework', 'onesec_200_2.0_IT.data.xml')
    keys_file_path = os.path.join(os.getcwd(), 'data', 'training', 'onesec', 'IT', 'WSD_framework',
                                  'onesec_200_2.0_IT.key.txt')
    it_mapping_dict = build_dict(keys_file_path)
    (sentences_list, labeled_sentences_list, masks_builder,
     pos_labeled_list, lex_labeled_list, dom_labeled_list) = parse_onesec(it_data_path, it_mapping_dict,
                                                                          config_params, babelnet_wordnet,
                                                                          babelnet_lex, babelnet_domain)
    data_x.extend(sentences_list[:10_000])
    data_y.extend(labeled_sentences_list[:10_000])
    masks.extend(masks_builder[:10_000])
    data_pos.extend(pos_labeled_list[:10_000])
    data_lex.extend(lex_labeled_list[:10_000])
    data_dom.extend(dom_labeled_list[:10_000])
    logging.info('Parsed the IT dataset')

    return data_x, data_y, masks, data_pos, data_lex, data_dom


def load_multilingual(load=False):
    resources_path = os.path.join(os.getcwd(), 'resources')
    dataset_save_path = os.path.join(resources_path, 'multilingual_dataset.pkl')

    if load and os.path.exists(dataset_save_path) and os.path.getsize(dataset_save_path) > 0:
        logging.info('Dataset is loaded.')
        return load_pickle(dataset_save_path)

    config_file_path = os.path.join(os.getcwd(), 'config.yaml')

    file_path = os.path.join(
        os.getcwd(), 'data', 'training', 'WSD_Training_Corpora',
        'SemCor+OMSTI', 'semcor+omsti.gold.key.txt')
    gold_dict = build_dict(file_path)

    path = os.path.join(os.getcwd(), 'data', 'training',
                        'WSD_Training_Corpora',
                        'SemCor', 'semcor.data.xml')

    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    bn2lex_path = os.path.join(resources_path, 'babelnet2lexnames.tsv')
    bn2dom_path = os.path.join(resources_path, 'babelnet2wndomains.tsv')

    _, wordnet_babelnet = build_bn2wn_dict(bn2wn_path)

    babelnet_lex, _ = build_bn2lex_dict(bn2lex_path)

    babelnet_domain, _ = build_bn2dom_dict(bn2dom_path)

    (en_x, en_y, mask_builder,
     pos_labeled_list_, lex_labeled_list_,
     dom_labeled_list_) = parse_dataset(path, gold_dict, config_file_path,
                                        wordnet_babelnet, babelnet_lex,
                                        babelnet_domain, multilingual=True)
    onesec_x, onesec_y, onesec_masks, onesec_pos, onesec_lex, onesec_dom = load_onesec()

    data_x, data_y, masks, data_pos, data_lex, data_dom = [], [], [], [], [], []

    data_x.extend(en_x[:10000])
    data_y.extend(en_y[:10000])
    masks.extend(mask_builder[:10000])
    data_pos.extend(pos_labeled_list_[:10000])
    data_lex.extend(lex_labeled_list_[:10000])
    data_dom.extend(dom_labeled_list_[:10000])

    data_x.extend(onesec_x)
    data_y.extend(onesec_y)
    masks.extend(onesec_masks)
    data_pos.extend(onesec_pos)
    data_lex.extend(onesec_lex)
    data_dom.extend(onesec_dom)

    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(data_x)
    tokenizer.fit_on_texts(data_y)
    tokenizer.word_index.update({'<PAD>': 0})
    tokenizer.index_word.update({0: '<PAD>'})

    pos_tokenizer = Tokenizer(oov_token='<OOV>')
    pos_tokenizer.fit_on_texts(data_pos)
    pos_tokenizer.word_index.update({'<PAD>': 0})
    pos_tokenizer.index_word.update({0: '<PAD>'})

    lex_tokenizer = Tokenizer(oov_token='factotum')
    lex_tokenizer.fit_on_texts(data_lex)
    lex_vocab = ['adj.all', 'adj.pert', 'adj.ppl', 'adv.all', 'noun.Tops', 'noun.act',
                 'noun.animal', 'noun.artifact', 'noun.attribute', 'noun.body', 'noun.cognition',
                 'noun.communication', 'noun.event', 'noun.feeling', 'noun.food', 'noun.group', 'noun.location',
                 'noun.motive', 'noun.object', 'noun.person', 'noun.phenomenon', 'noun.plant', 'noun.possession',
                 'noun.process', 'noun.quantity', 'noun.relation', 'noun.shape', 'noun.state', 'noun.substance',
                 'noun.time', 'verb.body', 'verb.change', 'verb.cognition', 'verb.communication',
                 'verb.competition', 'verb.consumption', 'verb.contact', 'verb.creation', 'verb.emotion',
                 'verb.motion', 'verb.perception', 'verb.possession', 'verb.social', 'verb.stative',
                 'verb.weather', 'factotum', 'X', 'N', 'V', 'J', 'R']
    lex_tokenizer.word_index = (dict((w, i)
                                     for i, w in enumerate(lex_vocab, start=1)))
    lex_tokenizer.word_index.update({'<PAD>': 0})
    lex_tokenizer.index_word = dict(
        (i, w) for w, i in lex_tokenizer.word_index.items())

    dom_tokenizer = Tokenizer(oov_token='factotum')
    dom_tokenizer.fit_on_texts(data_dom)
    dom_tokenizer.word_index.update({'<PAD>': 0})
    dom_tokenizer.index_word.update({0: '<PAD>'})

    word_tokens = [word for word in tokenizer.word_index if 'bn:' not in word]
    sense_tokens = [word for word in tokenizer.word_index if 'bn:' in word]
    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    pos_vocab_size = len(pos_tokenizer.word_index.items())
    lex_vocab_size = len(lex_tokenizer.word_index.items())
    dom_vocab_size = len(dom_tokenizer.word_index.items())

    config_file = open(config_file_path)
    config_params = yaml.load(config_file)
    elmo = config_params["use_elmo"]

    data_x_ = data_x if elmo else tokenizer.texts_to_sequences(data_x)
    data_y_ = tokenizer.texts_to_sequences(data_y)
    data_pos_ = pos_tokenizer.texts_to_sequences(data_pos)
    data_lex_ = lex_tokenizer.texts_to_sequences(data_lex)
    data_dom_ = dom_tokenizer.texts_to_sequences(data_dom)

    print(f"""The Dataset contains total of \n{vocabulary_size} unique tokens,
            {len(sense_tokens)} sense tokens.\nTotal {output_size} unique tokens.
            {dom_vocab_size} domains,\n{lex_vocab_size} Lex-names,\n{pos_vocab_size} POS-TAGs""")

    lemma2synsets_file_path = os.path.join(os.getcwd(), 'data', 'training', 'lemma2synsets4.0.xx.wn.ALL.txt')
    lemma_synsets = get_lemma2synsets(lemma2synsets_file_path)

    dataset = {
        'train_x': data_x_,
        'train_y': data_y_,
        'mask_builder': masks,
        'vocabulary_size': vocabulary_size,
        'output_size': output_size,
        'tokenizer': tokenizer,
        'pos_labeled_list': data_pos_,
        'lex_labeled_list': data_lex_,
        'dom_labeled_list': data_dom_,
        'pos_vocab_size': pos_vocab_size,
        'lex_vocab_size': lex_vocab_size,
        'dom_vocab_size': dom_vocab_size,
        'lemma_synsets': lemma_synsets
    }

    save_pickle(save_to=dataset_save_path, save_what=dataset)
    tokenizer_path = os.path.join(os.getcwd(), 'resources', 'multitask_tokenizer.pkl')
    save_pickle(save_to=tokenizer_path, save_what=tokenizer)

    return dataset


if __name__ == "__main__":
    initialize_logger()
    # semcor_dataset = load_dataset(summarize=True, elmo=False)
    multilingual_dataset = load_multilingual()
