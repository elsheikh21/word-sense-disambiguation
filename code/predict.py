import os
import yaml
import logging
import numpy as np
from tensorflow.keras.models import load_model

from tqdm import tqdm
from keras_elmo import ElmoEmbeddingLayer

from predict_utils import (f1_m, parse_test,
                           predict_sense, test_generator)
from utilities import (build_bn2wn_dict, load_pickle,
                       initialize_logger, configure_tf,
                       save_pickle, build_bn2lex_dict,
                       build_bn2dom_dict)


def predict_babelnet(input_path: str, output_path: str, resources_path: str):
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    # Load configuration params
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    config_file = open(config_path)
    config_params = yaml.load(config_file)
    batch_size = config_params['batch_size']

    # Parse and process Test set
    elmo = config_params["use_elmo"]
    tokenizer_path = os.path.join(resources_path, 'tokenizer.pkl')
    tokenizer = load_pickle(tokenizer_path)

    data_x, test_mask_builder = parse_test(
        input_path, tokenizer, batch_size, elmo)  # Raw Data

    word_tokens = [word for word in tokenizer.word_index
                   if not word.startswith('wn:')]
    sense_tokens = [word for word in tokenizer.word_index
                    if word.startswith('wn:')]
    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    # Load model
    model_path = os.path.join(resources_path, 'Baseline_model.h5')
    custom_objs = {
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer,
        'f1_m': f1_m
    }
    model = load_model(model_path, custom_objects=custom_objs)
    logging.info(f'{model._name} is loaded.')

    # Model Predictions
    predictions = []
    for batch_x, batch_test in tqdm(test_generator(np.array(data_x), batch_size, output_size,
                                                   elmo, np.array(test_mask_builder), tokenizer, False),
                                    desc="Predicting_Senses"):
        # Output Shape (batch_size, max_len_per_batch, output_vocab_size)
        batch_pred = model.predict_on_batch([batch_x, batch_test])
        # Output Shape (batch_size, max_len_per_batch)
        y_hat = np.argmax(batch_pred, axis=-1)
        predictions.extend(y_hat)

    # Dictionaries to get mappings
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    _, wordnet_babelnet = build_bn2wn_dict(bn2wn_path)

    # Save predictions to a file
    id_bn_list = []  # stands for predictions in {word_id babelnet_sense}
    for i, sentence in enumerate(tqdm(data_x)):
        for j, word in enumerate(sentence):
            if len(test_mask_builder[i][j]) == 2:  # So it is an instance
                prediction = predictions[i][j]
                prediction_sense = tokenizer.index_word.get(
                    prediction, '<OOV>')
                if 'wn:' not in prediction_sense or 'bn:' not in prediction_sense:
                    prediction_sense = predict_sense(token=word)   # Fallback
                word_id = test_mask_builder[i][j][1]
                id_ = word_id[word_id.find('.') + 1:]
                bn = wordnet_babelnet.get(prediction_sense, None)
                if id_ is None or bn is None:
                    continue
                id_bn_list.append(f'{id_}\t{bn}')

    with open(output_path, mode="w+") as output_file:
        for id_bn in tqdm(id_bn_list, desc="Writing model predictions"):
            output_file.write(f'{id_bn}\n')


def predict_wordnet_domains(input_path: str, output_path: str, resources_path: str):
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    # Load configuration params
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    config_file = open(config_path)
    config_params = yaml.load(config_file)

    # Parse and process Test set
    elmo = config_params["use_elmo"]
    batch_size = config_params["batch_size"]
    tokenizer_path = os.path.join(resources_path, 'tokenizer.pkl')
    tokenizer = load_pickle(tokenizer_path)

    data_x, test_mask_builder = parse_test(
        input_path, tokenizer, batch_size, elmo)  # Raw Data

    word_tokens = [word for word in tokenizer.word_index
                   if not word.startswith('wn:')]
    sense_tokens = [word for word in tokenizer.word_index
                    if word.startswith('wn:')]
    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    # Load model
    model_path = os.path.join(resources_path, 'Baseline_model.h5')
    custom_objs = {
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer
    }

    model = load_model(model_path, custom_objects=custom_objs)
    logging.info(f'{model._name} is loaded.')

    # Model Predictions
    predictions = []
    for batch_x, batch_test in tqdm(test_generator(np.array(data_x), batch_size, output_size,
                                                   elmo, np.array(test_mask_builder), tokenizer, False),
                                    desc="Predicting_Domains"):
        # Output Shape (batch_size, max_len_per_batch, output_vocab_size)
        batch_pred = model.predict_on_batch([batch_x, batch_test])
        # Output Shape (batch_size, max_len_per_batch)
        y_hat = np.argmax(batch_pred, axis=-1)
        predictions.extend(y_hat)

    # Dictionaries to get mappings
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    _, wordnet_babelnet = build_bn2wn_dict(bn2wn_path)

    babelnet2wndom_path = os.path.join(
        resources_path, 'babelnet2wndomains.tsv')
    babelnet_domains, _ = build_bn2dom_dict(babelnet2wndom_path)

    # Save predictions to a file
    id_dom_list = []  # stands for predictions in {word_id babelnet_sense}
    for i, sentence in enumerate(tqdm(data_x, desc='Formatting model predictions into WordID_Domain')):
        for j, word in enumerate(sentence):
            if len(test_mask_builder[i][j]) == 2:  # So it is an instance
                prediction = predictions[i][j]
                prediction_sense = tokenizer.index_word.get(
                    prediction, '<OOV>')
                if 'wn:' not in prediction_sense or 'bn:' not in prediction_sense:
                    prediction_sense = predict_sense(token=word)  # Fallback
                word_id = test_mask_builder[i][j][1]
                id_ = word_id[word_id.find('.') + 1:]
                bn = wordnet_babelnet.get(prediction_sense, None)
                dom = babelnet_domains.get(bn)
                if id_ is None or dom is None:
                    continue
                id_dom_list.append(f'{id_}\t{dom}')

    with open(output_path, mode="w+") as output_file:
        for id_dom in tqdm(id_dom_list, desc="Writing model predictions"):
            output_file.write(f'{id_dom}\n')


def predict_lexicographer(input_path: str, output_path: str, resources_path: str):
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    # Load configuration params
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    config_file = open(config_path)
    config_params = yaml.load(config_file)

    # Parse and process Test set
    elmo = config_params["use_elmo"]
    batch_size = config_params["batch_size"]
    tokenizer_path = os.path.join(resources_path, 'tokenizer.pkl')
    tokenizer = load_pickle(tokenizer_path)

    data_x, test_mask_builder = parse_test(
        input_path, tokenizer, batch_size, elmo)  # Raw Data

    word_tokens = [word for word in tokenizer.word_index
                   if not word.startswith('wn:')]
    sense_tokens = [word for word in tokenizer.word_index
                    if word.startswith('wn:')]
    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    # Load model
    model_path = os.path.join(resources_path, 'Baseline_model.h5')
    custom_objs = {
        'ElmoEmbeddingLayer': ElmoEmbeddingLayer
    }

    model = load_model(model_path, custom_objects=custom_objs)
    logging.info(f'{model._name} is loaded.')

    # Model Predictions
    predictions = []
    for batch_x, batch_test in tqdm(test_generator(np.array(data_x), batch_size, output_size,
                                                   elmo, np.array(test_mask_builder), tokenizer, False),
                                    desc="Predicting_Lexes"):
        # Output Shape (batch_size, max_len_per_batch, output_vocab_size)
        batch_pred = model.predict_on_batch([batch_x, batch_test])
        # Output Shape (batch_size, max_len_per_batch)
        y_hat = np.argmax(batch_pred, axis=-1)
        predictions.extend(y_hat)

    # Dictionaries to get mappings
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    _, wordnet_babelnet = build_bn2wn_dict(bn2wn_path)
    bn2lex_path = os.path.join(resources_path, 'babelnet2lexnames.tsv')
    babelnet_lex, _ = build_bn2lex_dict(bn2lex_path)

    # Save predictions to a file
    id_lex_list = []  # stands for predictions in {word_id babelnet_sense}
    for i, sentence in enumerate(tqdm(data_x, desc='Formatting model predictions into ID_LEX')):
        for j, word in enumerate(sentence):
            if len(test_mask_builder[i][j]) == 2:  # So it is an instance
                prediction = predictions[i][j]
                prediction_sense = tokenizer.index_word.get(
                    prediction, '<OOV>')
                if 'wn:' not in prediction_sense or 'bn:' not in prediction_sense:
                    prediction_sense = predict_sense(token=word)  # Fallback
                word_id = test_mask_builder[i][j][1]
                id_ = word_id[word_id.find('.') + 1:]
                bn = wordnet_babelnet.get(prediction_sense, None)
                lex = babelnet_lex.get(bn, 'factotum')
                if id_ is None or lex is None:
                    continue
                id_lex_list.append(f'{id_}\t{lex}')

    with open(output_path, mode="w+") as output_file:
        for id_lex in tqdm(id_lex_list, desc="Writing model predictions"):
            output_file.write(f'{id_lex}\n')


if __name__ == '__main__':
    initialize_logger()
    configure_tf()

    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    resources_path = os.path.join(cwd, 'resources')
    input_path = os.path.join(data_path, 'evaluation',
                              'WSD_Unified_Evaluation_Datasets', 'ALL',
                              'ALL.data.xml')
    output_path = os.path.join(resources_path, 'output.txt')
    predict_babelnet(input_path, output_path, resources_path)

    output_path = os.path.join(resources_path, 'output_dom.txt')
    predict_wordnet_domains(input_path, output_path, resources_path)

    output_path = os.path.join(resources_path, 'output_lex.txt')
    predict_lexicographer(input_path, output_path, resources_path)
