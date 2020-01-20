import logging
import os

import numpy as np

try:
    os.environ['TF_KERAS'] = '1'
    from keras_self_attention import SeqSelfAttention
except:
    os.system('pip install keras_self_attention')
    os.environ['TF_KERAS'] = '1'
    from keras_self_attention import SeqSelfAttention

from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import load_model
from tqdm import tqdm

from predict_utils import (parse_test, test_generator,
                           predict_multilingual_sense)
from utilities import (get_lemma2synsets, load_pickle,
                       build_bn2wn_dict, build_dict, configure_workspace)


def predict_multilingual(input_path: str, output_path: str, resources_path: str, lang: str) -> None:
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
    :param lang: the language of the dataset specified in input_path
    :return: None
    """
    # load the model
    model_path = os.path.join(resources_path, 'SensEmbed_BiLSTM_ATT_MultiTask_model.h5')
    model = load_model(model_path, custom_objects={'SeqSelfAttention': SeqSelfAttention})
    logging.info(f'{model._name} is loaded.')

    # load tokenizer, fetch our vocabulary size
    tokenizer_path = os.path.join(resources_path, 'multilingual_tokenizer.pkl')
    tokenizer = load_pickle(tokenizer_path)

    word_tokens = [word for word in tokenizer.word_index if 'bn:' not in word]
    sense_tokens = [word for word in tokenizer.word_index if 'bn:' in word]
    vocabulary_size = len(word_tokens)
    output_size = vocabulary_size + len(sense_tokens)

    batch_size = 8  # hard coded; as this was the one worked on Colab Google

    # Parse the testing dataset
    gold_dict_path = input_path.replace("data.xml", "gold.key.txt")
    gold_dict = build_dict(gold_dict_path)
    data_x, mask_x = parse_test(input_path, tokenizer=tokenizer, gold_dict=gold_dict, batch_size=batch_size)

    # Getting the model predictions
    predictions = []
    for batch_x, batch_mask in tqdm(test_generator(np.array(data_x), batch_size, output_size,
                                                   use_elmo=False, mask_builder=np.array(mask_x),
                                                   tokenizer=tokenizer, use_bert=False),
                                    desc="Predicting Senses"):
        # Output Shape (batch_size, max_len_per_batch, output_vocab_size)
        batch_pred = model.predict_on_batch([batch_x, batch_mask])
        y_hat = np.argmax(batch_pred[0], axis=-1)
        predictions.extend(y_hat)

    # load lemma2synsets
    lemma2synsets_file_path = os.path.join(os.getcwd(), 'resources', 'lemma2synsets4.0.xx.wn.ALL.txt')
    lemma_synsets = get_lemma2synsets(lemma2synsets_file_path)

    # load wordnet 2 babelnet synsets' mapping
    bn2wn_path = os.path.join(resources_path, "babelnet2wordnet.tsv")
    _, wordnet_babelnet_ = build_bn2wn_dict(bn2wn_path)

    # Save predictions to a file
    id_bn_list = []
    # stands for predictions in {word_id babelnet_sense}
    _predictions = []
    for i, sentence in enumerate(tqdm(data_x, desc="Preparing models' predictions")):
        for j, word in enumerate(sentence):
            if len(mask_x[i][j]) == 2:  # So it is an instance
                prediction = predictions[i][j]
                prediction_sense_ = tokenizer.index_word.get(prediction, '<OOV>')
                if 'bn:' not in prediction_sense_:
                    # Fallback Strategy
                    prediction_sense = predict_multilingual_sense(word=word, word2idx=tokenizer.word_index,
                                                                  lemma_synsets=lemma_synsets,
                                                                  wordnet_babelnet=wordnet_babelnet_)
                else:
                    prediction_sense = prediction_sense_[prediction_sense_.find('bn:'):]
                word_id = mask_x[i][j][1]
                bn = prediction_sense if prediction_sense is not None else '<OOV>'
                if word_id is None or bn is None:
                    continue
                id_bn_list.append(f'{word_id}\t{bn}')
                _predictions.append(bn)

    # Writing model predictions
    with open(output_path, encoding='utf-8', mode="w+") as output_file:
        for id_bn in tqdm(id_bn_list, desc="Writing model predictions"):
            output_file.write(f'{id_bn}\n')

    # Fetching the ground truth of the data
    ground_truth = []
    ground_truth_path = input_path.replace("data.xml", "gold.key.txt")
    with open(ground_truth_path, encoding='utf-8', mode='r') as ground_truth_file:
        lines = ground_truth_file.read().splitlines()
        for line in lines:
            sense_key = line.split()[1]
            ground_truth.append(sense_key)

    # Compute F1_Score
    _, _, f1score, _ = precision_recall_fscore_support(ground_truth, _predictions, average='micro')
    print(f'{model._name} F1_score: {f1score}')


if __name__ == '__main__':
    configure_workspace()
    input_path = os.path.join(os.getcwd(), 'data', 'evaluation',
                              'multilingual_eval', 'semeval2013.de.data.xml')
    output_path = os.path.join(os.getcwd(), 'resources', 'multilingual_output.txt')
    resources_path = os.path.join(os.getcwd(), 'resources')
    lang = 'en'
    predict_multilingual(input_path, output_path, resources_path, lang)
