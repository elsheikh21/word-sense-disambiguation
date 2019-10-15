import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from predict_utils import f1_m, build_gold, parse_test


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

    # 1. Parse Test set into test_x, test_y and process them
    tokenizer = os.path.join(resources_path, 'tokenizer.pkl')
    input_path_ = os.path.join(input_path, 'ALL.data.xml')
    gold_dict_path = os.path.join(input_path, 'ALL.gold.key.txt')

    gold_dict = build_gold(gold_dict_path)
    test_x, test_y = parse_test(input_path_, gold_dict, tokenizer)
    # TODO: 2. Load masks and dictionaries and hyperparameters

    # Load model and predict
    model_path = os.path.join(resources_path, 'model.h5')
    model = load_model(model_path)
    predictions = model.predict(test_x, verbose=1, workers=0,
                                use_multiprocessing=True)
    y_hat = np.argmax(predictions[0])

    # Save predictions and compute F1 Score
    np.savetxt(output_path, y_hat, fmt="%s")
    f1_score = f1_m(test_y, y_hat)
    print(f"The model F1_Score: {f1_score}")


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
    pass


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
    pass


if __name__ == '__main__':
    cwd = os.getcwd()
    data_path = os.path.join(cwd, 'data')
    resources_path = os.path.join(cwd, 'resources')
    input_path = os.path.join(data_path, 'evaluation',
                              'WSD_Unified_Evaluation_Datasets', 'ALL')
    output_path = os.path.join(data_path, 'output.npy')
    predict_babelnet(input_path, output_path, resources_path)
