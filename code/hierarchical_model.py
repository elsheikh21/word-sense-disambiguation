import datetime
import os
from argparse import ArgumentParser

import numpy as np
import yaml
from tensorflow.keras.callbacks import (TensorBoard, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (LSTM, Add, Bidirectional,
                                     Dense, Embedding, Input,
                                     Softmax, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

from keras_elmo import ElmoEmbeddingLayer
from model_utils import visualize_plot_mdl, plot_history
from parsing_dataset import load_dataset, multitask_train_generator
from utilities import configure_tf, initialize_logger, save_pickle


def parse_args():
    parser = ArgumentParser(description="WSD")
    parser.add_argument("--model_type", default='baseline', type=str,
                        help="""Choose the model: baseline: BiLSTM Model.
                                attention: Attention Stacked BiLSTM Model.
                                seq2seq: Seq2Seq Attention.""")

    return vars(parser.parse_args())


def baseline_model(vocabulary_size, config_params,
                   output_size, lex_output_size,
                   dom_output_size, tokenizer=None,
                   visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size, name='Input')

    if tokenizer is not None:
        embedding = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
    else:
        embedding_size = int(config_params['embedding_size'])
        embedding = Embedding(input_dim=vocabulary_size,
                              output_dim=embedding_size,
                              mask_zero=True,
                              name="Embeddings")(in_sentences)

    bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                input_shape=(None, None, embedding_size)
                                ),
                           merge_mode='sum')(embedding)

    stacked_bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                        recurrent_dropout=0.2,
                                        return_sequences=True,
                                        input_shape=(None, None, embedding_size)
                                        ),
                                   merge_mode='sum')(bilstm)

    lex_logits = TimeDistributed(Dense(lex_output_size), name='LEX_logits')(bilstm)
    dom_logits = TimeDistributed(Dense(dom_output_size), name='DOM_logits')(bilstm)
    wsd_logits = TimeDistributed(Dense(output_size), name='WSD_logits')(stacked_bilstm)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')
    logits_mask = Add()([wsd_logits, in_mask])

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    lex_output = Softmax(name="LEX_output")(lex_logits)
    dom_output = Softmax(name="DOM_output")(dom_logits)

    model = Model(inputs=[in_sentences, in_mask],
                  outputs=[wsd_output, dom_output, lex_output],
                  name="Hierarchical")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def train_multitask_model(model, dataset, config_params, use_elmo):
    name = model._name
    train_x, train_y = dataset.get('train_x'), dataset.get('train_y')
    output_size = dataset.get('output_size')
    mask_builder = dataset.get('mask_builder')
    tokenizer = dataset.get('tokenizer')
    dom_y = dataset.get("dom_labeled_list")
    lex_y = dataset.get("lex_labeled_list")
    del dataset

    log_dir = os.path.join("logs", "fit_generator_multitask",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoard(log_dir)

    check_dir = os.path.join("checkpoint_multitask", f'{name}.hdf5')
    model_chkpt = ModelCheckpoint(filepath=check_dir, monitor="loss", mode="min",
                                  save_best_only=True, save_weights_only=True, verbose=True)

    early_stopping = EarlyStopping(monitor="loss", patience=3)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6)

    epochs = int(config_params['epochs'])
    batch_size = int(config_params['batch_size'])
    cbks = [logger, model_chkpt, early_stopping, reduce_lr]

    resources_path = os.path.join(os.getcwd(), 'resources')
    try:
        history = model.fit_generator(multitask_train_generator(train_x, train_y,
                                                                dom_y, lex_y,
                                                                batch_size, output_size,
                                                                use_elmo, mask_builder,
                                                                tokenizer, use_bert=False),
                                      verbose=1, epochs=epochs,
                                      steps_per_epoch=np.ceil(len(train_x) / batch_size),
                                      callbacks=cbks)
        history_path = os.path.join(resources_path, f'{name}_history.pkl')
        save_pickle(history_path, history.history)
        plot_history(history, os.path.join(resources_path, f'{name}_history'))
        model.save(os.path.join(resources_path, f'{name}_model.h5'))
        model.save_weights(os.path.join(resources_path, f'{name}_weights.h5'))
        return history
    except KeyboardInterrupt:
        model.save(os.path.join(
            resources_path, f'{name}.h5'))
        model.save_weights(os.path.join(
            resources_path, f'{name}_weights.h5'))


if __name__ == "__main__":
    params = parse_args()
    initialize_logger()

    # Load our config file
    config_file_path = os.path.join(os.getcwd(), "config.yaml")
    config_file = open(config_file_path)
    config_params = yaml.load(config_file)

    # run on CPU, change in Config file.
    if config_params["USE_CPU"]:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        configure_tf()

    elmo = config_params["use_elmo"]
    dataset = load_dataset(elmo=elmo)
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    lex_vocab_size = dataset.get("lex_vocab_size")
    dom_vocab_size = dataset.get("dom_vocab_size")
    tokenizer = dataset.get("tokenizer")

    model = None
    tokenizer = tokenizer if elmo else None

    model = baseline_model(vocabulary_size, config_params,
                           output_size, lex_vocab_size,
                           dom_vocab_size, tokenizer=tokenizer)

    history = train_multitask_model(model, dataset, config_params, elmo)
