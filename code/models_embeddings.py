import logging
import os
from argparse import ArgumentParser

import numpy as np
import yaml
from tensorflow.keras.layers import (LSTM, Add, Bidirectional,
                                     Dense, Embedding, Input,
                                     Softmax, TimeDistributed,
                                     Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tqdm import tqdm

os.environ['TF_KERAS'] = '1'
try:
    from keras_self_attention import SeqSelfAttention
except ModuleNotFoundError:
    os.system('pip install keras_self_attention')
    from keras_self_attention import SeqSelfAttention
from keras_attention import Attention
from keras_elmo import ElmoEmbeddingLayer
from model_utils import visualize_plot_mdl
from parsing_dataset import load_dataset
from train_multitask import train_multitask_model
from utilities import configure_tf, initialize_logger


def parse_args():
    parser = ArgumentParser(description="WSD")
    parser.add_argument("--model_type", default='baseline', type=str,
                        help="""Choose the model: baseline: BiLSTM Model.
                                attention: Attention Stacked BiLSTM Model.
                                seq2seq: Seq2Seq Attention.""")

    return vars(parser.parse_args())


def pretrained_embeddings(vocab_dict, embedding_size, embbeddings_file):
    # first, build index mapping words in the embeddings set, to their embedding vector
    # Gets pretrained embeddings file path
    pretrained_embeddings_path = os.path.join(os.getcwd(), 'resources', embbeddings_file)
    # Creates a dict
    embeddings_index = dict()
    # Opens the utf-8 encoded file, loops on every line
    # and assign the value as per char
    with open(pretrained_embeddings_path, encoding='utf-8', mode='r') as f:
        for line in tqdm(f, desc='Loading embeddings dict'):
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float64')
            embeddings_index[word] = coeffs

    # For our vocabulary size
    vocab_size = len(vocab_dict)

    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, index in tqdm(vocab_dict.items(),
                            desc='Creating embeddings matrix'):
        if index > vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros
                embedding_matrix[index] = embedding_vector

    logging.info('Pretrained Embeddings are created.')
    # Returns embeddings matrix
    return embedding_matrix


def baseline_model(vocabulary_size, config_params, output_size, weights=None, tokenizer=None, visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size, name='Input')

    embedding_size = int(config_params['embedding_size'])

    if tokenizer is not None:
        embedding = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
    elif weights is not None:
        embedding_size = weights.shape[1]
        train = True  # To fine-tune pretrained embeddings or not
        embedding = Embedding(input_dim=output_size, output_dim=embedding_size,
                              weights=[weights], trainable=train,
                              mask_zero=True)(in_sentences)
    else:
        embedding = Embedding(input_dim=vocabulary_size,
                              output_dim=embedding_size,
                              mask_zero=True)(in_sentences)

    bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                input_shape=(None, None, embedding_size)
                                ),
                           merge_mode='sum')(embedding)

    logits = TimeDistributed(Dense(output_size))(bilstm)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')

    logits_mask = Add()([logits, in_mask])
    output = Softmax()(logits_mask)

    model = Model(inputs=[in_sentences, in_mask], outputs=output, name="SensEmbed_Baseline")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def multitask_baseline_model(vocabulary_size, config_params, output_size, pos_vocab_size, lex_vocab_size, weights=None, tokenizer=None, visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size, name='Input')

    if tokenizer is not None:
        embedding = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
    elif weights is not None:
        embedding_size = weights.shape[1]
        train = True  # To fine-tune pretrained embeddings or not
        embedding = Embedding(input_dim=output_size, output_dim=embedding_size,
                              weights=[weights], trainable=train,
                              mask_zero=True)(in_sentences)
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

    logits = TimeDistributed(Dense(output_size))(bilstm)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')

    logits_mask = Add()([logits, in_mask])

    pos_logits = TimeDistributed(Dense(pos_vocab_size),
                                 name='POS_logits')(bilstm)
    lex_logits = TimeDistributed(Dense(lex_vocab_size),
                                 name='LEX_logits')(bilstm)

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    pos_output = Softmax(name="POS_output")(pos_logits)
    lex_output = Softmax(name="LEX_output")(lex_logits)

    model = Model(inputs=[in_sentences, in_mask],
                  outputs=[wsd_output, pos_output, lex_output],
                  name='SensEmbed_BiLSTM_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def attention_model(vocabulary_size, config_params, output_size, weights=None, tokenizer=None, visualize=False, plot=False):
    hidden_size = config_params['hidden_size']
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size)
    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')

    if tokenizer is not None:
        embedding = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
    elif weights is not None:
        embedding_size = weights.shape[1]
        train = True  # To fine-tune pretrained embeddings or not
        embedding = Embedding(input_dim=output_size, output_dim=embedding_size,
                              weights=[weights], trainable=train,
                              mask_zero=True)(in_sentences)
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

    attention = SeqSelfAttention(attention_activation='sigmoid',
                                 name='Attention')(bilstm)

    logits = TimeDistributed(Dense(output_size))(attention)
    logits_mask = Add()([logits, in_mask])

    output = Softmax()(logits_mask)

    model = Model(inputs=[in_sentences, in_mask], outputs=output, name="SensEmbed_Attention")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def multitask_attention_model(vocabulary_size, config_params, output_size, pos_vocab_size, lex_vocab_size,  weights=None, tokenizer=None, visualize=False, plot=False):
    hidden_size = config_params['hidden_size']
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size)
    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')

    if tokenizer is not None:
        embedding = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
    elif weights is not None:
        embedding_size = weights.shape[1]
        train = True  # To fine-tune pretrained embeddings or not
        embedding = Embedding(input_dim=output_size, output_dim=embedding_size,
                              weights=[weights], trainable=train,
                              mask_zero=True)(in_sentences)
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

    attention = SeqSelfAttention(attention_activation='sigmoid',
                                 name='Attention')(bilstm)

    logits = TimeDistributed(Dense(output_size))(attention)
    logits_mask = Add()([logits, in_mask])

    pos_logits = TimeDistributed(Dense(pos_vocab_size),
                                 name='POS_logits')(attention)
    lex_logits = TimeDistributed(Dense(lex_vocab_size),
                                 name='LEX_logits')(attention)

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    pos_output = Softmax(name="POS_output")(pos_logits)
    lex_output = Softmax(name="LEX_output")(lex_logits)

    model = Model(inputs=[in_sentences, in_mask],
                  outputs=[wsd_output, pos_output, lex_output],
                  name='SensEmbed_BiLSTM_ATT_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def seq2seq_model(vocabulary_size, config_params, output_size, weights=None, tokenizer=None, visualize=False, plot=False):
    drop, rdrop = 0.2, 0.2
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    encoder_inputs = Input(shape=(None,), dtype=input_type,
                           batch_size=batch_size)
    in_mask = Input(shape=(None, output_size),
                    batch_size=batch_size, name='Candidate_Synsets_Mask')

    if tokenizer is not None:
        encoder_embeddings = ElmoEmbeddingLayer()(encoder_inputs)
        embedding_size = 1024
    elif weights is not None:
        embedding_size = weights.shape[1]
        train = True  # To fine-tune pretrained embeddings or not
        encoder_embeddings = Embedding(input_dim=output_size, output_dim=embedding_size,weights=[weights], trainable=train, mask_zero=True)(encoder_inputs)
    else:
        embedding_size = int(config_params['embedding_size'])
        encoder_embeddings = Embedding(
            input_dim=vocabulary_size, output_dim=embedding_size,
            mask_zero=True, name="Embeddings")(encoder_inputs)

    encoder_bilstm = Bidirectional(LSTM(hidden_size, dropout=drop,
                                        recurrent_dropout=rdrop,
                                        return_sequences=True,
                                        return_state=True,
                                        input_shape=(
                                            None, None, embedding_size)
                                        ),
                                   merge_mode='sum',
                                   name='Encoder_BiLSTM_1')(encoder_embeddings)

    encoder_bilstm2 = Bidirectional(LSTM(hidden_size, dropout=drop,
                                         recurrent_dropout=rdrop,
                                         return_sequences=True,
                                         return_state=True,
                                         input_shape=(
                                             None, None, embedding_size)
                                         ),
                                    merge_mode='sum', name='Encoder_BiLSTM_2')

    (encoder_outputs, forward_h, forward_c, backward_h,
     backward_c) = encoder_bilstm2(encoder_bilstm)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    encoder_attention = SeqSelfAttention(
        attention_activation='sigmoid', name='Attention')(encoder_outputs)

    decoder_fwd_lstm, _, _ = LSTM(hidden_size, dropout=drop,
                                  recurrent_dropout=rdrop,
                                  return_sequences=True,
                                  return_state=True,
                                  input_shape=(None, None, embedding_size),
                                  name='Decoder_FWD_LSTM')(encoder_attention,
                                                           initial_state=[forward_h, backward_h])

    decoder_bck_lstm, _, _ = LSTM(hidden_size,
                                  dropout=drop,
                                  recurrent_dropout=rdrop,
                                  return_sequences=True,
                                  return_state=True,
                                  input_shape=(None, None, embedding_size),
                                  go_backwards=True,
                                  name='Decoder_BWD_LSTM')(decoder_fwd_lstm)

    decoder_bilstm = Concatenate()([decoder_fwd_lstm, decoder_bck_lstm])

    decoder_output = TimeDistributed(Dense(output_size),
                                     name='TimeDist_Dense')(decoder_bilstm)

    logits_mask = Add()([decoder_output, in_mask])

    decoder_outputs = Softmax()(logits_mask)

    model = Model([encoder_inputs, in_mask], outputs=decoder_outputs, name="SensEmbed_Seq2Seq_Attention")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def multitask_seq2seq_model(vocabulary_size, config_params,
                            output_size, pos_vocab_size,
                            lex_vocab_size, weights=None,
                            tokenizer=None, visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])
    embedding_size = int(config_params['embedding_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size, name='Input')

    if tokenizer is not None:
        embeddings = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
    elif weights is not None:
        embedding_size = weights.shape[1]
        train = True  # To fine-tune pretrained embeddings or not
        embeddings = Embedding(input_dim=output_size,
                               output_dim=embedding_size,
                               weights=[weights], trainable=train, mask_zero=False)(in_sentences)
    else:
        embeddings = Embedding(input_dim=vocabulary_size,
                               output_dim=embedding_size,
                               mask_zero=True,
                               name="Embeddings")(in_sentences)

    bilstm, forward_h, _, backward_h, _ = Bidirectional(LSTM(hidden_size, return_sequences=True,
                                                             return_state=True, dropout=0.2, recurrent_dropout=0.2,
                                                             input_shape=(None, None, embedding_size)),
                                                        merge_mode='sum',
                                                        name='Encoder_BiLSTM')(embeddings)
    state_h = Concatenate()([forward_h, backward_h])

    context, _ = Attention(hidden_size)([bilstm, state_h])

    concat = Concatenate()([bilstm, context])

    decoder_fwd_lstm = LSTM(hidden_size, dropout=0.2,
                            recurrent_dropout=0.2,
                            return_sequences=True,
                            input_shape=(None, None, embedding_size),
                            name='Decoder_FWD_LSTM')(concat)

    decoder_bck_lstm = LSTM(hidden_size,
                            dropout=0.2,
                            recurrent_dropout=0.2,
                            return_sequences=True,
                            input_shape=(None, None, embedding_size),
                            go_backwards=True,
                            name='Decoder_BWD_LSTM')(decoder_fwd_lstm)

    decoder_bilstm = Concatenate()([decoder_fwd_lstm, decoder_bck_lstm])

    logits = TimeDistributed(
        Dense(output_size), name='WSD_logits')(decoder_bilstm)

    in_mask = Input(shape=(None, output_size),
                    batch_size=batch_size, name='Candidate_Synsets_Mask')
    logits_mask = Add(name="Masked_logits")([logits, in_mask])

    pos_logits = TimeDistributed(Dense(pos_vocab_size),
                                 name='POS_logits')(decoder_bilstm)
    lex_logits = TimeDistributed(Dense(lex_vocab_size),
                                 name='LEX_logits')(decoder_bilstm)

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    pos_output = Softmax(name="POS_output")(pos_logits)
    lex_output = Softmax(name="LEX_output")(lex_logits)

    model = Model(inputs=[in_sentences, in_mask],
                  outputs=[wsd_output, pos_output, lex_output],
                  name='SensEmbed_Seq2Seq_Attention_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


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

    load_embeddings = config_params["load_embeddings"]
    elmo = config_params["use_elmo"] if not load_embeddings else False
    dataset = load_dataset(elmo=elmo)
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    tokenizer = dataset.get("tokenizer")
    pos_vocab_size = dataset.get("pos_vocab_size")
    lex_vocab_size = dataset.get("lex_vocab_size")

    model = None
    tokenizer = tokenizer if elmo else None
    pretrained_embeddings = pretrained_embeddings(tokenizer.word_index,
                                                  embedding_size=200, embbeddings_file='glove.twitter.27B.200d.txt')
    """
    To Choose either Sense Embeddings or Glove Embeddings:

    embedding_size=400, embbeddings_file='sense_embeddings.vec'
    embedding_size=200, embbeddings_file='glove.twitter.27B.200d.txt'
    """

    model = multitask_seq2seq_model(vocabulary_size, config_params,
                                    output_size, pos_vocab_size, lex_vocab_size,
                                    weights=pretrained_embeddings, tokenizer=tokenizer)

    history = train_multitask_model(model, dataset, config_params, use_elmo=elmo)

    """
    model = baseline_model(vocabulary_size, config_params, output_size,
                           weights=pretrained_embeddings, tokenizer=None,
                           visualize=True, plot=True)
    history = train_model(model, dataset, config_params, elmo, shuffle=True)
    """
