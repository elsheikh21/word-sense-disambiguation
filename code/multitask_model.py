import os
from argparse import ArgumentParser

import yaml

os.environ['TF_KERAS'] = '1'
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import (LSTM, Add, Bidirectional, Concatenate,
                                     Dense, Embedding, Input, Softmax,
                                     TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

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

    params = vars(parser.parse_args())

    return params


def baseline_model(vocabulary_size, config_params,
                   pos_vocab_size, lex_vocab_size,
                   output_size, tokenizer=None,
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
                  name='BiLSTM_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=["acc"])

    visualize_plot_mdl(visualize, plot, model)

    return model


def attention_model(vocabulary_size, config_params, output_size,
                    pos_vocab_size, lex_vocab_size,
                    depth=2, visualize=False,
                    plot=False, tokenizer=None):
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

    model = Model(inputs=[in_sentences, logits_mask],
                  outputs=[wsd_output, pos_output, lex_output],
                  name='BiLSTM_ATT_MultiTask')

    visualize_plot_mdl(visualize, plot, model)

    return model


def seq2seq_model(vocabulary_size, config_params, output_size,
                  pos_vocab_size, lex_vocab_size, tokenizer=None,
                  visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])
    embedding_size = int(config_params['embedding_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size, name='Input')

    if tokenizer is not None:
        embeddings = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
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

    encoder_attention = SeqSelfAttention(attention_activation='sigmoid',
                                         name='Attention')([bilstm, state_h])

    concat = Concatenate()([encoder_attention, bilstm])

    decoder_fwd_lstm, _, _ = LSTM(hidden_size, dropout=0.2,
                                  recurrent_dropout=0.2,
                                  return_sequences=True,
                                  input_shape=(None, None, embedding_size),
                                  name='Decoder_FWD_LSTM')(concat)

    decoder_bck_lstm, _, _ = LSTM(hidden_size,
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

    logits_mask = Add(name="Masked logits")([logits, in_mask])
    pos_logits = TimeDistributed(Dense(pos_vocab_size),
                                 name='POS_logits')(decoder_bilstm)
    lex_logits = TimeDistributed(Dense(lex_vocab_size),
                                 name='LEX_logits')(decoder_bilstm)

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    pos_output = Softmax(name="POS_output")(pos_logits)
    lex_output = Softmax(name="LEX_output")(lex_logits)

    model = Model(inputs=[in_sentences, logits_mask],
                  outputs=[wsd_output, pos_output, lex_output],
                  name='Seq2Seq_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=["acc"])

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

    elmo = config_params["use_elmo"]
    dataset = load_dataset(elmo=elmo)
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    tokenizer = dataset.get("tokenizer")
    pos_vocab_size = dataset.get("pos_vocab_size")
    lex_vocab_size = dataset.get("lex_vocab_size")

    model = None
    tokenizer = tokenizer if elmo else None

    if params["model_type"] == "baseline":
        model = baseline_model(vocabulary_size, config_params,
                               pos_vocab_size, lex_vocab_size,
                               output_size, tokenizer=tokenizer)
        history = train_multitask_model(model, dataset, config_params, elmo)
    elif params["model_type"] == "attention":
        attention_model = attention_model(vocabulary_size, config_params,
                                          pos_vocab_size, lex_vocab_size,
                                          output_size, tokenizer=tokenizer)
        attention_history = train_multitask_model(attention_model, dataset,
                                                  config_params, elmo)
    elif params["model_type"] == "seq2seq":
        seq2seq_model = seq2seq_model(vocabulary_size, config_params,
                                      pos_vocab_size, lex_vocab_size,
                                      output_size, tokenizer=tokenizer)
        seq2seq_history = train_multitask_model(seq2seq_model, dataset,
                                                config_params, elmo)
