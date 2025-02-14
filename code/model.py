import os
from argparse import ArgumentParser

import yaml

os.environ['TF_KERAS'] = '1'
try:
    from keras_self_attention import SeqSelfAttention
except ModuleNotFoundError:
    os.system('pip install keras_self_attention')
    from keras_self_attention import SeqSelfAttention
from tensorflow.keras.layers import (LSTM, Add, Bidirectional,
                                     Concatenate, Dense, Embedding, Input,
                                     Softmax, TimeDistributed)

from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.models import Model

from keras_elmo import ElmoEmbeddingLayer
from model_utils import visualize_plot_mdl
from parsing_dataset import load_dataset
from utilities import configure_tf, initialize_logger
from train import train_model


def parse_args():
    parser = ArgumentParser(description="WSD")
    parser.add_argument("--model_type", default='baseline', type=str,
                        help="""Choose the model: baseline: BiLSTM Model.
                                attention: Attention Stacked BiLSTM Model.
                                seq2seq: Seq2Seq Attention.""")

    return vars(parser.parse_args())


def baseline_model(vocabulary_size, config_params,
                   output_size, tokenizer=None,
                   visualize=False, plot=False):
    name = 'Baseline'
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    input_type = 'string' if tokenizer is not None else None
    in_sentences = Input(shape=(None,), dtype=input_type,
                         batch_size=batch_size, name='Input')

    if tokenizer is not None:
        embedding = ElmoEmbeddingLayer()(in_sentences)
        embedding_size = 1024
        name = f'Elmo_{name}'
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
    output = Softmax()(logits_mask)

    model = Model(inputs=[in_sentences, in_mask], outputs=output, name=name)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def attention_model(vocabulary_size, config_params, output_size,
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

    output = Softmax()(logits_mask)

    model = Model(inputs=[in_sentences, in_mask], outputs=output, name="Attention")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def seq2seq_model(vocabulary_size, config_params, output_size,
                  tokenizer=None, visualize=False, plot=False):
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

    model = Model([encoder_inputs, in_mask], outputs=decoder_outputs, name="Seq2Seq_Attention")

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

    elmo = config_params["use_elmo"]
    dataset = load_dataset(elmo=elmo)
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    tokenizer = dataset.get("tokenizer")

    model = None
    tokenizer = tokenizer if elmo else None

    if params["model_type"] == "baseline":
        model = baseline_model(vocabulary_size, config_params,
                               output_size, tokenizer=tokenizer)
        history = train_model(model, dataset, config_params, elmo, shuffle=True)
    elif params["model_type"] == "attention":
        attention_model = attention_model(vocabulary_size,
                                          config_params, output_size,
                                          tokenizer=tokenizer)
        attention_history = train_model(attention_model, dataset, config_params, elmo, shuffle=True)
    elif params["model_type"] == "seq2seq":
        seq2seq_model = seq2seq_model(vocabulary_size, config_params,
                                      output_size, tokenizer=tokenizer)
        seq2seq_history = train_model(seq2seq_model, dataset, config_params, elmo, shuffle=True)
