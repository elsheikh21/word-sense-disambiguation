import datetime
import logging
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import yaml
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (TensorBoard, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (LSTM, Softmax, Add, Bidirectional, Dense,
                                     Input, TimeDistributed, Reshape,
                                     Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

try:
    from bert.tokenization import FullTokenizer
except ModuleNotFoundError:
    os.system('pip install bert-tensorflow')
    from bert.tokenization import FullTokenizer

from keras_bert import BertEmbeddingLayer
from model_utils import visualize_plot_mdl, plot_history
from parsing_dataset import load_dataset, bert_train_generator, bert_multitask_train_generator
from utilities import configure_workspace, save_pickle


os.environ['TF_KERAS'] = '1'
try:
    from keras_self_attention import SeqSelfAttention
except ModuleNotFoundError:
    os.system('pip install keras_self_attention')
    from keras_self_attention import SeqSelfAttention

from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="WSD")
    parser.add_argument("--model_type", default='baseline', type=str,
                        help="""Choose the model: baseline: BiLSTM Model.
                                attention: Attention Stacked BiLSTM Model.
                                seq2seq: Seq2Seq Attention.""")

    return vars(parser.parse_args())


def baseline_model(output_size, max_seq_len, config_params, visualize=False, plot=False):
    hidden_size = int(config_params.get('hidden_size'))
    embedding_size = 768
    batch_size = int(config_params.get('batch_size'))

    in_id = Input(shape=(max_seq_len, ), name="input_ids")
    in_mask = Input(shape=(max_seq_len, ), name="input_masks")
    in_segment = Input(shape=(max_seq_len, ), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output_ = BertEmbeddingLayer(n_fine_tune_layers=3,
                                      pooling="mean")(bert_inputs)
    bert_output = Reshape((max_seq_len, embedding_size))(bert_output_)

    bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=True
                                ))(bert_output)

    logits = TimeDistributed(Dense(output_size))(bilstm)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')
    bert_inputs.append(in_mask)

    logits_mask = Add()([logits, in_mask])
    output = Softmax()(logits_mask)

    mdl = Model(inputs=bert_inputs, outputs=output, name="Bert_BiLSTM")

    mdl.compile(loss="sparse_categorical_crossentropy",
                optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, mdl)
    logging.info(f'{mdl._name} is created.')

    return mdl


def multitask_baseline_model(output_size, pos_vocab_size, lex_vocab_size,
                             config_params, visualize=False, plot=False):
    embedding_size = 768
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])
    max_seq_len = 512
    
    in_id = Input(shape=(max_seq_len,), name="input_ids")
    in_mask = Input(shape=(max_seq_len,), name="input_masks")
    in_segment = Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output_ = BertEmbeddingLayer(n_fine_tune_layers=3,
                                      pooling="mean")(bert_inputs)
    bert_output = Reshape((max_seq_len, embedding_size))(bert_output_)

    bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                input_shape=(None, None, embedding_size)
                                ),
                           merge_mode='sum')(bert_output)

    logits = TimeDistributed(Dense(output_size))(bilstm)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')
    bert_inputs.append(in_mask)

    logits_mask = Add()([logits, in_mask])

    pos_logits = TimeDistributed(Dense(pos_vocab_size),
                                 name='POS_logits')(bilstm)
    lex_logits = TimeDistributed(Dense(lex_vocab_size),
                                 name='LEX_logits')(bilstm)

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    pos_output = Softmax(name="POS_output")(pos_logits)
    lex_output = Softmax(name="LEX_output")(lex_logits)

    model = Model(inputs=bert_inputs,
                  outputs=[wsd_output, pos_output, lex_output],
                  name='Bert_BiLSTM_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def attention_model(output_size, max_seq_len, config_params, visualize=False, plot=False):
    embedding_size = 768
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    in_id = Input(shape=(max_seq_len,), name="input_ids")
    in_mask = Input(shape=(max_seq_len,), name="input_masks")
    in_segment = Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output_ = BertEmbeddingLayer(n_fine_tune_layers=3,
                                      pooling="mean")(bert_inputs)
    bert_output = Reshape((max_seq_len, embedding_size))(bert_output_)

    bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=True
                                ))(bert_output)
    attention = SeqSelfAttention(attention_activation='sigmoid',
                                 name='Attention')(bilstm)

    logits = TimeDistributed(Dense(output_size))(attention)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')
    bert_inputs.append(in_mask)

    logits_mask = Add()([logits, in_mask])
    output = Softmax()(logits_mask)

    mdl = Model(inputs=bert_inputs, outputs=output, name="Bert_Attention_BiLSTM")

    mdl.compile(loss="sparse_categorical_crossentropy",
                optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, mdl)

    return mdl


def multitask_attention_model(output_size, pos_vocab_size, lex_vocab_size,
                              config_params, visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])
    embedding_size = 768
    max_seq_len = 512

    in_id = Input(shape=(max_seq_len,), name="input_ids")
    in_mask = Input(shape=(max_seq_len,), name="input_masks")
    in_segment = Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output_ = BertEmbeddingLayer(n_fine_tune_layers=3,
                                      pooling="mean")(bert_inputs)
    bert_output = Reshape((max_seq_len, embedding_size))(bert_output_)

    in_mask = Input(shape=(None, output_size), batch_size=batch_size,
                    name='Candidate_Synsets_Mask')
    bert_inputs.append(in_mask)

    bilstm = Bidirectional(LSTM(hidden_size, dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=True,
                                input_shape=(None, None, embedding_size)
                                ),
                           merge_mode='sum')(bert_output)

    attention = SeqSelfAttention(units=128,
                                 attention_activation='sigmoid',
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

    model = Model(inputs=bert_inputs,
                  outputs=[wsd_output, pos_output, lex_output],
                  name='Bert_BiLSTM_ATT_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def seq2seq_model(output_size, max_seq_len, config_params, visualize=False, plot=False):
    drop, rdrop = 0.2, 0.2
    embedding_size = 768
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])

    in_id = Input(shape=(max_seq_len,), name="input_ids")
    in_mask = Input(shape=(max_seq_len,), name="input_masks")
    in_segment = Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output_ = BertEmbeddingLayer(n_fine_tune_layers=3,
                                      pooling="mean")(bert_inputs)
    bert_output = Reshape((max_seq_len, embedding_size))(bert_output_)

    in_mask = Input(shape=(None, output_size),
                    batch_size=batch_size, name='Candidate_Synsets_Mask')

    bert_inputs.append(in_mask)

    encoder_bilstm = Bidirectional(LSTM(hidden_size, dropout=drop,
                                        recurrent_dropout=rdrop,
                                        return_sequences=True,
                                        return_state=True,
                                        input_shape=(
                                            None, None, embedding_size)
                                        ),
                                   merge_mode='sum',
                                   name='Encoder_BiLSTM_1')(bert_output)

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

    model = Model(bert_inputs, outputs=decoder_outputs, name="Bert_Attention_Seq2Seq")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['acc'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def multitask_seq2seq_model(output_size, pos_vocab_size,
                            lex_vocab_size, config_params,
                            visualize=False, plot=False):
    hidden_size = int(config_params['hidden_size'])
    batch_size = int(config_params['batch_size'])
    embedding_size = 768
    max_seq_len = 512
    in_id = Input(shape=(max_seq_len,), name="input_ids")
    in_mask = Input(shape=(max_seq_len,), name="input_masks")
    in_segment = Input(shape=(max_seq_len,), name="segment_ids")
    bert_inputs_ = [in_id, in_mask, in_segment]

    bert_output_ = BertEmbeddingLayer(n_fine_tune_layers=3,
                                      pooling="mean")(bert_inputs_)
    bert_output = Reshape((max_seq_len, embedding_size))(bert_output_)

    input_mask = Input(shape=(None, output_size),
                       batch_size=batch_size, name='Candidate_Synsets_Mask')

    bert_inputs_.append(input_mask)

    bilstm, forward_h, _, backward_h, _ = Bidirectional(LSTM(hidden_size,
                                                             return_sequences=True,
                                                             return_state=True,
                                                             dropout=0.2,
                                                             recurrent_dropout=0.2,
                                                             input_shape=(None, None, embedding_size)
                                                             ),
                                                        merge_mode='sum', name='Encoder_BiLSTM'
                                                        )(bert_output)

    state_h = Concatenate()([forward_h, backward_h])

    context = SeqSelfAttention(units=128)([bilstm, state_h])

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

    logits = TimeDistributed(Dense(output_size), name='WSD_logits')(decoder_bilstm)
    logits_mask = Add(name="Masked_logits")([logits, input_mask])

    pos_logits = TimeDistributed(Dense(pos_vocab_size), name='POS_logits')(decoder_bilstm)
    lex_logits = TimeDistributed(Dense(lex_vocab_size), name='LEX_logits')(decoder_bilstm)

    wsd_output = Softmax(name="WSD_output")(logits_mask)
    pos_output = Softmax(name="POS_output")(pos_logits)
    lex_output = Softmax(name="LEX_output")(lex_logits)

    model = Model(inputs=bert_inputs_, outputs=[wsd_output, pos_output, lex_output],
                  name='Bert_Attention_Seq2Seq_MultiTask')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adadelta(), metrics=['accuracy'])

    visualize_plot_mdl(visualize, plot, model)

    return model


def initialize_vars(session):
    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    K.set_session(session)


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  batches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The un-tokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The un-tokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_single_example(tokenizer, example, max_seq_length=512):
    """Converts a single InputExample into a single InputFeatures."""
    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=512):
    """Convert a set of InputExamples to a list of InputFeatures."""
    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc='Extracting features'):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids).astype(np.int32),
        np.array(input_masks).astype(np.int32),
        np.array(segment_ids).astype(np.int32),
        np.array(labels),
    )


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(
                guid=None, text_a=" ".join(text), text_b=None, label=label
            )
        )
    return InputExamples


def create_tokenizer_from_hub_module(bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(
        signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def train_model(model, dataset, config_params, use_elmo, shuffle=False):
    logging.info(f'Start training the {model._name} model.')
    name = model._name
    train_x, train_y = dataset.get('train_x'), dataset.get('train_y')
    output_size = dataset.get('output_size')
    mask_builder = dataset.get('mask_builder')
    tokenizer = dataset.get('tokenizer')
    del dataset

    log_dir = os.path.join("logs", "fit_generator",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoard(log_dir)

    check_dir = os.path.join("checkpoint", f'{name}.hdf5')
    model_chkpt = ModelCheckpoint(filepath=check_dir, monitor="loss", mode="min",
                                  save_best_only=True, save_weights_only=True, verbose=True)

    early_stopping = EarlyStopping(monitor="loss", patience=3)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6)

    epochs = int(config_params['epochs'])
    batch_size = int(config_params['batch_size'])
    cbks = [logger, model_chkpt, early_stopping, reduce_lr]

    resources_path = os.path.join(os.getcwd(), 'resources')
    try:
        history = model.fit_generator(bert_train_generator(train_x, train_y,
                                                           batch_size, output_size,
                                                           False, mask_builder,
                                                           tokenizer, shuffle=shuffle),
                                      verbose=1, epochs=epochs,
                                      steps_per_epoch=np.ceil(len(train_x[0]) / batch_size),
                                      callbacks=cbks)
        history_path = os.path.join(
            resources_path, f'{name}_history.pkl')
        save_pickle(history_path, history.history)
        model.save(os.path.join(resources_path, f'{name}_model.h5'))
        plot_history(history, os.path.join(
            resources_path, f'{name}_history'))
        model.save_weights(os.path.join(
            resources_path, f'{name}_weights.h5'))
        return history
    except KeyboardInterrupt:
        model.save(os.path.join(
            resources_path, f'{name}_model.h5'))
        model.save_weights(os.path.join(
            resources_path, f'{name}_weights.h5'))


def train_multitask_model(model, dataset, config_params, use_elmo=False, shuffle=False):
    logging.info(f'Start training the {model._name} model.')
    name = model._name
    train_x, train_y = dataset.get('train_x'), dataset.get('train_y')
    output_size = dataset.get('output_size')
    mask_builder = dataset.get('mask_builder')
    tokenizer = dataset.get('tokenizer')
    lex_y = dataset.get('lex_labeled_list')
    pos_y = dataset.get('pos_labeled_list')
    del dataset

    log_dir = os.path.join("logs", "fit_generator",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoard(log_dir)

    check_dir = os.path.join("checkpoint", f'{name}.hdf5')
    model_chkpt = ModelCheckpoint(filepath=check_dir, monitor="loss", mode="min",
                                  save_best_only=True, save_weights_only=True, verbose=True)

    early_stopping = EarlyStopping(monitor="loss", patience=3)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6)

    epochs = int(config_params['epochs'])
    batch_size = int(config_params['batch_size'])
    cbks = [logger, model_chkpt, early_stopping, reduce_lr]

    resources_path = os.path.join(os.getcwd(), 'resources')
    try:
        history = model.fit_generator(bert_multitask_train_generator(train_x, train_y, pos_y, lex_y,
                                                                     batch_size, output_size,
                                                                     use_elmo=use_elmo, mask_builder=mask_builder,
                                                                     tokenizer=tokenizer, shuffle=shuffle),
                                      verbose=1, epochs=epochs,
                                      steps_per_epoch=np.ceil(len(train_x[0]) / batch_size),
                                      callbacks=cbks)
        history_path = os.path.join(
            resources_path, f'{name}_history.pkl')
        save_pickle(history_path, history.history)
        plot_history(history, os.path.join(
            resources_path, f'{name}_history'))
        model.save(os.path.join(
            resources_path, f'{name}_model.h5'))
        model.save_weights(os.path.join(
            resources_path, f'{name}_weights.h5'))
        return history
    except KeyboardInterrupt:
        model.save(os.path.join(
            resources_path, f'{name}_model.h5'))
        model.save_weights(os.path.join(
            resources_path, f'{name}_weights.h5'))


def process_bert_data(max_seq_len=512):
    elmo = config_params["use_elmo"]
    dataset = load_dataset(elmo=elmo)

    # Parse data in Bert format
    train_x = dataset.get("train_x")

    train_text = []
    for example in train_x:
        train_text.append(" ".join(str(n) for n in example))

    train_text = [' '.join(t.split()[0:max_seq_len]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    # print(train_text.shape)  # (37_184, 1)
    train_labels = dataset.get("train_y")

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()
    logging.info('Tokenizer is created from TF_Hub')

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_labels)

    # Extract features
    (train_input_ids, train_input_masks,
     train_segment_ids, train_labels) = convert_examples_to_features(tokenizer,
                                                                     train_examples,
                                                                     max_seq_length=max_seq_len)

    bert_inputs = [train_input_ids, train_input_masks, train_segment_ids]

    dataset.update(train_x=bert_inputs, train_y=train_labels, tokenizer=tokenizer)
    return dataset


if __name__ == '__main__':
    # Initialize session
    sess = tf.Session()
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.INFO)

    params = parse_args()

    config_params = configure_workspace()
    elmo = config_params["use_elmo"]

    dataset = process_bert_data()
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    pos_vocab_size = dataset.get("pos_vocab_size")
    lex_vocab_size = dataset.get("lex_vocab_size")

    model = multitask_attention_model(output_size, pos_vocab_size, lex_vocab_size, config_params)
    # max_seq_len = 512
    # model = attention_model(output_size, max_seq_len, config_params)

    # Instantiate variables
    initialize_vars(sess)

    history = train_multitask_model(model, dataset, config_params, use_elmo=elmo)
    # history = train_model(model, dataset, config_params, elmo)
