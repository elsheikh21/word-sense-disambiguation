import tensorflow as tf

import model_bert as mb
import models_embeddings as me
import multitask_model as mt
from model import baseline_model
from train import train_model
from train_multitask import train_multitask_model


def build_train_model(parser_params, config_params, dataset):
    """
    Models are to be choose from the parser parameters, for single task models
    the user can choose either between models with Embeddings to be trained, ElMo,
    Bert, Glove, SensEmbeddings.

    :param parser_params:
    :param config_params:
    :param dataset:

    :return:
    """
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    tokenizer = dataset.get("tokenizer")
    elmo = config_params["use_elmo"]

    model_type = parser_params["model_type"]

    if model_type == "baseline":
        model = baseline_model(vocabulary_size, config_params,
                               output_size, tokenizer=tokenizer)
        _ = train_model(model, dataset, config_params, elmo, shuffle=True)
    elif model_type == "attention":
        attention_model = attention_model(vocabulary_size,
                                          config_params, output_size,
                                          tokenizer=tokenizer)
        _ = train_model(attention_model, dataset,
                        config_params, elmo, shuffle=True)
    elif model_type == "seq2seq":
        seq2seq_model = seq2seq_model(vocabulary_size, config_params,
                                      output_size, tokenizer=tokenizer)
        _ = train_model(seq2seq_model, dataset,
                        config_params, elmo, shuffle=True)
    elif model_type == "sense_embed_baseline":
        tokenizer = tokenizer if elmo else None
        embeddings_dim = parser_params['embeddings_dimensions']
        embeddings_file = parser_params['embeddings_file']
        pretrained_embeddings = me.pretrained_embeddings(tokenizer.word_index,
                                                         embedding_size=embeddings_dim,
                                                         embbeddings_file=embeddings_file)

        sensembed_model = me.baseline_model(vocabulary_size, config_params, output_size,
                                            weights=pretrained_embeddings, tokenizer=None,
                                            visualize=False, plot=False)

        _ = train_model(sensembed_model, dataset,
                        config_params, elmo, shuffle=True)
    elif model_type == "sense_embed_attention":
        tokenizer = tokenizer if elmo else None
        embeddings_dim = parser_params['embeddings_dimensions']
        embeddings_file = parser_params['embeddings_file']
        pretrained_embeddings = me.pretrained_embeddings(tokenizer.word_index,
                                                         embedding_size=embeddings_dim,
                                                         embbeddings_file=embeddings_file)

        sensembed_atten_model = me.attention_model(vocabulary_size, config_params, output_size,
                                                   weights=pretrained_embeddings, tokenizer=None,
                                                   visualize=False, plot=False)

        _ = train_model(sensembed_atten_model, dataset,
                        config_params, elmo, shuffle=True)
    elif model_type == "sense_embed_seq2seq":
        tokenizer = tokenizer if elmo else None
        embeddings_dim = parser_params['embeddings_dimensions']
        embeddings_file = parser_params['embeddings_file']
        pretrained_embeddings = me.pretrained_embeddings(tokenizer.word_index,
                                                         embedding_size=embeddings_dim,
                                                         embbeddings_file=embeddings_file)

        sensembed_seq2seq_model = me.seq2seq(vocabulary_size, config_params, output_size,
                                             weights=pretrained_embeddings, tokenizer=None,
                                             visualize=False, plot=False)

        _ = train_model(sensembed_seq2seq_model, dataset,
                        config_params, elmo, shuffle=True)
    elif model_type == "bert_baseline":
        sess = tf.Session()
        max_seq_len = parser_params['bert_max_seq_len']
        dataset = mb.process_bert_data(max_seq_len)
        model = mb.baseline_model(output_size, max_seq_len, config_params)
        mb.initialize_vars(sess)
        _ = train_model(model, dataset, config_params, elmo)
    elif model_type == "bert_attention":
        sess = tf.Session()
        max_seq_len = parser_params['bert_max_seq_len']
        dataset = mb.process_bert_data(max_seq_len)
        model = mb.attention_model(output_size, max_seq_len, config_params)
        mb.initialize_vars(sess)
        _ = train_model(model, dataset, config_params, elmo)
    elif model_type == "bert_seq2seq":
        sess = tf.Session()
        max_seq_len = parser_params['bert_max_seq_len']
        dataset = mb.process_bert_data(max_seq_len)
        model = mb.seq2seq_model(output_size, max_seq_len, config_params)
        mb.initialize_vars(sess)
        _ = train_model(model, dataset, config_params, elmo)


def build_train_multitask_model(parser_params, config_params, dataset):
    vocabulary_size = dataset.get("vocabulary_size")
    output_size = dataset.get("output_size")
    tokenizer = dataset.get("tokenizer")
    pos_vocab_size = dataset.get("pos_vocab_size")
    lex_vocab_size = dataset.get("lex_vocab_size")
    elmo = config_params["use_elmo"]

    model_type = parser_params["model_type"]
    if model_type == "baseline":
        model = mt.baseline_model(vocabulary_size, config_params,
                                  pos_vocab_size, lex_vocab_size,
                                  output_size, tokenizer=tokenizer)
        _ = train_multitask_model(model, dataset, config_params, elmo)
    elif model_type == "attention":
        attention_model = mt.attention_model(vocabulary_size, config_params,
                                             output_size, pos_vocab_size,
                                             lex_vocab_size, tokenizer=tokenizer)
        _ = train_multitask_model(
            attention_model, dataset, config_params, elmo)
    elif model_type == "seq2seq":
        seq2seq_model = mt.seq2seq_model(vocabulary_size, config_params,
                                         pos_vocab_size, lex_vocab_size,
                                         output_size, tokenizer=tokenizer)
        _ = train_multitask_model(seq2seq_model, dataset, config_params, elmo)
    elif model_type == "sense_embed_baseline":
        tokenizer = tokenizer if elmo else None
        embeddings_dim = parser_params['embeddings_dimensions']
        embeddings_file = parser_params['embeddings_file']
        pretrained_embeddings = me.pretrained_embeddings(tokenizer.word_index,
                                                         embedding_size=embeddings_dim,
                                                         embbeddings_file=embeddings_file)

        sensembed_model = me.multitask_baseline_model(vocabulary_size, config_params,
                                                      output_size, pos_vocab_size,
                                                      lex_vocab_size, weights=pretrained_embeddings, tokenizer=None)

        _ = train_multitask_model(
            sensembed_model, dataset, config_params, use_elmo=elmo, shuffle=True)
    elif model_type == "sense_embed_attention":
        tokenizer = tokenizer if elmo else None
        embeddings_dim = parser_params['embeddings_dimensions']
        embeddings_file = parser_params['embeddings_file']
        pretrained_embeddings = me.pretrained_embeddings(tokenizer.word_index,
                                                         embedding_size=embeddings_dim,
                                                         embbeddings_file=embeddings_file)

        model = me.multitask_attention_model(vocabulary_size, config_params,
                                             output_size, pos_vocab_size, lex_vocab_size,
                                             weights=pretrained_embeddings, tokenizer=None,
                                             visualize=False, plot=False)
        _ = train_multitask_model(model, dataset, config_params, use_elmo=elmo)
    elif model_type == "sense_embed_seq2seq":
        tokenizer = tokenizer if elmo else None
        embeddings_dim = parser_params['embeddings_dimensions']
        embeddings_file = parser_params['embeddings_file']
        pretrained_embeddings = me.pretrained_embeddings(tokenizer.word_index,
                                                         embedding_size=embeddings_dim,
                                                         embbeddings_file=embeddings_file)

        model = me.multitask_seq2seq_model(vocabulary_size, config_params,
                                           output_size, pos_vocab_size, lex_vocab_size,
                                           weights=pretrained_embeddings, tokenizer=tokenizer)

        _ = train_multitask_model(model, dataset, config_params, use_elmo=elmo)
    elif model_type == "bert_baseline":
        sess = tf.Session()
        max_seq_len = parser_params['bert_max_seq_len']
        dataset = mb.process_bert_data(max_seq_len)
        pos_vocab_size = dataset.get("pos_vocab_size")
        lex_vocab_size = dataset.get("lex_vocab_size")
        model = mb.multitask_baseline_model(
            output_size, pos_vocab_size, lex_vocab_size, config_params)
        mb.initialize_vars(sess)
        _ = mb.train_multitask_model(
            model, dataset, config_params, use_elmo=elmo)
    elif model_type == "bert_attention":
        sess = tf.Session()
        max_seq_len = parser_params['bert_max_seq_len']
        dataset = mb.process_bert_data(max_seq_len)
        pos_vocab_size = dataset.get("pos_vocab_size")
        lex_vocab_size = dataset.get("lex_vocab_size")
        model = mb.multitask_attention_model(
            output_size, pos_vocab_size, lex_vocab_size, config_params)
        mb.initialize_vars(sess)
        _ = mb.train_multitask_model(
            model, dataset, config_params, use_elmo=elmo)
    elif model_type == "bert_seq2seq":
        sess = tf.Session()
        max_seq_len = parser_params['bert_max_seq_len']
        dataset = mb.process_bert_data(max_seq_len)
        pos_vocab_size = dataset.get("pos_vocab_size")
        lex_vocab_size = dataset.get("lex_vocab_size")
        model = mb.multitask_seq2seq_model(
            output_size, pos_vocab_size, lex_vocab_size, config_params)
        mb.initialize_vars(sess)
        _ = mb.train_multitask_model(
            model, dataset, config_params, use_elmo=elmo)
