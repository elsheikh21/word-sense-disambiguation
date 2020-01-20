import datetime
import os

import numpy as np
from tensorflow.keras.callbacks import (TensorBoard, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)

from model_utils import plot_history
from parsing_dataset import multitask_train_generator
from utilities import save_pickle, build_bn2wn_dict


def train_multitask_model(model, dataset, config_params, use_elmo, shuffle=False):
    name = model._name
    train_x, train_y = dataset.get('train_x'), dataset.get('train_y')
    output_size = dataset.get('output_size')
    mask_builder = dataset.get('mask_builder')
    tokenizer = dataset.get('tokenizer')
    pos_y = dataset.get("pos_labeled_list")
    lex_y = dataset.get("lex_labeled_list")
    lemma_synsets = dataset.get("lemma_synsets", None)
    del dataset

    resources_path = os.path.join(os.getcwd(), 'resources')
    bn2wn_path = os.path.join(resources_path, 'babelnet2wordnet.tsv')
    _, wordnet_babelnet = build_bn2wn_dict(bn2wn_path)

    log_dir = os.path.join("logs", "fit_generator_multitask",
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = TensorBoard(log_dir)

    if not os.path.isdir('checkpoint_multitask'):
        os.mkdir('checkpoint_multitask')

    check_dir = os.path.join("checkpoint_multitask", f'{name}.hdf5')
    model_chkpt = ModelCheckpoint(filepath=check_dir, monitor="loss",
                                  mode="min", save_best_only=True,
                                  save_weights_only=True, verbose=True)

    early_stopping = EarlyStopping(monitor="loss", patience=3)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-9)

    epochs = int(config_params['epochs'])
    batch_size = int(config_params['batch_size'])
    cbks = [logger, model_chkpt, early_stopping, reduce_lr]

    resources_path = os.path.join(os.getcwd(), 'resources')
    try:
        history = model.fit_generator(multitask_train_generator(train_x, train_y,
                                                                pos_y, lex_y,
                                                                batch_size, output_size,
                                                                use_elmo, mask_builder,
                                                                tokenizer, use_bert=False,
                                                                shuffle=shuffle,
                                                                lemma_synsets_dict=lemma_synsets,
                                                                wn2bn=wordnet_babelnet),
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
        model.save(os.path.join(resources_path, f'{name}.h5'))
        model.save_weights(os.path.join(resources_path, f'{name}_weights.h5'))
