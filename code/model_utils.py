import logging
import os

import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

from utilities import load_pickle


def plot_history(history, save_to=None):
    loss_list = [s for s in history.history.keys()
                 if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys()
                     if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys()
                if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys()
                    if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label=f'Training loss ({history.history[l][-1]:.5f})')
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label=f'Validation loss ({history.history[l][-1]:.5f})')

    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_to is not None:
        plt.savefig(f'{save_to}_loss.png')
    plt.show()

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label=f'Training accuracy ({history.history[l][-1]:.5f})')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label=f'Validation accuracy ({history.history[l][-1]:.5f})')

    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_to is not None:
        plt.savefig(f'{save_to}_acc.png')
    plt.show()


def visualize_plot_mdl(visualize, plot, model):
    # Balabizo acts as placeholder
    name = model._name if model._name is not None else 'Balabizo'
    if visualize:
        print(f'\n{name} Model Summary: \n')
        model.summary()
    if plot:
        img_path = os.path.join(os.getcwd(), 'resources', f'{name}_Model.png')
        plot_model(model, to_file=img_path, show_shapes=True)
        logging.info(f"{name} model image saved")
    logging.info(f"{name} model is created & compiled")


def train_model_huge_dataset(dir_path, model):
    # TODO: LOOP ON PICKLE FILES, LOAD THEM, TRAIN MODEL, COMPUTE F1 SCORE (ITERATIVELY)
    # TODO: READ http://nlpprogress.com/english/word_sense_disambiguation.html
    # TODO: READ ABOUT REGULARIZERS IN LSTMs
    pass


if __name__ == '__main__':
    history_path = os.path.join(
        os.getcwd(), 'resources', 'SensEmbed_BiLSTM_ATT_MultiTask_history.pkl')
    hist = load_pickle(history_path)
    plot_history(hist)
