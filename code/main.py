from argparse import ArgumentParser

from models_main import build_train_model, build_train_multitask_model
from parsing_dataset import load_dataset
from utilities import configure_workspace


def parse_args():
    parser = ArgumentParser(
        description="Multilingual Word Sense Disambiguation")
    parser.add_argument("--model_type", default='baseline', type=str)
    parser.add_argument("--single_task_models", default=True, type=bool)
    parser.add_argument("--embeddings_file",
                        default='glove.twitter.27B.200d.txt', type=str)
    parser.add_argument("--embeddings_dimensions", default=200, type=int)
    parser.add_argument("--bert_max_seq_len", default=512, type=int)
    return vars(parser.parse_args())


if __name__ == '__main__':
    parser_params = parse_args()
    config_params = configure_workspace()

    elmo = config_params["use_elmo"]
    dataset = load_dataset(summarize=False, elmo=elmo, use_omsti=False)

    if parser_params['single_task_models']:
        build_train_model(parser_params, config_params, dataset)
    else:
        build_train_multitask_model(parser_params, config_params, dataset)

# TODO: Make Readme more understandable, Add documentation
# TODO: Add multilingual embeddings from SensEmBert
# TODO: Run Models, Compute F1_Scores
