'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function
import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from neuroner import neuromodel

def parse_arguments(arguments=None):
    ''' Parse the NeuroNER arguments

    arguments:
        arguments the arguments, optionally given as argument
    '''
    default_param = neuromodel._get_default_param()

    parser = argparse.ArgumentParser(description='''NeuroNER CLI''', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--parameters_filepath', required=False, default=os.path.join('.','parameters.ini'), help='The parameters file')
    parser.add_argument('--character_embedding_dimension', required=False, default=default_param['character_embedding_dimension'], help='')
    parser.add_argument('--character_lstm_hidden_state_dimension', required=False, default=default_param['character_lstm_hidden_state_dimension'], help='')
    parser.add_argument('--check_for_digits_replaced_with_zeros', required=False, default=default_param['check_for_digits_replaced_with_zeros'], help='')
    parser.add_argument('--check_for_lowercase', required=False, default=default_param['check_for_lowercase'], help='')
    parser.add_argument('--dataset_text_folder', required=False, default=default_param['dataset_text_folder'], help='')
    parser.add_argument('--debug', required=False, default=default_param['debug'], help='')
    parser.add_argument('--dropout_rate', required=False, default=default_param['dropout_rate'], help='')
    parser.add_argument('--experiment_name', required=False, default=default_param['experiment_name'], help='')
    parser.add_argument('--freeze_token_embeddings', required=False, default=default_param['freeze_token_embeddings'], help='')
    parser.add_argument('--gradient_clipping_value', required=False, default=default_param['gradient_clipping_value'], help='')
    parser.add_argument('--learning_rate', required=False, default=default_param['learning_rate'], help='')
    parser.add_argument('--load_only_pretrained_token_embeddings', required=False, default=default_param['load_only_pretrained_token_embeddings'], help='')
    parser.add_argument('--load_all_pretrained_token_embeddings', required=False, default=default_param['load_all_pretrained_token_embeddings'], help='')
    parser.add_argument('--main_evaluation_mode', required=False, default=default_param['main_evaluation_mode'], help='')
    parser.add_argument('--maximum_number_of_epochs', required=False, default=default_param['maximum_number_of_epochs'], help='')
    parser.add_argument('--number_of_cpu_threads', required=False, default=default_param['number_of_cpu_threads'], help='')
    parser.add_argument('--number_of_gpus', required=False, default=default_param['number_of_gpus'], help='')
    parser.add_argument('--optimizer', required=False, default=default_param['optimizer'], help='')
    parser.add_argument('--output_folder', required=False, default=default_param['output_folder'], help='')
    parser.add_argument('--patience', required=False, default=default_param['patience'], help='')
    parser.add_argument('--plot_format', required=False, default=default_param['plot_format'], help='')
    parser.add_argument('--pretrained_model_folder', required=False, default=default_param['pretrained_model_folder'], help='')
    parser.add_argument('--reload_character_embeddings', required=False, default=default_param['reload_character_embeddings'], help='')
    parser.add_argument('--reload_character_lstm', required=False, default=default_param['reload_character_lstm'], help='')
    parser.add_argument('--reload_crf', required=False, default=default_param['reload_crf'], help='')
    parser.add_argument('--reload_feedforward', required=False, default=default_param['reload_feedforward'], help='')
    parser.add_argument('--reload_token_embeddings', required=False, default=default_param['reload_token_embeddings'], help='')
    parser.add_argument('--reload_token_lstm', required=False, default=default_param['reload_token_lstm'], help='')
    parser.add_argument('--remap_unknown_tokens_to_unk', required=False, default=default_param['remap_unknown_tokens_to_unk'], help='')
    parser.add_argument('--spacylanguage', required=False, default=default_param['spacylanguage'], help='')
    parser.add_argument('--tagging_format', required=False, default=default_param['tagging_format'], help='')
    parser.add_argument('--token_embedding_dimension', required=False, default=default_param['token_embedding_dimension'], help='')
    parser.add_argument('--token_lstm_hidden_state_dimension', required=False, default=default_param['token_lstm_hidden_state_dimension'], help='')
    parser.add_argument('--token_pretrained_embedding_filepath', required=False, default=default_param['token_pretrained_embedding_filepath'], help='')
    parser.add_argument('--tokenizer', required=False, default=default_param['tokenizer'], help='')
    parser.add_argument('--train_model', required=False, default=default_param['train_model'], help='')
    parser.add_argument('--use_character_lstm', required=False, default=default_param['use_character_lstm'], help='')
    parser.add_argument('--use_crf', required=False, default=default_param['use_crf'], help='')
    parser.add_argument('--use_pretrained_model', required=False, default=default_param['use_pretrained_model'], help='')
    parser.add_argument('--verbose', required=False, default=default_param['verbose'], help='')

    # load data to local folder
    parser.add_argument('--fetch_data', required=False, default='', help='')
    parser.add_argument('--fetch_trained_model', required=False, default='', help='')

    try:
        arguments = parser.parse_args(args=arguments)
    except:
        parser.print_help()
        sys.exit(0)

    # http://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
    arguments = vars(arguments) 
    
    return arguments

def main(argv=sys.argv):
    ''' NeuroNER main method

    Args:
        parameters_filepath the path to the parameters file
        output_folder the path to the output folder
    '''
    arguments = parse_arguments(argv[1:])

    # fetch data and models from the package
    if arguments['fetch_data'] or arguments['fetch_trained_model']:

        if arguments['fetch_data']:
            neuromodel.fetch_data(arguments['fetch_data'])
        if arguments['fetch_trained_model']:
            neuromodel.fetch_model(arguments['fetch_trained_model'])

        msg = """When the fetch_data and fetch_trained_model arguments are specified, other
            arguments are ignored. Remove these arguments to train or apply a model."""
        print(msg)
        sys.exit(0)

    # create the model
    nn = neuromodel.NeuroNER(**arguments)
    nn.fit()
    nn.close()

if __name__ == "__main__":
    main()
