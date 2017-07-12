'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function
import os
import argparse
from argparse import RawTextHelpFormatter
import sys
from neuroner import NeuroNER

def parse_arguments(arguments=None):
    ''' Parse the NeuroNER arguments

    arguments:
        arguments the arguments, optionally given as argument
    '''
    parser = argparse.ArgumentParser(description='''NeuroNER CLI''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--parameters_filepath', required=False, default=os.path.join('.','parameters.ini'), help='The parameters file')

    argument_default_value = 'argument_default_dummy_value_please_ignore_d41d8cd98f00b204e9800998ecf8427e'
    parser.add_argument('--character_embedding_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--character_lstm_hidden_state_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--check_for_digits_replaced_with_zeros', required=False, default=argument_default_value, help='')
    parser.add_argument('--check_for_lowercase', required=False, default=argument_default_value, help='')
    parser.add_argument('--dataset_text_folder', required=False, default=argument_default_value, help='')
    parser.add_argument('--debug', required=False, default=argument_default_value, help='')
    parser.add_argument('--dropout_rate', required=False, default=argument_default_value, help='')
    parser.add_argument('--experiment_name', required=False, default=argument_default_value, help='')
    parser.add_argument('--freeze_token_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--gradient_clipping_value', required=False, default=argument_default_value, help='')
    parser.add_argument('--learning_rate', required=False, default=argument_default_value, help='')
    parser.add_argument('--load_only_pretrained_token_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--load_all_pretrained_token_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--main_evaluation_mode', required=False, default=argument_default_value, help='')
    parser.add_argument('--maximum_number_of_epochs', required=False, default=argument_default_value, help='')
    parser.add_argument('--number_of_cpu_threads', required=False, default=argument_default_value, help='')
    parser.add_argument('--number_of_gpus', required=False, default=argument_default_value, help='')
    parser.add_argument('--optimizer', required=False, default=argument_default_value, help='')
    parser.add_argument('--output_folder', required=False, default=argument_default_value, help='')
    parser.add_argument('--patience', required=False, default=argument_default_value, help='')
    parser.add_argument('--plot_format', required=False, default=argument_default_value, help='')
    parser.add_argument('--pretrained_model_folder', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_character_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_character_lstm', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_crf', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_feedforward', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_token_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_token_lstm', required=False, default=argument_default_value, help='')
    parser.add_argument('--remap_unknown_tokens_to_unk', required=False, default=argument_default_value, help='')
    parser.add_argument('--spacylanguage', required=False, default=argument_default_value, help='')
    parser.add_argument('--tagging_format', required=False, default=argument_default_value, help='')
    parser.add_argument('--token_embedding_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--token_lstm_hidden_state_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--token_pretrained_embedding_filepath', required=False, default=argument_default_value, help='')
    parser.add_argument('--tokenizer', required=False, default=argument_default_value, help='')
    parser.add_argument('--train_model', required=False, default=argument_default_value, help='')
    parser.add_argument('--use_character_lstm', required=False, default=argument_default_value, help='')
    parser.add_argument('--use_crf', required=False, default=argument_default_value, help='')
    parser.add_argument('--use_pretrained_model', required=False, default=argument_default_value, help='')
    parser.add_argument('--verbose', required=False, default=argument_default_value, help='')

    try:
        arguments = parser.parse_args(args=arguments)
    except:
        parser.print_help()
        sys.exit(0)

    arguments = vars(arguments) # http://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
    arguments['argument_default_value'] = argument_default_value
    return arguments

def main(argv=sys.argv):
    ''' NeuroNER main method

    Args:
        parameters_filepath the path to the parameters file
        output_folder the path to the output folder
    '''
    # Parse arguments
    arguments = parse_arguments(argv[1:])
    
    nn = NeuroNER(**arguments)
    nn.fit()
    nn.close() 

if __name__ == "__main__":
    main()


