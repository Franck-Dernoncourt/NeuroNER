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

import warnings
warnings.filterwarnings('ignore')


def load_parameters(parameters_filepath, arguments={}, verbose=True):
    '''
    Load parameters from the ini file if specified, take into account any command line argument, and ensure that each parameter is cast to the correct type.
    Command line arguments take precedence over parameters specified in the parameter file.
    '''
    parameters = {'pretrained_model_folder':'../trained_models/conll_2003_en',
                  'dataset_text_folder':'../data/conll2003/en',
                  'character_embedding_dimension':25,
                  'character_lstm_hidden_state_dimension':25,
                  'check_for_digits_replaced_with_zeros':True,
                  'check_for_lowercase':True,
                  'debug':False,
                  'dropout_rate':0.5,
                  'experiment_name':'experiment',
                  'freeze_token_embeddings':False,
                  'gradient_clipping_value':5.0,
                  'learning_rate':0.005,
                  'load_only_pretrained_token_embeddings':False,
                  'main_evaluation_mode':'conll',
                  'maximum_number_of_epochs':100,
                  'number_of_cpu_threads':8,
                  'number_of_gpus':0,
                  'optimizer':'sgd',
                  'output_folder':'../output',
                  'patience':10,
                  'plot_format':'pdf',
                  'reload_character_embeddings':True,
                  'reload_character_lstm':True,
                  'reload_crf':True,
                  'reload_feedforward':True,
                  'reload_token_embeddings':True,
                  'reload_token_lstm':True,
                  'remap_unknown_tokens_to_unk':True,
                  'spacylanguage':'en',
                  'tagging_format':'bioes',
                  'token_embedding_dimension':100,
                  'token_lstm_hidden_state_dimension':100,
                  'token_pretrained_embedding_filepath':'../data/word_vectors/glove.6B.100d.txt',
                  'tokenizer':'spacy',
                  'train_model':True,
                  'use_character_lstm':True,
                  'use_crf':True,
                  'use_pretrained_model':False,
                  'verbose':False}
    # If a parameter file is specified, load it
    if len(parameters_filepath) > 0:
        conf_parameters = configparser.ConfigParser()
        conf_parameters.read(parameters_filepath, encoding="UTF-8")
        nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
        for k,v in nested_parameters.items():
            parameters.update(v)
    # Ensure that any arguments the specified in the command line overwrite parameters specified in the parameter file
    for k,v in arguments.items():
        if arguments[k] != arguments['argument_default_value']:
            parameters[k] = v
    for k,v in parameters.items():
        v = str(v)
        # If the value is a list delimited with a comma, choose one element at random.
        if ',' in v:
            v = random.choice(v.split(','))
            parameters[k] = v
        # Ensure that each parameter is cast to the correct type
        if k in ['character_embedding_dimension','character_lstm_hidden_state_dimension','token_embedding_dimension',
                 'token_lstm_hidden_state_dimension','patience','maximum_number_of_epochs','maximum_training_time','number_of_cpu_threads','number_of_gpus']:
            parameters[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
            parameters[k] = float(v)
        elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm', 'use_crf', 'train_model', 'use_pretrained_model', 'debug', 'verbose',
                 'reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings', 'reload_token_lstm', 'reload_feedforward', 'reload_crf',
                 'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'freeze_token_embeddings', 'load_only_pretrained_token_embeddings']:
            parameters[k] = distutils.util.strtobool(v)
    # If loading pretrained model, set the model hyperparameters according to the pretraining parameters 
    if parameters['use_pretrained_model']:
        pretraining_parameters = load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'], 'parameters.ini'), verbose=False)[0]
        for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension', 'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
            if parameters[name] != pretraining_parameters[name]:
                print('WARNING: parameter {0} was overwritten from {1} to {2} to be consistent with the pretrained model'.format(name, parameters[name], pretraining_parameters[name]))
                parameters[name] = pretraining_parameters[name]
    if verbose: pprint(parameters)
    # TODO: update conf_parameters to reflect the overriding
    return parameters, conf_parameters

def get_valid_dataset_filepaths(parameters):
    dataset_filepaths = {}
    dataset_brat_folders = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'], '{0}.txt'.format(dataset_type))
        dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'], dataset_type)
        dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'], '{0}_compatible_with_brat.txt'.format(dataset_type))

        # Conll file exists
        if os.path.isfile(dataset_filepaths[dataset_type]) and os.path.getsize(dataset_filepaths[dataset_type]) > 0:
            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                # Check compatibility between conll and brat files
                brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                if os.path.exists(dataset_compatible_with_brat_filepath):
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath
                conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type], dataset_brat_folders[dataset_type])

            # Brat text files do not exist
            else:

                # Populate brat text and annotation files based on conll file
                conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type], dataset_compatible_with_brat_filepath, dataset_brat_folders[dataset_type], dataset_brat_folders[dataset_type])
                dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

        # Conll file does not exist
        else:
            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'], '{0}_{1}.txt'.format(dataset_type, parameters['tokenizer']))
                if os.path.exists(dataset_filepath_for_tokenizer):
                    conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer, dataset_brat_folders[dataset_type])
                else:
                    # Populate conll file based on brat files
                    brat_to_conll.brat_to_conll(dataset_brat_folders[dataset_type], dataset_filepath_for_tokenizer, parameters['tokenizer'], parameters['spacylanguage'])
                dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

            # Brat text files do not exist
            else:
                del dataset_filepaths[dataset_type]
                del dataset_brat_folders[dataset_type]
                continue

        if parameters['tagging_format'] == 'bioes':
            # Generate conll file with BIOES format
            bioes_filepath = os.path.join(parameters['dataset_text_folder'], '{0}_bioes.txt'.format(utils.get_basename_without_extension(dataset_filepaths[dataset_type])))
            utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type], bioes_filepath)
            dataset_filepaths[dataset_type] = bioes_filepath

    return dataset_filepaths, dataset_brat_folders

def check_parameter_compatiblity(parameters, dataset_filepaths):
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            raise IOError("If train_model is set to True, both train and valid set must exist in the specified dataset folder: {0}".format(parameters['dataset_text_folder']))
    elif parameters['use_pretrained_model']:
        if 'train' in dataset_filepaths and 'valid' in dataset_filepaths:
            print("WARNING: train and valid set exist in the specified dataset folder, but train_model is set to FALSE: {0}".format(parameters['dataset_text_folder']))
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            raise IOError("For prediction mode, either test set and deploy set must exist in the specified dataset folder: {0}".format(parameters['dataset_text_folder']))
    else: #if not parameters['train_model'] and not parameters['use_pretrained_model']:
        raise ValueError('At least one of train_model and use_pretrained_model must be set to True.')

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in ['reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings', 'reload_token_lstm', 'reload_feedforward', 'reload_crf']]):
            raise ValueError('If use_pretrained_model is set to True, at least one of reload_character_embeddings, reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, reload_crf must be set to True.')

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])
        

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


