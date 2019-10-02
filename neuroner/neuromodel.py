import codecs
import configparser
import copy
import distutils.util
import glob
import os
import pickle
from pprint import pprint
import random
import shutil
import sys
import time
import warnings
import pkg_resources

import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from neuroner import train
from neuroner import dataset
from neuroner.entity_lstm import EntityLSTM
from neuroner import utils
from neuroner import conll_to_brat
from neuroner import evaluate
from neuroner import brat_to_conll
from neuroner import utils_nlp

# http://stackoverflow.com/questions/42217532/tensorflow-version-1-0-0-rc2-on-windows-opkernel-op-bestsplits-device-typ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# print('NeuroNER version: {0}'.format('1.0.0'))
# print('TensorFlow version: {0}'.format(tf.__version__))
warnings.filterwarnings('ignore')

def fetch_model(name):
    """
    Fetch a pre-trained model and copy to a local "trained_models" folder
     If name is provided, fetch from the package folder.

    Args:
        name (str): Name of a model folder.
    """
    # get content from package and write to local dir
    # model comprises of:
    # dataset.pickle
    # model.ckpt.data-00000-of-00001
    # model.ckpt.index
    # model.ckpt.meta
    # parameters.ini
    _fetch(name, content_type="trained_models") 


def fetch_data(name):
    """
    Fetch a dataset. If name is provided, fetch from the package folder. If url
    is provided, fetch from a remote location.

    Args:
        name (str): Name of a dataset.
        url (str): URL of a model folder.
    """
    # get content from package and write to local dir
    _fetch(name, content_type="data")


def _fetch(name, content_type=None):
    """
    Load data or models from the package folder.

    Args:
        name (str): name of the resource
        content_type (str): either "data" or "trained_models"

    Returns:
        fileset (dict): dictionary containing the file content
    """
    package_name = 'neuroner'
    resource_path = '/'.join((content_type, name))

    # get dirs
    root_dir = os.path.dirname(pkg_resources.resource_filename(package_name,
        '__init__.py'))
    src_dir = os.path.join(root_dir, resource_path)
    dest_dir = os.path.join('.', content_type, name)

    if pkg_resources.resource_isdir(package_name, resource_path):

        # copy from package to dest dir
        if os.path.isdir(dest_dir):
            msg = "Directory '{}' already exists.".format(dest_dir)
            print(msg)
        else:
            shutil.copytree(src_dir, dest_dir)
            msg = "Directory created: '{}'.".format(dest_dir)
            print(msg)
    else:
        msg = "{} not found in {} package.".format(name,package_name)
        print(msg)


def _get_default_param():
    """
    Get the default parameters.

    """
    param = {'pretrained_model_folder':'./trained_models/conll_2003_en',
             'dataset_text_folder':'./data/conll2003/en',
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
             'load_all_pretrained_token_embeddings':False,
             'main_evaluation_mode':'conll',
             'maximum_number_of_epochs':100,
             'number_of_cpu_threads':8,
             'number_of_gpus':0,
             'optimizer':'sgd',
             'output_folder':'./output',
             'output_scores':False,
             'patience':10,
             'parameters_filepath': os.path.join('.','parameters.ini'),
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
             'token_pretrained_embedding_filepath':'./data/word_vectors/glove.6B.100d.txt',
             'tokenizer':'spacy',
             'train_model':True,
             'use_character_lstm':True,
             'use_crf':True,
             'use_pretrained_model':False,
             'verbose':False}

    return param


def _get_config_param(param_filepath=None):
    """
    Get the parameters from the config file.
    """
    param = {}

    # If a parameter file is specified, load it
    if param_filepath:
        param_file_txt = configparser.ConfigParser()
        param_file_txt.read(param_filepath, encoding="UTF-8")
        nested_parameters = utils.convert_configparser_to_dictionary(param_file_txt)

        for k, v in nested_parameters.items():
            param.update(v)

    return param, param_file_txt


def _clean_param_dtypes(param):
    """
    Ensure data types are correct in the parameter dictionary.

    Args:
        param (dict): dictionary of parameter settings.
    """

    # Set the data type
    for k, v in param.items():
        v = str(v)
        # If the value is a list delimited with a comma, choose one element at random.
        # NOTE: review this behaviour.
        if ',' in v:
            v = random.choice(v.split(','))
            param[k] = v

        # Ensure that each parameter is cast to the correct type
        if k in ['character_embedding_dimension',
            'character_lstm_hidden_state_dimension', 'token_embedding_dimension',
            'token_lstm_hidden_state_dimension', 'patience',
            'maximum_number_of_epochs', 'maximum_training_time',
            'number_of_cpu_threads', 'number_of_gpus']:
            param[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value']:
            param[k] = float(v)
        elif k in ['remap_unknown_tokens_to_unk', 'use_character_lstm',
            'use_crf', 'train_model', 'use_pretrained_model', 'debug', 'verbose',
            'reload_character_embeddings', 'reload_character_lstm',
            'reload_token_embeddings', 'reload_token_lstm',
            'reload_feedforward', 'reload_crf', 'check_for_lowercase',
            'check_for_digits_replaced_with_zeros', 'output_scores',
            'freeze_token_embeddings', 'load_only_pretrained_token_embeddings',
            'load_all_pretrained_token_embeddings']:
            param[k] = distutils.util.strtobool(v)

    return param


def load_parameters(**kwargs):
    '''
    Load parameters from the ini file if specified, take into account any
    command line argument, and ensure that each parameter is cast to the
    correct type.

    Command line arguments take precedence over parameters specified in the
    parameter file.
    '''
    param = {}
    param_default = _get_default_param()

    # use parameter path if provided, otherwise use default
    try:
        if kwargs['parameters_filepath']:
            parameters_filepath = kwargs['parameters_filepath']
    except:
        parameters_filepath = param_default['parameters_filepath']

    param_config, param_file_txt = _get_config_param(parameters_filepath)

    # Parameter file settings should overwrite default settings
    for k, v in param_config.items():
        param[k] = v

    # Command line args should overwrite settings in the parameter file
    for k, v in kwargs.items():
        param[k] = v

    # Any missing args can be set to default
    for k, v in param_default.items():
        if k not in param:
            param[k] = param_default[k]

    # clean the data types
    param = _clean_param_dtypes(param)

    # if loading a pretrained model, set to pretrain hyperparameters
    if param['use_pretrained_model']:

        pretrain_path = os.path.join(param['pretrained_model_folder'],
            'parameters.ini')

        if os.path.isfile(pretrain_path):
            pretrain_param, _ = _get_config_param(pretrain_path)
            pretrain_param = _clean_param_dtypes(pretrain_param)

            pretrain_list = ['use_character_lstm', 'character_embedding_dimension',
                'character_lstm_hidden_state_dimension', 'token_embedding_dimension',
                'token_lstm_hidden_state_dimension', 'use_crf']

            for name in pretrain_list:
                if param[name] != pretrain_param[name]:
                    msg = """WARNING: parameter '{0}' was overwritten from '{1}' to '{2}'
                        for consistency with the pretrained model""".format(name,
                            param[name], pretrain_param[name])
                    print(msg)
                    param[name] = pretrain_param[name]
        else:
            msg = """Warning: pretraining parameter file not found."""
            print(msg)

    # update param_file_txt to reflect the overriding
    param_to_section = utils.get_parameter_to_section_of_configparser(param_file_txt)
    for k, v in param.items():
        try:
            param_file_txt.set(param_to_section[k], k, str(v))
        except:
            pass

    pprint(param)

    return param, param_file_txt


def get_valid_dataset_filepaths(parameters):
    """
    Get valid filepaths for the datasets.
    """
    dataset_filepaths = {}
    dataset_brat_folders = {}

    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'], 
            '{0}.txt'.format(dataset_type))
        dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'], 
            dataset_type)
        dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'], 
            '{0}_compatible_with_brat.txt'.format(dataset_type))

        # Conll file exists
        if os.path.isfile(dataset_filepaths[dataset_type]) \
        and os.path.getsize(dataset_filepaths[dataset_type]) > 0:

            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) \
            and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                # Check compatibility between conll and brat files
                brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                if os.path.exists(dataset_compatible_with_brat_filepath):
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

                conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type], 
                    dataset_brat_folders[dataset_type])

            # Brat text files do not exist
            else:

                # Populate brat text and annotation files based on conll file
                conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type], dataset_compatible_with_brat_filepath, 
                    dataset_brat_folders[dataset_type], dataset_brat_folders[dataset_type])
                dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

        # Conll file does not exist
        else:
            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) \
            and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'], 
                    '{0}_{1}.txt'.format(dataset_type, parameters['tokenizer']))
                if os.path.exists(dataset_filepath_for_tokenizer):
                    conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer, 
                        dataset_brat_folders[dataset_type])
                else:
                    # Populate conll file based on brat files
                    brat_to_conll.brat_to_conll(dataset_brat_folders[dataset_type], 
                        dataset_filepath_for_tokenizer, parameters['tokenizer'], 
                        parameters['spacylanguage'])
                dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

            # Brat text files do not exist
            else:
                del dataset_filepaths[dataset_type]
                del dataset_brat_folders[dataset_type]
                continue

        if parameters['tagging_format'] == 'bioes':
            # Generate conll file with BIOES format
            bioes_filepath = os.path.join(parameters['dataset_text_folder'], 
                '{0}_bioes.txt'.format(utils.get_basename_without_extension(dataset_filepaths[dataset_type])))
            utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type], 
                bioes_filepath)
            dataset_filepaths[dataset_type] = bioes_filepath

    return dataset_filepaths, dataset_brat_folders


def check_param_compatibility(parameters, dataset_filepaths):
    """
    Check parameters are compatible.
    """
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            msg = """If train_model is set to True, both train and valid set must exist 
                in the specified dataset folder: {0}""".format(parameters['dataset_text_folder'])
            raise IOError(msg)
    elif parameters['use_pretrained_model']:
        if 'train' in dataset_filepaths and 'valid' in dataset_filepaths:
            msg = """WARNING: train and valid set exist in the specified dataset folder, 
                but train_model is set to FALSE: {0}""".format(parameters['dataset_text_folder'])
            print(msg)
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            msg = """For prediction mode, either test set and deploy set must exist 
                in the specified dataset folder: {0}""".format(parameters['dataset_text_folder'])
            raise IOError(msg)
    # if not parameters['train_model'] and not parameters['use_pretrained_model']:
    else:
        raise ValueError("At least one of train_model and use_pretrained_model must be set to True.")

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in ['reload_character_embeddings', 'reload_character_lstm', 
            'reload_token_embeddings', 'reload_token_lstm', 'reload_feedforward', 'reload_crf']]):
            msg = """If use_pretrained_model is set to True, at least one of reload_character_embeddings, 
                reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, 
                reload_crf must be set to True."""
            raise ValueError(msg)

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])

    try:
        if parameters['output_scores'] and parameters['use_crf']:
            warn_msg = """Warning when use_crf is True, scores are decoded
            using the crf. As a result, the scores cannot be directly interpreted
            in terms of class prediction.
            """
            warnings.warn(warn_msg)
    except KeyError:
        parameters['output_scores'] = False



class NeuroNER(object):
    """
    NeuroNER model.

    Args:
        param_filepath (type): description
        pretrained_model_folder (type): description
        dataset_text_folder (type): description
        character_embedding_dimension (type): description
        character_lstm_hidden_state_dimension (type): description
        check_for_digits_replaced_with_zeros (type): description
        check_for_lowercase (type): description
        debug (type): description
        dropout_rate (type): description
        experiment_name (type): description
        freeze_token_embeddings (type): description
        gradient_clipping_value (type): description
        learning_rate (type): description
        load_only_pretrained_token_embeddings (type): description
        load_all_pretrained_token_embeddings (type): description
        main_evaluation_mode (type): description
        maximum_number_of_epochs (type): description
        number_of_cpu_threads (type): description
        number_of_gpus (type): description
        optimizer (type): description
        output_folder (type): description
        output_scores (bool): description
        patience (type): description
        plot_format (type): description
        reload_character_embeddings (type): description
        reload_character_lstm (type): description
        reload_crf (type): description
        reload_feedforward (type): description
        reload_token_embeddings (type): description
        reload_token_lstm (type): description
        remap_unknown_tokens_to_unk (type): description
        spacylanguage (type): description
        tagging_format (type): description
        token_embedding_dimension (type): description
        token_lstm_hidden_state_dimension (type): description
        token_pretrained_embedding_filepath (type): description
        tokenizer (type): description
        train_model (type): description
        use_character_lstm (type): description
        use_crf (type): description
        use_pretrained_model (type): description
        verbose (type): description
    """

    prediction_count = 0

    def __init__(self, **kwargs):

        # Set parameters
        self.parameters, self.conf_parameters = load_parameters(**kwargs)

        self.dataset_filepaths, self.dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters)
        self._check_param_compatibility(self.parameters, self.dataset_filepaths)

        # Load dataset
        self.modeldata = dataset.Dataset(verbose=self.parameters['verbose'], debug=self.parameters['debug'])
        token_to_vector = self.modeldata.load_dataset(self.dataset_filepaths, self.parameters)

        # Launch session. Automatically choose a device
        # if the specified one doesn't exist
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=self.parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=self.parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': self.parameters['number_of_gpus']},
            allow_soft_placement=True,
            log_device_placement=False)

        self.sess = tf.Session(config=session_conf)
        with self.sess.as_default():

            # Initialize or load pretrained model
            self.model = EntityLSTM(self.modeldata, self.parameters)
            self.sess.run(tf.global_variables_initializer())

            if self.parameters['use_pretrained_model']:
                self.transition_params_trained = self.model.restore_from_pretrained_model(self.parameters,
                    self.modeldata, self.sess, token_to_vector=token_to_vector)
            else:
                self.model.load_pretrained_token_embeddings(self.sess, self.modeldata,
                    self.parameters, token_to_vector)
                self.transition_params_trained = np.random.rand(len(self.modeldata.unique_labels)+2,
                    len(self.modeldata.unique_labels)+2)

    def _create_stats_graph_folder(self, parameters):
        """
        Initialize stats_graph_folder.

        Args:
            parameters (type): description.
        """
        experiment_timestamp = utils.get_current_time_in_miliseconds()
        dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
        model_name = '{0}_{1}'.format(dataset_name, experiment_timestamp)
        utils.create_folder_if_not_exists(parameters['output_folder'])

        # Folder where to save graphs
        stats_graph_folder = os.path.join(parameters['output_folder'], model_name) 
        utils.create_folder_if_not_exists(stats_graph_folder)
        return stats_graph_folder, experiment_timestamp

    def _get_valid_dataset_filepaths(self, parameters, dataset_types=['train', 'valid', 'test', 'deploy']):
        """
        Get paths for the datasets.

        Args:
            parameters (type): description.
            dataset_types (type): description.
        """
        dataset_filepaths = {}
        dataset_brat_folders = {}

        for dataset_type in dataset_types:
            dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'], 
                '{0}.txt'.format(dataset_type))
            dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'], 
                dataset_type)
            dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'], 
                '{0}_compatible_with_brat.txt'.format(dataset_type))

            # Conll file exists
            if os.path.isfile(dataset_filepaths[dataset_type]) \
            and os.path.getsize(dataset_filepaths[dataset_type]) > 0:
                # Brat text files exist
                if os.path.exists(dataset_brat_folders[dataset_type]) and \
                len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                    # Check compatibility between conll and brat files
                    brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                    if os.path.exists(dataset_compatible_with_brat_filepath):
                        dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath
                    conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type], 
                        dataset_brat_folders[dataset_type])

                # Brat text files do not exist
                else:
                    # Populate brat text and annotation files based on conll file
                    conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type], 
                        dataset_compatible_with_brat_filepath, dataset_brat_folders[dataset_type], 
                        dataset_brat_folders[dataset_type])
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

            # Conll file does not exist
            else:
                # Brat text files exist
                if os.path.exists(dataset_brat_folders[dataset_type]) \
                and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                    dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'], 
                        '{0}_{1}.txt'.format(dataset_type, parameters['tokenizer']))
                    if os.path.exists(dataset_filepath_for_tokenizer):
                        conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer, 
                            dataset_brat_folders[dataset_type])
                    else:
                        # Populate conll file based on brat files
                        brat_to_conll.brat_to_conll(dataset_brat_folders[dataset_type], 
                            dataset_filepath_for_tokenizer, parameters['tokenizer'], parameters['spacylanguage'])
                    dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

                # Brat text files do not exist
                else:
                    del dataset_filepaths[dataset_type]
                    del dataset_brat_folders[dataset_type]
                    continue

            if parameters['tagging_format'] == 'bioes':
                # Generate conll file with BIOES format
                bioes_filepath = os.path.join(parameters['dataset_text_folder'],
                    '{0}_bioes.txt'.format(utils.get_basename_without_extension(dataset_filepaths[dataset_type])))
                utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type],
                    bioes_filepath)
                dataset_filepaths[dataset_type] = bioes_filepath

        return dataset_filepaths, dataset_brat_folders

    def _check_param_compatibility(self, parameters, dataset_filepaths):
        """
        Check parameters are compatible.

        Args:
            parameters (type): description.
            dataset_filepaths (type): description.
        """
        check_param_compatibility(parameters, dataset_filepaths)

    def fit(self):
        """
        Fit the model.
        """
        parameters = self.parameters
        conf_parameters = self.conf_parameters
        dataset_filepaths = self.dataset_filepaths
        modeldata = self.modeldata
        dataset_brat_folders = self.dataset_brat_folders
        sess = self.sess
        model = self.model
        transition_params_trained = self.transition_params_trained
        stats_graph_folder, experiment_timestamp = self._create_stats_graph_folder(parameters)

        # Initialize and save execution details
        start_time = time.time()
        results = {}
        results['epoch'] = {}
        results['execution_details'] = {}
        results['execution_details']['train_start'] = start_time
        results['execution_details']['time_stamp'] = experiment_timestamp
        results['execution_details']['early_stop'] = False
        results['execution_details']['keyboard_interrupt'] = False
        results['execution_details']['num_epochs'] = 0
        results['model_options'] = copy.copy(parameters)

        model_folder = os.path.join(stats_graph_folder, 'model')
        utils.create_folder_if_not_exists(model_folder)
        with open(os.path.join(model_folder, 'parameters.ini'), 'w') as parameters_file:
            conf_parameters.write(parameters_file)
        pickle.dump(modeldata, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

        tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
        utils.create_folder_if_not_exists(tensorboard_log_folder)
        tensorboard_log_folders = {}
        for dataset_type in dataset_filepaths.keys():
            tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 
                'tensorboard_logs', dataset_type)
            utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

        # Instantiate the writers for TensorBoard
        writers = {}
        for dataset_type in dataset_filepaths.keys():
            writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type], 
                graph=sess.graph)

        # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings
        embedding_writer = tf.summary.FileWriter(model_folder)

        embeddings_projector_config = projector.ProjectorConfig()
        tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
        tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
        token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
        tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '.')

        tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
        tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
        character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
        tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '.')

        projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

        # Write metadata for TensorBoard embeddings
        token_list_file = codecs.open(token_list_file_path,'w', 'UTF-8')
        for token_index in range(modeldata.vocabulary_size):
            token_list_file.write('{0}\n'.format(modeldata.index_to_token[token_index]))
        token_list_file.close()

        character_list_file = codecs.open(character_list_file_path,'w', 'UTF-8')
        for character_index in range(modeldata.alphabet_size):
            if character_index == modeldata.PADDING_CHARACTER_INDEX:
                character_list_file.write('PADDING\n')
            else:
                character_list_file.write('{0}\n'.format(modeldata.index_to_character[character_index]))
        character_list_file.close()


        # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
        # number of epochs with no improvement on the validation test in terms of F1-score
        bad_counter = 0
        previous_best_valid_f1_score = 0
        epoch_number = -1
        try:
            while True:
                step = 0
                epoch_number += 1
                print('\nStarting epoch {0}'.format(epoch_number))

                epoch_start_time = time.time()

                if epoch_number != 0:
                    # Train model: loop over all sequences of training set with shuffling
                    sequence_numbers=list(range(len(modeldata.token_indices['train'])))
                    random.shuffle(sequence_numbers)
                    for sequence_number in sequence_numbers:
                        transition_params_trained = train.train_step(sess, modeldata, 
                            sequence_number, model, parameters)
                        step += 1
                        if step % 10 == 0:
                            print('Training {0:.2f}% done'.format(step/len(sequence_numbers)*100),
                                end='\r', flush=True)

                epoch_elapsed_training_time = time.time() - epoch_start_time
                print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time),
                    flush=True)

                y_pred, y_true, output_filepaths = train.predict_labels(sess, model,
                    transition_params_trained, parameters, modeldata, epoch_number,
                    stats_graph_folder, dataset_filepaths)

                # Evaluate model: save and plot results
                evaluate.evaluate_model(results, modeldata, y_pred, y_true, stats_graph_folder,
                    epoch_number, epoch_start_time, output_filepaths, parameters)

                if parameters['use_pretrained_model'] and not parameters['train_model']:
                    conll_to_brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder)
                    break

                # Save model
                model.saver.save(sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(epoch_number)))

                # Save TensorBoard logs
                summary = sess.run(model.summary_op, feed_dict=None)
                writers['train'].add_summary(summary, epoch_number)
                writers['train'].flush()
                utils.copytree(writers['train'].get_logdir(), model_folder)


                # Early stop
                valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
                if  valid_f1_score > previous_best_valid_f1_score:
                    bad_counter = 0
                    previous_best_valid_f1_score = valid_f1_score
                    conll_to_brat.output_brat(output_filepaths, dataset_brat_folders,
                        stats_graph_folder, overwrite=True)
                    self.transition_params_trained = transition_params_trained
                else:
                    bad_counter += 1
                print("The last {0} epochs have not shown improvements on the validation set.".format(bad_counter))

                if bad_counter >= parameters['patience']:
                    print('Early Stop!')
                    results['execution_details']['early_stop'] = True
                    break

                if epoch_number >= parameters['maximum_number_of_epochs']:
                    break

        except KeyboardInterrupt:
            results['execution_details']['keyboard_interrupt'] = True
            print('Training interrupted')

        print('Finishing the experiment')
        end_time = time.time()
        results['execution_details']['train_duration'] = end_time - start_time
        results['execution_details']['train_end'] = end_time
        evaluate.save_results(results, stats_graph_folder)
        for dataset_type in dataset_filepaths.keys():
            writers[dataset_type].close()

    def predict(self, text):
        """
        Predict

        Args:
            text (str): Description.
        """
        self.prediction_count += 1        

        if self.prediction_count == 1:
            self.parameters['dataset_text_folder'] = os.path.join('.', 'data', 'temp')
            self.stats_graph_folder, _ = self._create_stats_graph_folder(self.parameters)

        # Update the deploy folder, file, and modeldata 
        dataset_type = 'deploy'

        # Delete all deployment data    
        for filepath in glob.glob(os.path.join(self.parameters['dataset_text_folder'], 
            '{0}*'.format(dataset_type))):
            if os.path.isdir(filepath): 
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)

        # Create brat folder and file
        dataset_brat_deploy_folder = os.path.join(self.parameters['dataset_text_folder'], 
            dataset_type)
        utils.create_folder_if_not_exists(dataset_brat_deploy_folder)
        dataset_brat_deploy_filepath = os.path.join(dataset_brat_deploy_folder, 
            'temp_{0}.txt'.format(str(self.prediction_count).zfill(5)))
            #self._get_dataset_brat_deploy_filepath(dataset_brat_deploy_folder) 
        with codecs.open(dataset_brat_deploy_filepath, 'w', 'UTF-8') as f:
            f.write(text)

        # Update deploy filepaths
        dataset_filepaths, dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters,
            dataset_types=[dataset_type])
        self.dataset_filepaths.update(dataset_filepaths)
        self.dataset_brat_folders.update(dataset_brat_folders)    

        # Update the dataset for the new deploy set
        self.modeldata.update_dataset(self.dataset_filepaths, [dataset_type])

        # Predict labels and output brat
        output_filepaths = {}
        prediction_output = train.prediction_step(self.sess, self.modeldata,
            dataset_type, self.model, self.transition_params_trained,
            self.stats_graph_folder, self.prediction_count, self.parameters,
            self.dataset_filepaths)

        _, _, output_filepaths[dataset_type] = prediction_output
        conll_to_brat.output_brat(output_filepaths, self.dataset_brat_folders, 
            self.stats_graph_folder, overwrite=True)

        # Print and output result
        text_filepath = os.path.join(self.stats_graph_folder, 'brat', 'deploy', 
            os.path.basename(dataset_brat_deploy_filepath))
        annotation_filepath = os.path.join(self.stats_graph_folder, 'brat', 
            'deploy', '{0}.ann'.format(utils.get_basename_without_extension(dataset_brat_deploy_filepath)))
        text2, entities = brat_to_conll.get_entities_from_brat(text_filepath, 
            annotation_filepath, verbose=True)
        assert(text == text2)
        return entities

    def get_params(self):
        return self.parameters

    def close(self):
        self.__del__()

    def __del__(self):
        self.sess.close()
