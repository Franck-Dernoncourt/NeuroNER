'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function
import tensorflow as tf
import os
import utils
import numpy as np
import matplotlib
import copy
import distutils.util
import pickle
import glob
import brat_to_conll
import conll_to_brat
import codecs
import utils_nlp
matplotlib.use('Agg')
import dataset as ds
import time
import random
import evaluate
import configparser
import train
from pprint import pprint
from entity_lstm import EntityLSTM
from tensorflow.contrib.tensorboard.plugins import projector

# http://stackoverflow.com/questions/42217532/tensorflow-version-1-0-0-rc2-on-windows-opkernel-op-bestsplits-device-typ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('NeuroNER version: {0}'.format('1.0-dev'))
print('TensorFlow version: {0}'.format(tf.__version__))

import warnings
warnings.filterwarnings('ignore')


def load_parameters(parameters_filepath=os.path.join('.','parameters.ini'), verbose=True):
    '''
    Load parameters from the ini file, and ensure that each parameter is cast to the correct type
    '''
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath)
    nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    parameters = {}
    for k,v in nested_parameters.items():
        parameters.update(v)
    for k,v in parameters.items():
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
    if verbose: pprint(parameters)
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
    
def main():

    parameters, conf_parameters = load_parameters()
    dataset_filepaths, dataset_brat_folders = get_valid_dataset_filepaths(parameters)
    check_parameter_compatiblity(parameters, dataset_filepaths)

    # Load dataset
    dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
    dataset.load_dataset(dataset_filepaths, parameters)

    # Create graph and session
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
            allow_soft_placement=True, # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
            log_device_placement=False
            )

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Initialize and save execution details
            start_time = time.time()
            experiment_timestamp = utils.get_current_time_in_miliseconds()
            results = {}
            results['epoch'] = {}
            results['execution_details'] = {}
            results['execution_details']['train_start'] = start_time
            results['execution_details']['time_stamp'] = experiment_timestamp
            results['execution_details']['early_stop'] = False
            results['execution_details']['keyboard_interrupt'] = False
            results['execution_details']['num_epochs'] = 0
            results['model_options'] = copy.copy(parameters)

            dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
            model_name = '{0}_{1}'.format(dataset_name, results['execution_details']['time_stamp'])

            output_folder=os.path.join('..', 'output')
            utils.create_folder_if_not_exists(output_folder)
            stats_graph_folder=os.path.join(output_folder, model_name) # Folder where to save graphs
            utils.create_folder_if_not_exists(stats_graph_folder)
            model_folder = os.path.join(stats_graph_folder, 'model')
            utils.create_folder_if_not_exists(model_folder)
            with open(os.path.join(model_folder, 'parameters.ini'), 'w') as parameters_file:
                conf_parameters.write(parameters_file)
            tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
            utils.create_folder_if_not_exists(tensorboard_log_folder)
            tensorboard_log_folders = {}
            for dataset_type in dataset_filepaths.keys():
                tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs', dataset_type)
                utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])
            pickle.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

            # Instantiate the model
            # graph initialization should be before FileWriter, otherwise the graph will not appear in TensorBoard
            model = EntityLSTM(dataset, parameters)

            # Instantiate the writers for TensorBoard
            writers = {}
            for dataset_type in dataset_filepaths.keys():
                writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type], graph=sess.graph)
            embedding_writer = tf.summary.FileWriter(model_folder) # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings

            embeddings_projector_config = projector.ProjectorConfig()
            tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
            tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
            token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
            tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '..')

            tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
            tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
            character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
            tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '..')

            projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

            # Write metadata for TensorBoard embeddings
            token_list_file = codecs.open(token_list_file_path,'w', 'UTF-8')
            for token_index in range(dataset.vocabulary_size):
                token_list_file.write('{0}\n'.format(dataset.index_to_token[token_index]))
            token_list_file.close()

            character_list_file = codecs.open(character_list_file_path,'w', 'UTF-8')
            for character_index in range(dataset.alphabet_size):
                if character_index == dataset.PADDING_CHARACTER_INDEX:
                    character_list_file.write('PADDING\n')
                else:
                    character_list_file.write('{0}\n'.format(dataset.index_to_character[character_index]))
            character_list_file.close()


            # Initialize the model
            sess.run(tf.global_variables_initializer())
            if not parameters['use_pretrained_model']:
                model.load_pretrained_token_embeddings(sess, dataset, parameters)

            # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
            bad_counter = 0 # number of epochs with no improvement on the validation test in terms of F1-score
            previous_best_valid_f1_score = 0
            transition_params_trained = np.random.rand(len(dataset.unique_labels)+2,len(dataset.unique_labels)+2)
            model_saver = tf.train.Saver(max_to_keep=parameters['maximum_number_of_epochs'])  # defaults to saving all variables
            epoch_number = -1
            try:
                while True:
                    step = 0
                    epoch_number += 1
                    print('\nStarting epoch {0}'.format(epoch_number))

                    epoch_start_time = time.time()

                    if parameters['use_pretrained_model'] and epoch_number == 0:
                        # Restore pretrained model parameters
                        transition_params_trained = train.restore_model_parameters_from_pretrained_model(parameters, dataset, sess, model, model_saver)
                    elif epoch_number != 0:
                        # Train model: loop over all sequences of training set with shuffling
                        sequence_numbers=list(range(len(dataset.token_indices['train'])))
                        random.shuffle(sequence_numbers)
                        for sequence_number in sequence_numbers:
                            transition_params_trained = train.train_step(sess, dataset, sequence_number, model, transition_params_trained, parameters)
                            step += 1
                            if step % 10 == 0:
                                print('Training {0:.2f}% done'.format(step/len(sequence_numbers)*100), end='\r', flush=True)

                    epoch_elapsed_training_time = time.time() - epoch_start_time
                    print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)

                    y_pred, y_true, output_filepaths = train.predict_labels(sess, model, transition_params_trained, parameters, dataset, epoch_number, stats_graph_folder, dataset_filepaths)

                    # Evaluate model: save and plot results
                    evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder, epoch_number, epoch_start_time, output_filepaths, parameters)

                    if parameters['use_pretrained_model'] and not parameters['train_model']:
                        conll_to_brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder)
                        break

                    # Save model
                    model_saver.save(sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(epoch_number)))

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
                        conll_to_brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder, overwrite=True)
                    else:
                        bad_counter += 1
                    print("The last {0} epochs have not shown improvements on the validation set.".format(bad_counter))

                    if bad_counter >= parameters['patience']:
                        print('Early Stop!')
                        results['execution_details']['early_stop'] = True
                        break

                    if epoch_number >= parameters['maximum_number_of_epochs']: break


            except KeyboardInterrupt:
                results['execution_details']['keyboard_interrupt'] = True
                print('Training interrupted')

            print('Finishing the experiment')
            end_time = time.time()
            results['execution_details']['train_duration'] = end_time - start_time
            results['execution_details']['train_end'] = end_time
            evaluate.save_results(results, stats_graph_folder)

    sess.close() # release the session's resources


if __name__ == "__main__":
    main()


