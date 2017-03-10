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
import collections
import utils
import networkx as nx
import numpy as np
import matplotlib
import copy
import subprocess
import utils_nlp
import re
from matplotlib.cbook import ls_mapper
import distutils
import distutils.util
import pickle
from fileinput import close
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import token
import sklearn.preprocessing
import sklearn.metrics
import dataset as ds
import codecs
import time
import math
import random
import evaluate
import configparser
import train
from pprint import pprint
from entity_lstm import EntityLSTM
from tensorflow.contrib.tensorboard.plugins import projector

# http://stackoverflow.com/questions/42217532/tensorflow-version-1-0-0-rc2-on-windows-opkernel-op-bestsplits-device-typ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('TensorFlow version: {0}'.format(tf.__version__))

# Hide some other warnings
import warnings
warnings.filterwarnings('ignore')


def load_parameters():
    '''
    Load parameters from the ini file, and ensure that each parameter is cast to the correct type
    '''
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(os.path.join('.','parameters.ini'))
    nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    parameters = {}
    for k,v in nested_parameters.items():
        parameters.update(v)
    for k,v in parameters.items():
        if k in ['remove_unknown_tokens','character_embedding_dimension','character_lstm_hidden_state_dimension','token_embedding_dimension',
                 'token_lstm_hidden_state_dimension','patience','maximum_number_of_epochs','maximum_training_time','number_of_cpu_threads','number_of_gpus']:
            parameters[k] = int(v)
        if k in ['dropout_rate', 'learning_rate']:
            parameters[k] = float(v)
        if k in ['use_character_lstm','is_character_lstm_bidirect','is_token_lstm_bidirect','use_crf']:
            parameters[k] = distutils.util.strtobool(v)
    pprint(parameters)

    dataset_filepaths = {}
    dataset_filepaths['train'] = os.path.join(parameters['dataset_text_folder'], 'train.txt')
    dataset_filepaths['valid'] = os.path.join(parameters['dataset_text_folder'], 'valid.txt')
    dataset_filepaths['test']  = os.path.join(parameters['dataset_text_folder'], 'test.txt')

    return parameters, dataset_filepaths

def main():

    parameters, dataset_filepaths = load_parameters()

    # Load dataset
    dataset = ds.Dataset()
    dataset.load_dataset(dataset_filepaths, parameters)

    # Create graph and session
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          device_count={'CPU': 1, 'GPU': 1},
          allow_soft_placement=True, #  automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
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
            tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
            utils.create_folder_if_not_exists(tensorboard_log_folder)
            tensorboard_log_folders = {}
            for dataset_type in ['train', 'valid', 'test']:
                tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs', dataset_type)
                utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

            pickle.dump(dataset, open(os.path.join(stats_graph_folder, 'dataset.pickle'), 'wb'))

            # Instantiate the model
            # graph initialization should be before FileWriter, otherwise the graph will not appear in TensorBoard
            model = EntityLSTM(dataset, parameters)

            # Instantiate the writers for TensorBoard
            writers = {}
            for dataset_type in ['train', 'valid', 'test']:
                writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type], graph=sess.graph)
            embedding_writer = tf.summary.FileWriter(model_folder) # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings

            embeddings_projector_config = projector.ProjectorConfig()
            tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
            tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
            token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
            tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '..')

            tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
            tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
            character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv') #  'metadata.tsv'
            tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '..')

            projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

            # Write metadata for TensorBoard embeddings
            token_list_file = open(token_list_file_path,'w')
            for token_index in range(dataset.vocabulary_size):
                token_list_file.write('{0}\n'.format(dataset.index_to_token[token_index]))
            token_list_file.close()

            character_list_file = open(character_list_file_path,'w')
            print('len(dataset.character_to_index): {0}'.format(len(dataset.character_to_index)))
            print('len(dataset.index_to_character): {0}'.format(len(dataset.index_to_character)))
            for character_index in range(dataset.alphabet_size):
                if character_index == dataset.PADDING_CHARACTER_INDEX:
                    character_list_file.write('PADDING\n')
                else:
                    character_list_file.write('{0}\n'.format(dataset.index_to_character[character_index]))
            character_list_file.close()


            # Initialize the model
            sess.run(tf.global_variables_initializer())
            model.load_pretrained_token_embeddings(sess, dataset, parameters)


            # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
            step = 0
            bad_counter = 0 # number of epochs with no improvement on the validation test in terms of F1-score
            previous_best_valid_f1_score = 0
            transition_params_trained = np.random.rand(len(dataset.unique_labels),len(dataset.unique_labels))
            model_saver = tf.train.Saver()  # defaults to saving all variables
            epoch_number = -1
            try:
                while True:
                    epoch_number += 1
                    #epoch_number = math.floor(step / len(dataset.token_indices['train']))
                    print('\nStarting epoch {0}'.format(epoch_number), end='')

                    epoch_start_time = time.time()
                    #print('step: {0}'.format(step))

                    # Train model: loop over all sequences of training set with shuffling
                    sequence_numbers=list(range(len(dataset.token_indices['train'])))
                    random.shuffle(sequence_numbers)
                    for sequence_number in sequence_numbers:
                        transition_params_trained = train.train_step(sess, dataset, sequence_number, model, transition_params_trained, parameters)
                        step += 1
                        if step % 100 == 0:
                            print('.',end='', flush=True)
                            #break
                    print('.', flush=True)
                    #print('step: {0}'.format(step))

                    # Predict labels using trained model
                    y_pred = {}
                    y_true  = {}
                    output_filepaths = {}
                    for dataset_type in ['train', 'valid', 'test']:
                        #print('dataset_type:     {0}'.format(dataset_type))
                        prediction_output = train.prediction_step(sess, dataset, dataset_type, model, transition_params_trained,
                                                                  step, stats_graph_folder, epoch_number, parameters)
                        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
#                         model_options = None

                    epoch_elapsed_training_time = time.time() - epoch_start_time
                    print('epoch_elapsed_training_time: {0:.2f} seconds'.format(epoch_elapsed_training_time))

                    results['execution_details']['num_epochs'] = epoch_number

                    # Evaluate model: save and plot results
                    evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder, epoch_number, epoch_start_time, output_filepaths, parameters)

                    # Save model
                    model_saver.save(sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(epoch_number)))#, global_step, latest_filename, meta_graph_suffix, write_meta_graph, write_state)

                    # Save TensorBoard logs
                    summary = sess.run(model.summary_op, feed_dict=None)
                    writers['train'].add_summary(summary, epoch_number)

                    # Early stop
                    valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['micro']
                    if  valid_f1_score > previous_best_valid_f1_score:
                        bad_counter = 0
                        previous_best_valid_f1_score = valid_f1_score
                    else:
                        bad_counter += 1


                    if bad_counter > parameters['patience']:
                        print('Early Stop!')
                        results['execution_details']['early_stop'] = True
                        break

                    if epoch_number > parameters['maximum_number_of_epochs']: break

#                     break # debugging

            except KeyboardInterrupt:
                results['execution_details']['keyboard_interrupt'] = True
        #         assess_model.save_results(results, stats_graph_folder)
                print('Training interrupted')

            print('Finishing the experiment')
            end_time = time.time()
            results['execution_details']['train_duration'] = end_time - start_time
            results['execution_details']['train_end'] = end_time
            evaluate.save_results(results, stats_graph_folder)

    sess.close() # release the session's resources


if __name__ == "__main__":
    while True:
        main()
#         break # debugging
    #cProfile.run('main()') # if you want to do some profiling


