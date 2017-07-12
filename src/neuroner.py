import matplotlib
matplotlib.use('Agg')
import train
import dataset as ds
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from entity_lstm import EntityLSTM
import utils
import os
import conll_to_brat
import glob
import codecs
import shutil
import time
import copy
import evaluate
import random
import pickle
import brat_to_conll
import numpy as np
import utils_nlp
import distutils
import configparser
from pprint import pprint
# http://stackoverflow.com/questions/42217532/tensorflow-version-1-0-0-rc2-on-windows-opkernel-op-bestsplits-device-typ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('NeuroNER version: {0}'.format('1.0-dev'))
print('TensorFlow version: {0}'.format(tf.__version__))
import warnings
warnings.filterwarnings('ignore')


class NeuroNER(object):
    argument_default_value = 'argument_default_dummy_value_please_ignore_d41d8cd98f00b204e9800998ecf8427e'
    prediction_count = 0

    def _create_stats_graph_folder(self, parameters):
        # Initialize stats_graph_folder
        experiment_timestamp = utils.get_current_time_in_miliseconds()
        dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
        model_name = '{0}_{1}'.format(dataset_name, experiment_timestamp)
        utils.create_folder_if_not_exists(parameters['output_folder'])
        stats_graph_folder = os.path.join(parameters['output_folder'], model_name) # Folder where to save graphs
        utils.create_folder_if_not_exists(stats_graph_folder)
        return stats_graph_folder, experiment_timestamp

    def _load_parameters(self, parameters_filepath, arguments={}, verbose=True):
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
                      'experiment_name':'test',
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
            conf_parameters.read(parameters_filepath)
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
                     'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'freeze_token_embeddings', 'load_only_pretrained_token_embeddings', 'load_all_pretrained_token_embeddings']:
                parameters[k] = distutils.util.strtobool(v)
        # If loading pretrained model, set the model hyperparameters according to the pretraining parameters 
        if parameters['use_pretrained_model']:
            pretraining_parameters = self._load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'], 'parameters.ini'), verbose=False)[0]
            for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension', 'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
                if parameters[name] != pretraining_parameters[name]:
                    print('WARNING: parameter {0} was overwritten from {1} to {2} to be consistent with the pretrained model'.format(name, parameters[name], pretraining_parameters[name]))
                    parameters[name] = pretraining_parameters[name]
        if verbose: pprint(parameters)
        # Update conf_parameters to reflect final parameter values
        conf_parameters = configparser.ConfigParser()
        conf_parameters.read(os.path.join('test', 'test-parameters-training.ini'))
        parameter_to_section = utils.get_parameter_to_section_of_configparser(conf_parameters)
        for k, v in parameters.items():
            conf_parameters.set(parameter_to_section[k], k, str(v))

        return parameters, conf_parameters    
    
    def _get_valid_dataset_filepaths(self, parameters, dataset_types=['train', 'valid', 'test', 'deploy']):
        dataset_filepaths = {}
        dataset_brat_folders = {}
        for dataset_type in dataset_types:
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
    
    def _check_parameter_compatiblity(self, parameters, dataset_filepaths):
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


    def __init__(self,
                 parameters_filepath=argument_default_value, 
                 pretrained_model_folder=argument_default_value,
                 dataset_text_folder=argument_default_value, 
                 character_embedding_dimension=argument_default_value,
                 character_lstm_hidden_state_dimension=argument_default_value,
                 check_for_digits_replaced_with_zeros=argument_default_value,
                 check_for_lowercase=argument_default_value,
                 debug=argument_default_value,
                 dropout_rate=argument_default_value,
                 experiment_name=argument_default_value,
                 freeze_token_embeddings=argument_default_value,
                 gradient_clipping_value=argument_default_value,
                 learning_rate=argument_default_value,
                 load_only_pretrained_token_embeddings=argument_default_value,
                 load_all_pretrained_token_embeddings=argument_default_value,
                 main_evaluation_mode=argument_default_value,
                 maximum_number_of_epochs=argument_default_value,
                 number_of_cpu_threads=argument_default_value,
                 number_of_gpus=argument_default_value,
                 optimizer=argument_default_value,
                 output_folder=argument_default_value,
                 patience=argument_default_value,
                 plot_format=argument_default_value,
                 reload_character_embeddings=argument_default_value,
                 reload_character_lstm=argument_default_value,
                 reload_crf=argument_default_value,
                 reload_feedforward=argument_default_value,
                 reload_token_embeddings=argument_default_value,
                 reload_token_lstm=argument_default_value,
                 remap_unknown_tokens_to_unk=argument_default_value,
                 spacylanguage=argument_default_value,
                 tagging_format=argument_default_value,
                 token_embedding_dimension=argument_default_value,
                 token_lstm_hidden_state_dimension=argument_default_value,
                 token_pretrained_embedding_filepath=argument_default_value,
                 tokenizer=argument_default_value,
                 train_model=argument_default_value,
                 use_character_lstm=argument_default_value,
                 use_crf=argument_default_value,
                 use_pretrained_model=argument_default_value,
                 verbose=argument_default_value,
                 argument_default_value=argument_default_value):
        
        # Parse arguments
        arguments = dict( (k,str(v)) for k,v in locals().items() if k !='self')
        
        # Initialize parameters
        parameters, conf_parameters = self._load_parameters(arguments['parameters_filepath'], arguments=arguments)
        dataset_filepaths, dataset_brat_folders = self._get_valid_dataset_filepaths(parameters)
        self._check_parameter_compatiblity(parameters, dataset_filepaths)

        # Load dataset
        dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
        token_to_vector = dataset.load_dataset(dataset_filepaths, parameters)
        
        # Launch session
        session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
        inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
        device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
        allow_soft_placement=True, # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
        log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():
            # Create model and initialize or load pretrained model
            ### Instantiate the model
            model = EntityLSTM(dataset, parameters)
            ### Initialize the model and restore from pretrained model if needed
            sess.run(tf.global_variables_initializer())
            if not parameters['use_pretrained_model']:
                model.load_pretrained_token_embeddings(sess, dataset, parameters, token_to_vector)
                self.transition_params_trained = np.random.rand(len(dataset.unique_labels)+2,len(dataset.unique_labels)+2)
            else:
                self.transition_params_trained = model.restore_from_pretrained_model(parameters, dataset, sess, token_to_vector=token_to_vector)
            del token_to_vector

        self.dataset = dataset
        self.dataset_brat_folders = dataset_brat_folders
        self.dataset_filepaths = dataset_filepaths
        self.model = model
        self.parameters = parameters
        self.conf_parameters = conf_parameters
        self.sess = sess
   
    def fit(self):
        parameters = self.parameters
        conf_parameters = self.conf_parameters
        dataset_filepaths = self.dataset_filepaths
        dataset = self.dataset
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
        pickle.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))
            
        tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
        utils.create_folder_if_not_exists(tensorboard_log_folder)
        tensorboard_log_folders = {}
        for dataset_type in dataset_filepaths.keys():
            tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs', dataset_type)
            utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])
                
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


        # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
        bad_counter = 0 # number of epochs with no improvement on the validation test in terms of F1-score
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
                    sequence_numbers=list(range(len(dataset.token_indices['train'])))
                    random.shuffle(sequence_numbers)
                    for sequence_number in sequence_numbers:
                        transition_params_trained = train.train_step(sess, dataset, sequence_number, model, parameters)
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
                    conll_to_brat.output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder, overwrite=True)
                    self.transition_params_trained = transition_params_trained
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
        for dataset_type in dataset_filepaths.keys():
            writers[dataset_type].close()

    def predict(self, text):
        self.prediction_count += 1        
        
        if self.prediction_count == 1:
            self.parameters['dataset_text_folder'] = os.path.join('..', 'data', 'temp')
            self.stats_graph_folder, _ = self._create_stats_graph_folder(self.parameters)
        
        # Update the deploy folder, file, and dataset 
        dataset_type = 'deploy'
        ### Delete all deployment data    
        for filepath in glob.glob(os.path.join(self.parameters['dataset_text_folder'], '{0}*'.format(dataset_type))):
            if os.path.isdir(filepath): 
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)
        ### Create brat folder and file
        dataset_brat_deploy_folder = os.path.join(self.parameters['dataset_text_folder'], dataset_type)
        utils.create_folder_if_not_exists(dataset_brat_deploy_folder)
        dataset_brat_deploy_filepath = os.path.join(dataset_brat_deploy_folder, 'temp_{0}.txt'.format(str(self.prediction_count).zfill(5)))#self._get_dataset_brat_deploy_filepath(dataset_brat_deploy_folder) 
        with codecs.open(dataset_brat_deploy_filepath, 'w', 'UTF-8') as f:
            f.write(text)
        ### Update deploy filepaths
        dataset_filepaths, dataset_brat_folders = self._get_valid_dataset_filepaths(self.parameters, dataset_types=[dataset_type])
        self.dataset_filepaths.update(dataset_filepaths)
        self.dataset_brat_folders.update(dataset_brat_folders)        
        ### Update the dataset for the new deploy set
        self.dataset.update_dataset(self.dataset_filepaths, [dataset_type])
        
        # Predict labels and output brat
        output_filepaths = {}
        prediction_output = train.prediction_step(self.sess, self.dataset, dataset_type, self.model, self.transition_params_trained, self.stats_graph_folder, self.prediction_count, self.parameters, self.dataset_filepaths)
        _, _, output_filepaths[dataset_type] = prediction_output
        conll_to_brat.output_brat(output_filepaths, self.dataset_brat_folders, self.stats_graph_folder, overwrite=True)
        
        # Print and output result
        text_filepath = os.path.join(self.stats_graph_folder, 'brat', 'deploy', os.path.basename(dataset_brat_deploy_filepath))
        annotation_filepath = os.path.join(self.stats_graph_folder, 'brat', 'deploy', '{0}.ann'.format(utils.get_basename_without_extension(dataset_brat_deploy_filepath)))
        text2, entities = brat_to_conll.get_entities_from_brat(text_filepath, annotation_filepath, verbose=True)
        assert(text == text2)
        return entities
    
    def get_params(self):
        return self.parameters
    
    def close(self):
        self.__del__()
    
    def __del__(self):
        self.sess.close()
    

