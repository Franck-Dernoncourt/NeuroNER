import os
import tensorflow as tf
import numpy as np
import sklearn.metrics
from evaluate import remap_labels
from pprint import pprint
import pickle
import utils_tf
import main
import codecs
import utils_nlp
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def train_step(sess, dataset, sequence_number, model, transition_params_trained, parameters):
    # Perform one iteration
    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.token_to_index[dataset.UNK]
    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      model.dropout_keep_prob: 1-parameters['dropout_rate']
    }
    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_parameters],
                    feed_dict)
    return transition_params_trained

def prediction_step(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number, parameters, dataset_filepaths):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type,epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')

    for i in range(len(dataset.token_indices[dataset_type])):
        feed_dict = {
          model.input_token_indices: dataset.token_indices[dataset_type][i],
          model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
          model.input_token_lengths: dataset.token_lengths[dataset_type][i],
          model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
          model.dropout_keep_prob: 1.
        }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

        assert(len(predictions) == len(dataset.tokens[dataset_type][i]))
        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = dataset.labels[dataset_type][i]
        if parameters['tagging_format'] == 'bioes':
            prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
            gold_labels = utils_nlp.bioes_to_bio(gold_labels)
        for prediction, token, gold_label in zip(prediction_labels, dataset.tokens[dataset_type][i], gold_labels):
            while True:
                line = original_conll_file.readline()
                split_line = line.strip().split(' ')
                if '-DOCSTART-' in split_line[0] or len(split_line) == 0 or len(split_line[0]) == 0:
                    continue
                else:
                    token_original = split_line[0]
                    if parameters['tagging_format'] == 'bioes':
                        split_line.pop()
                    gold_label_original = split_line[-1]
                    assert(token == token_original and gold_label == gold_label_original) 
                    break            
            split_line.append(prediction)
            output_string += ' '.join(split_line) + '\n'
        output_file.write(output_string+'\n')

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

    output_file.close()
    original_conll_file.close()

    if dataset_type != 'deploy':
        if parameters['main_evaluation_mode'] == 'conll':
            conll_evaluation_script = os.path.join('.', 'conlleval')
            conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
            shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath, conll_output_filepath)
            os.system(shell_command)
            with open(conll_output_filepath, 'r') as f:
                classification_report = f.read()
                print(classification_report)
        else:
            new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions, all_y_true, dataset, parameters['main_evaluation_mode'])
            print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices, target_names=new_label_names))

    return all_predictions, all_y_true, output_filepath


def predict_labels(sess, model, transition_params_trained, parameters, dataset, epoch_number, stats_graph_folder, dataset_filepaths):
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number, parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths


def restore_model_parameters_from_pretrained_model(parameters, dataset, sess, model, model_saver):
    pretraining_dataset = pickle.load(open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb')) 
    pretrained_model_checkpoint_filepath = os.path.join(parameters['pretrained_model_folder'], 'model.ckpt')
    
    # Assert that the label sets are the same
    # Test set should have the same label set as the pretrained dataset
    assert pretraining_dataset.index_to_label == dataset.index_to_label
    
    # Assert that the model hyperparameters are the same
    pretraining_parameters = main.load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'], 'parameters.ini'), verbose=False)[0]
    for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension', 'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
        if parameters[name] != pretraining_parameters[name]:
            print("Parameters of the pretrained model:")
            pprint(pretraining_parameters)
            raise AssertionError("The parameter {0} ({1}) is different from the pretrained model ({2}).".format(name, parameters[name], pretraining_parameters[name]))
    
    # If the token and character mappings are exactly the same
    if pretraining_dataset.index_to_token == dataset.index_to_token and pretraining_dataset.index_to_character == dataset.index_to_character:
        
        # Restore the pretrained model
        model_saver.restore(sess, pretrained_model_checkpoint_filepath) # Works only when the dimensions of tensor variables are matched.
    
    # If the token and character mappings are different between the pretrained model and the current model
    else:
        
        # Resize the token and character embedding weights to match them with the pretrained model (required in order to restore the pretrained model)
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights, [pretraining_dataset.alphabet_size, parameters['character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights, [pretraining_dataset.vocabulary_size, parameters['token_embedding_dimension']])
    
        # Restore the pretrained model
        model_saver.restore(sess, pretrained_model_checkpoint_filepath) # Works only when the dimensions of tensor variables are matched.
        
        # Get pretrained embeddings
        character_embedding_weights, token_embedding_weights = sess.run([model.character_embedding_weights, model.token_embedding_weights]) 
        
        # Restore the sizes of token and character embedding weights
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights, [dataset.alphabet_size, parameters['character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights, [dataset.vocabulary_size, parameters['token_embedding_dimension']]) 
        
        # Re-initialize the token and character embedding weights
        sess.run(tf.variables_initializer([model.character_embedding_weights, model.token_embedding_weights]))
        
        # Load embedding weights from pretrained token embeddings first
        model.load_pretrained_token_embeddings(sess, dataset, parameters) 
        
        # Load embedding weights from pretrained model
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, token_embedding_weights, embedding_type='token')
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, character_embedding_weights, embedding_type='character') 
        
        del pretraining_dataset
        del character_embedding_weights
        del token_embedding_weights
    
    # Get transition parameters
    transition_params_trained = sess.run(model.transition_parameters)
    
    if not parameters['reload_character_embeddings']:
        sess.run(tf.variables_initializer([model.character_embedding_weights]))
    if not parameters['reload_character_lstm']:
        sess.run(tf.variables_initializer(model.character_lstm_variables))
    if not parameters['reload_token_embeddings']:
        sess.run(tf.variables_initializer([model.token_embedding_weights]))
    if not parameters['reload_token_lstm']:
        sess.run(tf.variables_initializer(model.token_lstm_variables))
    if not parameters['reload_feedforward']:
        sess.run(tf.variables_initializer(model.feedforward_variables))
    if not parameters['reload_crf']:
        sess.run(tf.variables_initializer(model.crf_variables))

    return transition_params_trained


