import os
import tensorflow as tf
import numpy as np
import sklearn.metrics
from evaluate import remap_labels


def train_step(sess, dataset, sequence_number, model, transition_params_trained, parameters):
    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.token_to_index[dataset.UNK]
    if len(token_indices_sequence)<2: return transition_params_trained
    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_indices_character: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
      model.dropout_keep_prob: parameters['dropout_rate']
    }
    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_params],
                    feed_dict)
    return transition_params_trained

def prediction_step(sess, dataset, dataset_type, model, transition_params_trained, step, stats_graph_folder, epoch_number, parameters):

    print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{2:03d}_{1:06d}_{0}.txt'.format(dataset_type,step,epoch_number))
    output_file = open(output_filepath, 'w')

    token_indices=dataset.token_indices[dataset_type]
    label_indices=dataset.label_indices[dataset_type]
    label_vector_indices=dataset.label_vector_indices[dataset_type]
    character_indices_padded = dataset.character_indices_padded[dataset_type]
    token_lengths = dataset.token_lengths[dataset_type]
    for i in range(len(token_indices)):

        feed_dict = {
          model.input_token_indices: token_indices[i],
          model.input_token_indices_character: character_indices_padded[i],
          model.input_token_lengths: token_lengths[i],
          model.input_label_indices_vector: label_vector_indices[i],
          model.dropout_keep_prob: 1.
        }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
        else:
            predictions = predictions.tolist()

        output_string = ''
        for prediction, token, gold_label in zip(predictions, dataset.tokens[dataset_type][i], dataset.labels[dataset_type][i]):
            output_string += '{0} {1} {2}\n'.format(token, gold_label, dataset.index_to_label[prediction])
        output_file.write(output_string+'\n')

        all_predictions.extend(predictions)
        all_y_true.extend(label_indices[i])

    output_file.close()


    new_y_pred, new_y_true, new_label_indices, new_label_names = remap_labels(all_predictions, all_y_true, dataset, parameters['main_evaluation_mode'])
    print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices, target_names=new_label_names))
    return all_predictions, all_y_true, output_filepath

