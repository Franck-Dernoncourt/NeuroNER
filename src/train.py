import os
import tensorflow as tf
import numpy as np
import sklearn.metrics

def remove_b_and_i_from_predictions(predictions, dataset):
    '''

    '''
    for prediction_number, prediction  in enumerate(predictions):
        prediction = int(prediction)
        prediction_label = dataset.index_to_label[prediction]
        #print('prediction_label : {0}'.format(prediction_label ))
        if prediction_label.startswith('I-'):
            new_prediction_label = 'B-' + prediction_label[2:]
            #print('new_prediction_label: {0}'.format(new_prediction_label))
            if new_prediction_label in dataset.unique_labels:
                predictions[prediction_number] = dataset.label_to_index[new_prediction_label]
                #print(prediction)


def train_step(sess, dataset, sequence_number, model, transition_params_trained, parameters):

    '''

    '''
    # Perform one iteration
    '''
    x_batch = range (20)
    y_batch = [[0,0,0,1,0]] * 20
    print('y_batch: {0}'.format(y_batch))
    feed_dict = {
      input_token_indices: x_batch,
      input_label_indices_vector: y_batch
    }
    '''

    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
#                         print("token_indices_sequence[i]: {0}".format(token_indices_sequence[i]))
            token_indices_sequence[i] = dataset.token_to_index[dataset.UNK]
#                         print("changed to UNK: {0}".format(token_indices_sequence[i]))

#     label_indices_sequence = dataset.label_indices['train'][sequence_number]
#     label_vector_indices_sequence = dataset.label_vector_indices['train'][sequence_number]
#     character_indices_padded_sequence = dataset.character_indices_padded['train'][sequence_number]
#     token_lengths_sequence = dataset.token_lengths['train'][sequence_number]

    #print('len(token_indices_sequence): {0}'.format(len(token_indices_sequence)))
    # TODO: match the names
    if len(token_indices_sequence)<2: return transition_params_trained
    feed_dict = {
      model.input_token_indices: token_indices_sequence,
      model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
      model.input_token_indices_character: dataset.character_indices_padded['train'][sequence_number],
      model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
      model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
#                   model.input_crf_transition_params: transition_params_random,
      model.dropout_keep_prob: parameters['dropout_rate']
    }
    _, _, loss, accuracy, transition_params_trained = sess.run(
                    [model.train_op, model.global_step, model.loss, model.accuracy, model.transition_params],
                    feed_dict)

    #print('loss: {0:0.3f}\taccuracy: {1:0.3f}'.format(loss, accuracy))
    #print('predictions: {0}'.format(predictions))
    return transition_params_trained

def prediction_step(sess, dataset, dataset_type, model, transition_params_trained, step, stats_graph_folder, epoch_number, parameters):
    '''

    '''

    print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{2:03d}_{1:06d}_{0}.txt'.format(dataset_type,step,epoch_number))
    output_file = open(output_filepath, 'w')

    # TODO: merge with feed_dict?
    token_indices=dataset.token_indices[dataset_type]
    label_indices=dataset.label_indices[dataset_type]
    label_vector_indices=dataset.label_vector_indices[dataset_type]
    character_indices_padded = dataset.character_indices_padded[dataset_type]
    token_lengths = dataset.token_lengths[dataset_type]
    #for i in range( 200 ):
    for i in range(len(token_indices)):

#         token_indices_sequence = token_indices[i]
#         label_vector_indices_sequence = label_vector_indices[i]
#         character_indices_padded_sequence = character_indices_padded[i]
#         token_lengths_sequence = token_lengths[i]
        #print('label_vector_indices_sequence:\n{0}'.format(label_vector_indices_sequence))

        feed_dict = {
          model.input_token_indices: token_indices[i],
          model.input_token_indices_character: character_indices_padded[i],
          model.input_token_lengths: token_lengths[i],
          model.input_label_indices_vector: label_vector_indices[i],
#                       model.input_crf_transition_params: transition_params_trained,
          model.dropout_keep_prob: 1.
        }
        #print('type(input_token_indices): {0}'.format(type(input_token_indices)))
        #print('type(input_label_indices_vector): {0}'.format(type(input_label_indices_vector)))
        #print('type(feed_dict): {0}'.format(type(feed_dict)))
        '''
        predictions = sess.run(
                        predictions_,
                        feed_dict=feed_dict)
        predictions = predictions.tolist()
        '''
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
        else:
            predictions = predictions.tolist()

        output_string = ''
        for prediction, token,gold_label in zip(predictions,dataset.tokens[dataset_type][i],dataset.labels[dataset_type][i]):
            output_string += '{0} {1} {2}\n'.format(token, gold_label, dataset.index_to_label[prediction])
        output_file.write(output_string+'\n')

        #print('predictions: {0}'.format(predictions)
        all_predictions.extend(predictions)
        all_y_true.extend(label_indices[i])

    output_file.close()


    #print('all_predictions: {0}'.format(all_predictions))
    #print('all_y_true: {0}'.format(all_y_true))
    # TODO: make pretty, move to evaluate
    #print('dataset.unique_labels: {0}'.format(dataset.unique_labels))
    remove_b_and_i_from_predictions(all_predictions, dataset)
    remove_b_and_i_from_predictions(all_y_true, dataset)
    #print(sklearn.metrics.classification_report(all_y_true, all_predictions,digits=4,labels=range(len(dataset.unique_labels)-1), target_names=dataset.unique_labels[:-1]))
    print(sklearn.metrics.classification_report(all_y_true, all_predictions,digits=4,labels=dataset.unique_label_indices_of_interest,
                                                                                                  target_names=dataset.unique_labels_of_interest))
    print(sklearn.metrics.classification_report(all_y_true, all_predictions,digits=4,labels=range(len(dataset.unique_labels)), target_names=dataset.unique_labels))

    return all_predictions, all_y_true, output_filepath

