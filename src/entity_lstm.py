import tensorflow as tf
import numpy as np
import codecs
import re
import time
import utils_tf

def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):

    with tf.variable_scope("bidirectional_LSTM"):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
        else:
            batch_size = tf.shape(sequence_length)[0]

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer, state_is_tuple=True)
                # initial state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                    lstm_cell["backward"],
                                                                    input,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length,
                                                                    initial_state_fw=initial_state["forward"],
                                                                    initial_state_bw=initial_state["backward"])
        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        else:
            # max pooling
#             outputs_forward, outputs_backward = outputs
#             output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
#             output = tf.reduce_max(output, axis=1, name='output')
            # last pooling
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

    return output


class EntityLSTM(object):
    """
    An LSTM architecture for named entity recognition.
    Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
    Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
    """
    def __init__(self, dataset, parameters):

        self.verbose = False

        # Placeholders for input, output and dropout
        self.input_token_indices = tf.placeholder(tf.int32, [None], name="input_token_indices")
        self.input_label_indices_vector = tf.placeholder(tf.float32, [None, dataset.number_of_classes], name="input_label_indices_vector")
        self.input_label_indices_flat = tf.placeholder(tf.int32, [None], name="input_label_indices_flat")
        self.input_token_indices_character = tf.placeholder(tf.int32, [None, None], name="input_token_indices")
        self.input_token_lengths = tf.placeholder(tf.int32, [None], name="input_token_lengths")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        if parameters['use_character_lstm']:
            # Character-level LSTM
            # https://github.com/Franck-Dernoncourt/nlp/blob/master/textclassifier_char/src/model.py
            # Idea: reshape so that we have a tensor [number_of_token, max_token_length, token_embeddings_size], which we pass to the LSTM

            # Character embedding layer
            with tf.variable_scope("character_embedding"):  # http://stackoverflow.com/questions/39665702/tensorflow-value-error-with-variable-scope
                self.character_embedding_weights = tf.get_variable(
                    "character_embedding_weights",
                    shape=[dataset.alphabet_size, parameters['character_embedding_dimension']],
                    initializer=initializer)
                embedded_characters = tf.nn.embedding_lookup(self.character_embedding_weights, self.input_token_indices_character, name='embedded_characters')
                if self.verbose: print("embedded_characters: {0}".format(embedded_characters))
                utils_tf.variable_summaries(self.character_embedding_weights)

            # Character LSTM layer
            with tf.variable_scope('character_lstm'):
                character_lstm_output = bidirectional_LSTM(embedded_characters, parameters['character_lstm_hidden_state_dimension'], initializer,
                                                           sequence_length=self.input_token_lengths, output_sequence=False)
               

        # Token embedding layer
        with tf.variable_scope("token_embedding"):
            self.token_embedding_weights = tf.get_variable(
                "token_embedding_weights",
                shape=[dataset.vocabulary_size, parameters['token_embedding_dimension']],
                initializer=initializer)
            embedded_tokens = tf.nn.embedding_lookup(self.token_embedding_weights, self.input_token_indices) #  [sequence_length, parameters['token_embedding_dimension']]
            utils_tf.variable_summaries(self.token_embedding_weights)

        # Concatenate character LSTM outputs and token embeddings
        if parameters['use_character_lstm']:
            with tf.variable_scope("concatenate_token_and_character_vectors"):
                if self.verbose: print('embedded_tokens: {0}'.format(embedded_tokens))
                token_lstm_input = tf.concat([character_lstm_output, embedded_tokens], axis=1, name='token_lstm_input')
                if self.verbose: print("token_lstm_input: {0}".format(token_lstm_input))
                #outputs = embedded_tokens
        else:
            token_lstm_input = embedded_tokens

        # Add dropout
        with tf.variable_scope("dropout"):
            token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name='token_lstm_input_drop')
            if self.verbose: print("token_lstm_input_drop: {0}".format(token_lstm_input_drop))
            token_lstm_input_drop_expanded = tf.expand_dims(token_lstm_input_drop, axis=0, name='token_lstm_input_drop_expanded')
            if self.verbose: print("token_lstm_input_drop_expanded: {0}".format(token_lstm_input_drop_expanded))

        # https://www.tensorflow.org/api_guides/python/contrib.rnn
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        #token_lstm_input_drop_expanded = tf.expand_dims(embedded_tokens, axis=0, name='token_lstm_input_drop_expanded')

        # Token LSTM layer
        with tf.variable_scope('token_lstm'):
            token_lstm_output = bidirectional_LSTM(token_lstm_input_drop_expanded, parameters['token_lstm_hidden_state_dimension'], initializer, output_sequence=True)
           
            token_lstm_output_squeezed = tf.squeeze(token_lstm_output, axis=0)

        # Needed only if Bidirectional LSTM is used for token level
        with tf.variable_scope("feedforward_after_lstm"):
            W = tf.get_variable(
                "W",
                shape=[2 * parameters['token_lstm_hidden_state_dimension'], parameters['token_lstm_hidden_state_dimension']],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[parameters['token_lstm_hidden_state_dimension']]), name="bias")
            outputs = tf.nn.xw_plus_b(token_lstm_output_squeezed, W, b, name="output_before_tanh")
            outputs = tf.nn.tanh(outputs, name="output_after_tanh")
            utils_tf.variable_summaries(W)
            utils_tf.variable_summaries(b)

        with tf.variable_scope("feedforward_before_crf"):
            W = tf.get_variable(
                "W",
                shape=[parameters['token_lstm_hidden_state_dimension'], dataset.number_of_classes],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[dataset.number_of_classes]), name="bias")
            scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
            self.unary_scores = scores
            self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")
            utils_tf.variable_summaries(W)
            utils_tf.variable_summaries(b)



        # CRF layer
        if parameters['use_crf']:
            with tf.variable_scope("crf"):
                
                # Add start and end tokens
                small_score = -1000.0
                large_score = 0.0
                sequence_length = tf.shape(self.unary_scores)[0]
                unary_scores_with_start_and_end = tf.concat([self.unary_scores, tf.tile( tf.constant(small_score, shape=[1, 2]) , [sequence_length, 1])], 1)
                start_unary_scores = [[small_score] * dataset.number_of_classes + [large_score, small_score]]
                end_unary_scores = [[small_score] * dataset.number_of_classes + [small_score, large_score]]
                self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 0)
                start_index = dataset.number_of_classes
                end_index = dataset.number_of_classes + 1
                input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[1]), self.input_label_indices_flat, tf.constant(end_index, shape=[1]) ], 0)

                # Apply CRF layer
                sequence_length = tf.shape(self.unary_scores)[0]
                sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths')
                unary_scores_expanded = tf.expand_dims(self.unary_scores, axis=0, name='unary_scores_expanded')
                input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end, axis=0, name='input_label_indices_flat_batch')
                if self.verbose: print('unary_scores_expanded: {0}'.format(unary_scores_expanded))
                if self.verbose: print('input_label_indices_flat_batch: {0}'.format(input_label_indices_flat_batch))
                if self.verbose: print("sequence_lengths: {0}".format(sequence_lengths))
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths)#, transition_params=crf_transition_params)

                self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
                self.accuracy = tf.constant(1)

        # Do not use CRF layer
        else:
            self.transition_params=tf.get_variable(
                "transition_params",
                shape=[dataset.number_of_classes, dataset.number_of_classes],
                initializer=initializer)
            utils_tf.variable_summaries(self.transition_params)

            # Calculate Mean cross-entropy loss
            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_label_indices_vector, name='softmax')
                self.loss =  tf.reduce_mean(losses, name='cross_entropy_mean_loss') #+ l2_reg_lambda * l2_loss
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label_indices_vector, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.define_training_procedure(parameters)
        self.summary_op = tf.summary.merge_all()

    def define_training_procedure(self, parameters):
        # Define training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if parameters['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(parameters['learning_rate'])        
        else:
            raise ValueError("The lr_method parameter must be either adam or sgd.")


        grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # By defining a global_step variable and passing it to the optimizer we allow TensorFlow handle the counting of training steps for us.
        # The global step will be automatically incremented by one every time you execute train_op.
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    # TODO: maybe move out of the class?
    def load_pretrained_token_embeddings(self, sess, dataset, parameters):
        if parameters['token_pretrained_embedding_filepath'] == '':
            return
        # Load embeddings
        start_time = time.time()
        print('Load token embeddings... ', end='', flush=True)
        file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
        count = -1
        initial_weights = sess.run(self.token_embedding_weights.read_value())
        token_to_vector = {}
        for cur_line in file_input:
            count += 1
            #if count > 1000:break
            cur_line = cur_line.strip()
            cur_line = cur_line.split(' ')
            if len(cur_line)==0:continue
            token = cur_line[0]
            vector =cur_line[1:]
            token_to_vector[token] = vector

        number_of_loaded_word_vectors = 0
        number_of_token_original_case_found = 0
        number_of_token_lowercase_found = 0
        number_of_token_lowercase_normalized_found = 0
        for token in dataset.token_to_index.keys():
            if token in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[token]
                number_of_token_original_case_found += 1
            elif token.lower() in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[token.lower()]
                number_of_token_lowercase_found += 1
            elif re.sub('\d', '0', token.lower()) in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[re.sub('\d', '0', token.lower())]
                number_of_token_lowercase_normalized_found += 1
            else:
                continue
            number_of_loaded_word_vectors += 1
        file_input.close()
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
        print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
        print("number_of_token_lowercase_normalized_found: {0}".format(number_of_token_lowercase_normalized_found))
        print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
        #print("len(dataset.token_to_index): {0}".format(len(dataset.token_to_index)))
        #print("len(dataset.index_to_token): {0}".format(len(dataset.index_to_token)))
        print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))
        sess.run(self.token_embedding_weights.assign(initial_weights))

