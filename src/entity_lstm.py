import tensorflow as tf
import numpy as np
import codecs
import re

def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):
    # Bidirectional LSTM
#             lstm_cell_forward = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#             lstm_cell_backward = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#     # LSTM cells
#     lstm_cell_forward = tf.contrib.rnn.LSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer)
#     print("lstm_cell_forward: {0}".format(lstm_cell_forward))
#     lstm_cell_backward = tf.contrib.rnn.LSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer)
#     print("lstm_cell_backward: {0}".format(lstm_cell_backward))
    with tf.variable_scope("bidirectional_LSTM"):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            #print("sequence_length: {0}".format(sequence_length))
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
            #print("sequence_length: {0}".format(sequence_length))
        else:
            batch_size = tf.shape(sequence_length)[0]

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.LSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer)
    #             print("lstm_cell_forward: {0}".format(lstm_cell_forward))
                # initial state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                #http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
    #             initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(initial_cell_state, initial_output_state)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

    #             tf.Variable(state_c, trainable=False),
    #             tf.Variable(state_h, trainable=False))
    #     initial_cell_backward = tf.get_variable("initial_cell_backward", shape=[batch_size, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
    #     initial_output_backward = tf.get_variable("initial_output_backward", shape=[batch_size, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
    # #             initial_backward = tuple([initial_cell_backward, initial_output_backward])
    #     initial_backward = tf.contrib.rnn.LSTMStateTuple(initial_cell_backward, initial_output_backward)
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
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
            output = tf.reduce_max(output, axis=1, name='output')
#             # last pooling
#             final_states_forward, final_states_backward = final_states
#             output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

    return output


class EntityLSTM(object):
    """
    An LSTM architecture for named entity recognition.
    Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
    Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
    """
    # TODO: parameters
    def __init__(self, dataset, parameters):
        '''
        number_of_classes = dataset.number_of_classes,
        dataset.vocabulary_size = dataset.vocabulary_size,
        dataset.alphabet_size = dataset.dataset.alphabet_size,
        parameters['token_embedding_dimension'] =  parameters['token_embedding_dimension'],
        parameters['character_embedding_dimension'] = parameters['character_embedding_dimension'],
        parameters['character_lstm_hidden_state_dimension'] = parameters['character_lstm_hidden_state_dimension'],
        n_hidden = parameters['token_lstm_hidden_state_dimension'],
        dropout = parameters['dropout_rate'],

        print('number_of_classes: {0}'.format(number_of_classes))
        print('dataset.vocabulary_size: {0}'.format(dataset.vocabulary_size))
        print('dataset.alphabet_size: {0}'.format(dataset.alphabet_size))
        '''

        self.verbose = False

        # Placeholders for input, output and dropout
        self.input_token_indices = tf.placeholder(tf.int32, [None], name="input_token_indices")
        self.input_label_indices_vector = tf.placeholder(tf.float32, [None, dataset.number_of_classes], name="input_label_indices_vector")
        self.input_label_indices_flat = tf.placeholder(tf.int32, [None], name="input_label_indices_flat")
        self.input_token_indices_character = tf.placeholder(tf.int32, [None, None], name="input_token_indices")
        self.input_token_lengths = tf.placeholder(tf.int32, [None], name="input_token_lengths")
#         self.input_crf_transition_params = tf.placeholder(tf.float32, [dataset.number_of_classes, dataset.number_of_classes], name="input_crf_transition_params")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
#         print('input_label_indices_vector: {0}'.format(input_label_indices_vector))

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        if parameters['use_character_lstm']:
            # Character-level LSTM
            # https://github.com/Franck-Dernoncourt/nlp/blob/master/textclassifier_char/src/model.py
            # Idea: reshape so that we have a tensor [number_of_token, max_token_length, token_embeddings_size], which we pass to the LSTM

            # Character embedding layer
            with tf.variable_scope("character_embedding"):  # http://stackoverflow.com/questions/39665702/tensorflow-value-error-with-variable-scope
    #             character_embedding_weights = tf.Variable(
    #                 tf.random_uniform([dataset.alphabet_size, parameters['character_embedding_dimension']], -1.0, 1.0),
    #                 #tf.random_uniform([dataset.vocabulary_size, dataset.number_of_classes], -1.0, 1.0),
    #                 name="character_embedding_weights")
                character_embedding_weights = tf.get_variable(
                    "character_embedding_weights",
                    shape=[dataset.alphabet_size, parameters['character_embedding_dimension']],
                    initializer=initializer)
                embedded_characters = tf.nn.embedding_lookup(character_embedding_weights, self.input_token_indices_character, name='embedded_characters')
                if self.verbose: print("embedded_characters: {0}".format(embedded_characters))

            # Character LSTM layer
            with tf.variable_scope('character_lstm'):
                character_lstm_output = bidirectional_LSTM(embedded_characters, parameters['character_lstm_hidden_state_dimension'], initializer,
                                                           sequence_length=self.input_token_lengths, output_sequence=False)
                # Unidirectional LSTM
    ##             character_lstm_cell = tf.contrib.rnn.BasicLSTMCell(parameters['character_lstm_hidden_state_dimension'], forget_bias=1.0)
    #             character_lstm_cell = tf.contrib.rnn.LSTMCell(parameters['character_lstm_hidden_state_dimension'], forget_bias=1.0, initializer=initializer)
    #             character_lstm_outputs, character_lstm_states = tf.nn.dynamic_rnn(character_lstm_cell, embedded_characters, dtype=tf.float32, sequence_length=input_token_lengths)
    #             character_lstm_output = character_lstm_states[1]

#                 # TODO: refactor to avoid repeating code, for bidirectional lstm as well as character/token lstms
#                 # Bidirectional LSTM
#                 # cells
#                 character_lstm_cell_forward = tf.contrib.rnn.LSTMCell(parameters['character_lstm_hidden_state_dimension'], forget_bias=1.0, initializer=initializer)
#                 # initial states
#                 batch_size = tf.shape(self.input_token_lengths)[0]
#                 initial_cell_forward = tf.get_variable("initial_cell_forward", shape=[1, parameters['character_lstm_hidden_state_dimension']], dtype=tf.float32, initializer=initializer)
#                 initial_output_forward = tf.get_variable("initial_output_forward", shape=[1, parameters['character_lstm_hidden_state_dimension']], dtype=tf.float32, initializer=initializer)
#                 # http://stackoverflow.com/questions/38806136/tensorflow-shape-of-a-tiled-tensor
#                 c_states_forward = tf.tile(initial_cell_forward, tf.stack([batch_size, 1]))
#     #             c_states_forward.set_shape([None, initial_cell_forward.get_shape()[1]])  # or `c.set_shape([None, 5])`
#                 h_states_forward = tf.tile(initial_output_forward, tf.stack([batch_size, 1]))
#     #             h_states_forward.set_shape([None, initial_output_forward.get_shape()[1]])  # or `c.set_shape([None, 5])`
#                 initial_forward = tf.contrib.rnn.LSTMStateTuple(c_states_forward, h_states_forward)
#     #
#                 character_lstm_cell_backward = tf.contrib.rnn.LSTMCell(parameters['character_lstm_hidden_state_dimension'], forget_bias=1.0, initializer=initializer)
#                 initial_cell_backward = tf.get_variable("initial_cell_backward", shape=[1, parameters['character_lstm_hidden_state_dimension']], dtype=tf.float32, initializer=initializer)
#                 initial_output_backward = tf.get_variable("initial_output_backward", shape=[1, parameters['character_lstm_hidden_state_dimension']], dtype=tf.float32, initializer=initializer)
#                 c_states_backward = tf.tile(initial_cell_backward, tf.stack([batch_size, 1]))
#     #             c_states_backward.set_shape([None, initial_cell_backward.get_shape()[1]])  # or `c.set_shape([None, 5])`
#                 h_states_backward = tf.tile(initial_output_backward, tf.stack([batch_size, 1]))
#     #             h_states_backward.set_shape([None, initial_output_backward.get_shape()[1]])  # or `c.set_shape([None, 5])`
#                 initial_backward = tf.contrib.rnn.LSTMStateTuple(c_states_backward, h_states_backward)
#                 # lstm
#                 character_lstm_outputs, character_lstm_states = tf.nn.bidirectional_dynamic_rnn(character_lstm_cell_forward,
#                                                                                                 character_lstm_cell_backward,
#                                                                                                 embedded_characters,
#                                                                                                 dtype=tf.float32,
#                                                                                                 sequence_length=self.input_token_lengths,
#                                                                                                 initial_state_fw=initial_forward,
#                                                                                                 initial_state_bw=initial_backward)
#                 # TODO: maxpool instead
#                 character_lstm_forward_states, character_lstm_backward_states = character_lstm_states
#                 character_lstm_output = tf.concat([character_lstm_forward_states[1], character_lstm_backward_states[1]], axis=1, name='character_lstm_output')
#                 # character_lstm_states dimension is [batch_size, cell.state_size].
#                 if self.verbose: print("character_lstm_outputs: {0}".format(character_lstm_outputs))
#                 if self.verbose: print("character_lstm_states: {0}".format(character_lstm_states))
#                 if self.verbose: print('character_lstm_states[0]: {0}'.format(character_lstm_states[0]))
#                 if self.verbose: print("character_lstm_states[1]: {0}".format(character_lstm_states[1]))
#                 if self.verbose: print("character_lstm_forward_states: {0}".format(character_lstm_forward_states))
#                 if self.verbose: print("character_lstm_backward_states: {0}".format(character_lstm_backward_states))
#                 if self.verbose: print("character_lstm_output: {0}".format(character_lstm_output))


        # [Using a pre-trained word embedding (word2vec or Glove) in TensorFlow](http://stackoverflow.com/q/35687678/395857)

        # Token embedding layer
        with tf.variable_scope("token_embedding"):
#             W = tf.Variable(
#                 tf.random_uniform([dataset.vocabulary_size, parameters['token_embedding_dimension']], -1.0, 1.0),
#                 #tf.random_uniform([dataset.vocabulary_size, dataset.number_of_classes], -1.0, 1.0),
#                 name="W")
            self.W = tf.get_variable(
                "W",
                shape=[dataset.vocabulary_size, parameters['token_embedding_dimension']],
                initializer=initializer)
            embedded_tokens = tf.nn.embedding_lookup(self.W, self.input_token_indices) #  [sequence_length, parameters['token_embedding_dimension']]
            #embedded_tokens_expanded = tf.expand_dims(embedded_tokens, -1) #  [sequence_length, parameters['token_embedding_dimension'], 1]

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
            #token_lstm_input_drop_expanded = tf.expand_dims(embedded_tokens, axis=0, name='token_lstm_input_drop_expanded')
            #n_steps = 9
            #x = tf.split(token_lstm_input_drop_expanded, n_steps, 0)

            # Unidirectional LSTM
#             lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#             lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, initializer=initializer)
#             outputs, states = tf.nn.dynamic_rnn(lstm_cell, token_lstm_input_drop_expanded, dtype=tf.float32)
            token_lstm_output = bidirectional_LSTM(token_lstm_input_drop_expanded, parameters['token_lstm_hidden_state_dimension'], initializer, output_sequence=True)
            #dims = tf.shape(outputs)
#             dims = outputs.get_shape ().as_list() #http://stackoverflow.com/questions/40666316/how-to-get-tensorflow-tensor-dimensions-shape-as-int-values
#             if self.verbose: print('dims: {0}'.format(dims))
#             if self.verbose: print(dims[1])
#             if self.verbose: print('type(dims[1]): {0}'.format(type(dims[1])))
            #outputs = tf.reshape(outputs, [dims[1], dims[2]])
            token_lstm_output_squeezed = tf.squeeze(token_lstm_output, axis=0)
            #outputs = tf.reshape(outputs, [9, 200])
#             if self.verbose: print('outputs: {0}'.format(outputs))

        # Needed only if Bidirectional LSTM is used for token level
        with tf.variable_scope("feedforward_after_lstm"):
            # http://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
            '''
            'W = tf.get_variable(
                tf.random_uniform([dataset.vocabulary_size, parameters['token_embedding_dimension']], -1.0, 1.0),
                name="W")
            '''
            W = tf.get_variable(
                "W",
                shape=[2 * parameters['token_lstm_hidden_state_dimension'], parameters['token_lstm_hidden_state_dimension']],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[parameters['token_lstm_hidden_state_dimension']]), name="b")
            outputs = tf.nn.xw_plus_b(token_lstm_output_squeezed, W, b, name="output_before_tanh")
            outputs = tf.nn.tanh(outputs, name="output_after_tanh")
            #scores = embedded_tokens

        with tf.variable_scope("feedforward_before_crf"):
            # http://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow
            '''
            'W = tf.get_variable(
                tf.random_uniform([dataset.vocabulary_size, parameters['token_embedding_dimension']], -1.0, 1.0),
                name="W")
            '''
            W = tf.get_variable(
                "W",
                #shape=[parameters['token_embedding_dimension'], dataset.number_of_classes],
                shape=[parameters['token_lstm_hidden_state_dimension'], dataset.number_of_classes],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[dataset.number_of_classes]), name="b")
            scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
            self.unary_scores = scores
            self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")
            #scores = embedded_tokens
            #predictions = tf.argmax(scores, 1, name="predictions")


        #tf.nn.softmax(scores, dim=-1, name=None)

        # CRF layer
        if parameters['use_crf']:
            with tf.variable_scope("crf"):
#             crf_transition_params = tf.get_variable(
#                 "transition_params",
#                 #shape=[parameters['token_embedding_dimension'], dataset.number_of_classes],
#                 shape=[dataset.number_of_classes, dataset.number_of_classes],
#                 initializer=initializer)
                # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
                # Compute the log-likelihood of the gold sequences and keep the transition
                # params for inference at test time.
#                 if self.verbose: print('scores: {0}'.format(scores))
# #                 unary_scores = scores
#                 if self.verbose: print('unary_scores: {0}'.format(unary_scores))
#                 #unary_scores = tf.reshape(unary_scores, [1,0])
#                 #unary_scores = tf.transpose(unary_scores, [1,0])
#                 if self.verbose: print('unary_scores after reshape: {0}'.format(unary_scores))
                sequence_lengths = tf.shape(self.unary_scores)[0]
                sequence_lengths = tf.expand_dims(sequence_lengths, axis=0, name='sequence_lengths')
                unary_scores_expanded = tf.expand_dims(self.unary_scores, axis=0, name='unary_scores_expanded')
                input_label_indices_flat_batch = tf.expand_dims(self.input_label_indices_flat, axis=0, name='input_label_indices_flat_batch')
                #input_label_indices_flat_batch = input_label_indices_flat
                if self.verbose: print('unary_scores_expanded: {0}'.format(unary_scores_expanded))
                if self.verbose: print('input_label_indices_flat_batch: {0}'.format(input_label_indices_flat_batch))
                if self.verbose: print("sequence_lengths: {0}".format(sequence_lengths))
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths)#, transition_params=crf_transition_params)

                self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
                self.accuracy = tf.constant(1)
                #predictions_ = tf.constant(1)

#                 self.unary_scores = tf.squeeze(unary_scores, axis=0)
#                 if self.verbose: print('unary_scores: {0}'.format(unary_scores))
    #             if self.verbose: print('self.input_crf_transition_params: {0}'.format(self.input_crf_transition_params))
                #unary_scores [0]
                #predictions_, _ = tf.contrib.crf.viterbi_decode(unary_scores, input_crf_transition_params)

        # Do not use CRF layer
        else:
            self.transition_params=tf.get_variable(
                "transition_params",
                shape=[dataset.number_of_classes, dataset.number_of_classes],
                initializer=initializer)

            # Calculate Mean cross-entropy loss
            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_label_indices_vector, name='softmax')
                self.loss =  tf.reduce_mean(losses, name='cross_entropy_mean_loss') #+ l2_reg_lambda * l2_loss
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label_indices_vector, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.define_training_procedure(parameters)

    def define_training_procedure(self, parameters):
        # Define training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if parameters['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(1e-3)
        elif parameters['optimizer'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(0.005)
        else:
            raise ValueError("The lr_method parameter must be either adam or sgd.")

        # https://github.com/google/prettytensor/issues/6
        # https://www.tensorflow.org/api_docs/python/framework/graph_collections

        #if self.verbose: print('tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) : {0}'.format(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) ))
        #if self.verbose: print('tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) : {0}'.format(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) ))
        #if self.verbose: print('tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) : {0}'.format(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) ))

        # https://github.com/blei-lab/edward/issues/286#ref-pullrequest-181330211 : utility function to get all tensorflow variables a node depends on

        grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # By defining a global_step variable and passing it to the optimizer we allow TensorFlow handle the counting of training steps for us.
        # The global step will be automatically incremented by one every time you execute train_op.
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    # TODO: maybe move out of the class?
    def load_pretrained_token_embeddings(self, sess, dataset, parameters):
        if parameters['token_pretrained_embedding_filepath'] == '':
            return
        # Load embeddings
        # https://github.com/dennybritz/cnn-text-classification-tf/issues/17
        print('Load embeddings')
        #full_word_embeddings_folder =os.path.join('..','data','word_vectors')
        #full_word_embeddings_filepath = os.path.join(full_word_embeddings_folder,'glove.6B.{0}d.txt'.format(token_embedding_size))
        file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
        count = -1
    #     case_sensitive = False
    #     initial_weights = np.random.uniform(-0.25,0.25,(vocabulary_size, token_embedding_size))
        initial_weights = sess.run(self.W.read_value())
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
            # TODO: shouldn't it apply to token_to_index instead?
    #         if not case_sensitive: token = token.lower()
            # For python 2.7
    #         if token not in dataset.token_to_index.viewkeys():continue
            # For python 3.5
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
        print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
        print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
        print("number_of_token_lowercase_normalized_found: {0}".format(number_of_token_lowercase_normalized_found))
        print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
        #print("len(dataset.token_to_index): {0}".format(len(dataset.token_to_index)))
        #print("len(dataset.index_to_token): {0}".format(len(dataset.index_to_token)))
        print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))
    #     sess.run(tf.global_variables_initializer())
        sess.run(self.W.assign(initial_weights))
        print('Load embeddings completed')

