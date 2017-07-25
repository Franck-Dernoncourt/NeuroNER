import codecs
import collections
import multiprocessing
import os
import pickle
import random
import re
import time
from functools import partial

import sklearn.preprocessing

import utils
import utils_nlp


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def _parse_dataset(self, dataset_filepath):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)

        line_count = -1
        tokens = []
        labels = []
        new_token_sequence = []
        new_label_sequence = []
        if dataset_filepath:
            f = codecs.open(dataset_filepath, 'r', 'UTF-8')
            for line in f:
                line_count += 1
                line = line.strip().split(' ')
                if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                    if len(new_token_sequence) > 0:
                        labels.append(new_label_sequence)
                        tokens.append(new_token_sequence)
                        new_token_sequence = []
                        new_label_sequence = []
                    continue
                token = str(line[0])
                label = str(line[-1])
                token_count[token] += 1
                label_count[label] += 1

                new_token_sequence.append(token)
                new_label_sequence.append(label)

                for character in token:
                    character_count[character] += 1

                if self.debug and line_count > 200: break  # for debugging purposes

            if len(new_token_sequence) > 0:
                labels.append(new_label_sequence)
                tokens.append(new_token_sequence)
            f.close()
        return labels, tokens, token_count, label_count, character_count

    def _token_to_indices(self, token_sequence, token_to_index, character_to_index):
        token_index = [token_to_index.get(token, self.UNK_TOKEN_INDEX) for token in token_sequence]
        characters = [list(token) for token in token_sequence]
        character_index = [
            [character_to_index.get(character, random.randint(1, max(self.index_to_character.keys()))) for character in
             token] for token in token_sequence]
        token_lengths = [len(token) for token in token_sequence]

        longest_token_length_in_sequence = max(token_lengths)
        character_index_padded = [
            utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_CHARACTER_INDEX) for
            temp_token_indices in character_index]

        return token_index, characters, character_index, token_lengths, character_index_padded

    def _convert_to_indices(self, dataset_types, parameters):
        tokens = self.tokens
        labels = self.labels
        token_to_index = self.token_to_index
        character_to_index = self.character_to_index
        label_to_index = self.label_to_index
        index_to_label = self.index_to_label

        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        characters = {}
        token_lengths = {}
        character_indices = {}
        character_indices_padded = {}
        pool = multiprocessing.Pool(parameters['number_of_cpu_threads'])

        for dataset_type in dataset_types:
            token_indices[dataset_type] = []
            characters[dataset_type] = []
            character_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            character_indices_padded[dataset_type] = []

            token_index, dataset_characters, character_index, dataset_token_lengths, character_index_padded = \
                zip(*pool.map(partial(self._token_to_indices, token_to_index=token_to_index,
                                      character_to_index=character_to_index), tokens[dataset_type]))

            token_indices[dataset_type] = token_index
            characters[dataset_type] = dataset_characters
            character_indices[dataset_type] = character_index
            token_lengths[dataset_type] = dataset_token_lengths
            character_indices_padded[dataset_type] = character_index_padded

            label_indices[dataset_type] = []
            for label_sequence in labels[dataset_type]:
                label_indices[dataset_type].append([label_to_index[label] for label in label_sequence])

        pool.close()

        if self.verbose:
            print('token_lengths[\'train\'][0][0:10]: {0}'.format(token_lengths['train'][0][0:10]))
        if self.verbose:
            print('characters[\'train\'][0][0:10]: {0}'.format(characters['train'][0][0:10]))
        if self.verbose:
            print('token_indices[\'train\'][0:10]: {0}'.format(token_indices['train'][0:10]))
        if self.verbose:
            print('label_indices[\'train\'][0:10]: {0}'.format(label_indices['train'][0:10]))
        if self.verbose:
            print('character_indices[\'train\'][0][0:10]: {0}'.format(character_indices['train'][0][0:10]))
        if self.verbose:
            print('character_indices_padded[\'train\'][0][0:10]: {0}'.format(
                character_indices_padded['train'][0][0:10]))  # Vectorize the labels
        # [Numpy 1-hot array](http://stackoverflow.com/a/42263603/395857)
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(max(index_to_label.keys()) + 1))
        label_vector_indices = {}
        for dataset_type in dataset_types:
            label_vector_indices[dataset_type] = []
            for label_indices_sequence in label_indices[dataset_type]:
                label_vector_indices[dataset_type].append(label_binarizer.transform(label_indices_sequence))

        if self.verbose:
            print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))
        if self.verbose:
            print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))

        return token_indices, label_indices, character_indices_padded, character_indices, token_lengths, characters, label_vector_indices

    def update_dataset(self, dataset_filepaths, dataset_types, parameters):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        Overwrites the data of type specified in dataset_types using the existing token_to_index, character_to_index, and label_to_index mappings. 
        '''
        for dataset_type in dataset_types:
            self.labels[dataset_type], self.tokens[dataset_type], _, _, _ = self._parse_dataset(
                dataset_filepaths.get(dataset_type, None))

        token_indices, label_indices, character_indices_padded, character_indices, token_lengths, characters, label_vector_indices = self._convert_to_indices(
            dataset_types, parameters)

        self.token_indices.update(token_indices)
        self.label_indices.update(label_indices)
        self.character_indices_padded.update(character_indices_padded)
        self.character_indices.update(character_indices)
        self.token_lengths.update(token_lengths)
        self.characters.update(characters)
        self.label_vector_indices.update(label_vector_indices)

    def load_dataset(self, dataset_filepaths, parameters, token_to_vector=None):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        '''
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)
        if parameters['token_pretrained_embedding_filepath'] != '':
            if token_to_vector == None:
                token_to_vector = utils_nlp.load_pretrained_token_embeddings(parameters)
        else:
            token_to_vector = {}
        if self.verbose: print("len(token_to_vector): {0}".format(len(token_to_vector)))

        # Load pretraining dataset to ensure that index to label is compatible to the pretrained model,
        #   and that token embeddings that are learned in the pretrained model are loaded properly.
        all_tokens_in_pretraining_dataset = []
        all_characters_in_pretraining_dataset = []
        if parameters['use_pretrained_model']:
            pretraining_dataset = pickle.load(
                open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
            all_tokens_in_pretraining_dataset = pretraining_dataset.index_to_token.values()
            all_characters_in_pretraining_dataset = pretraining_dataset.index_to_character.values()

        remap_to_unk_count_threshold = 1
        self.UNK_TOKEN_INDEX = 0
        self.PADDING_CHARACTER_INDEX = 0
        self.tokens_mapped_to_unk = []
        self.UNK = 'UNK'
        self.unique_labels = []
        labels = {}
        tokens = {}
        label_count = {}
        token_count = {}
        character_count = {}
        for dataset_type in ['train', 'valid', 'test', 'deploy']:
            labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type], \
            character_count[dataset_type] \
                = self._parse_dataset(dataset_filepaths.get(dataset_type, None))

            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(
                token_count['test'].keys()) + list(token_count['deploy'].keys()):
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token] + token_count['test'][
                token] + token_count['deploy'][token]

        if parameters['load_all_pretrained_token_embeddings']:
            for token in token_to_vector:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1
            for token in all_tokens_in_pretraining_dataset:
                if token not in token_count['all']:
                    token_count['all'][token] = -1
                    token_count['train'][token] = -1

        character_count['all'] = {}
        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()) + list(
                character_count['test'].keys()) + list(character_count['deploy'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][
                character] + character_count['test'][character] + character_count['deploy'][character]

        for character in all_characters_in_pretraining_dataset:
            if character not in character_count['all']:
                character_count['all'][character] = -1
                character_count['train'][character] = -1

        for dataset_type in dataset_filepaths.keys():
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(
                label_count['test'].keys()) + list(label_count['deploy'].keys()):
            label_count['all'][character] = label_count['train'][character] + label_count['valid'][character] + \
                                            label_count['test'][character] + label_count['deploy'][character]

        token_count['all'] = utils.order_dictionary(token_count['all'], 'value_key', reverse=True)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)
        character_count['all'] = utils.order_dictionary(character_count['all'], 'value', reverse=True)
        if self.verbose: print('character_count[\'all\']: {0}'.format(character_count['all']))

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
        if self.verbose: print(
            "parameters['remap_unknown_tokens_to_unk']: {0}".format(parameters['remap_unknown_tokens_to_unk']))
        if self.verbose: print("len(token_count['train'].keys()): {0}".format(len(token_count['train'].keys())))
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1

            if parameters['remap_unknown_tokens_to_unk'] == 1 and \
                    (token_count['train'][token] == 0 or \
                             parameters['load_only_pretrained_token_embeddings']) and \
                    not utils_nlp.is_token_in_pretrained_embeddings(token, token_to_vector, parameters) and \
                            token not in all_tokens_in_pretraining_dataset:
                if self.verbose: print("token: {0}".format(token))
                if self.verbose: print("token.lower(): {0}".format(token.lower()))
                if self.verbose: print("re.sub('\d', '0', token.lower()): {0}".format(re.sub('\d', '0', token.lower())))
                token_to_index[token] = self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1
        if self.verbose: print("number_of_unknown_tokens: {0}".format(number_of_unknown_tokens))

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
        if self.verbose: print("len(token_count['train']): {0}".format(len(token_count['train'])))
        if self.verbose: print("len(infrequent_token_indices): {0}".format(len(infrequent_token_indices)))

        # Ensure that both B- and I- versions exist for each label
        labels_without_bio = set()
        for label in label_count['all'].keys():
            new_label = utils_nlp.remove_bio_from_label_name(label)
            labels_without_bio.add(new_label)
        for label in labels_without_bio:
            if label == 'O':
                continue
            if parameters['tagging_format'] == 'bioes':
                prefixes = ['B-', 'I-', 'E-', 'S-']
            else:
                prefixes = ['B-', 'I-']
            for prefix in prefixes:
                l = prefix + label
                if l not in label_count['all']:
                    label_count['all'][l] = 0
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)

        if parameters['use_pretrained_model']:
            self.unique_labels = sorted(list(pretraining_dataset.label_to_index.keys()))
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError("The label {0} does not exist in the pretraining dataset. ".format(label) +
                                         "Please ensure that only the following labels exist in the dataset: {0}".format(
                                             ', '.join(self.unique_labels)))
            label_to_index = pretraining_dataset.label_to_index.copy()
        else:
            label_to_index = {}
            iteration_number = 0
            for label, count in label_count['all'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1
                self.unique_labels.append(label)

        if self.verbose: print('self.unique_labels: {0}'.format(self.unique_labels))

        character_to_index = {}
        iteration_number = 0
        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX: iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1

        if self.verbose: print('token_count[\'train\'][0:10]: {0}'.format(list(token_count['train'].items())[0:10]))
        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse=False)
        if self.verbose: print('token_to_index: {0}'.format(token_to_index))
        index_to_token = utils.reverse_dictionary(token_to_index)
        if parameters['remap_unknown_tokens_to_unk'] == 1: index_to_token[self.UNK_TOKEN_INDEX] = self.UNK
        if self.verbose: print('index_to_token: {0}'.format(index_to_token))

        if self.verbose: print('label_count[\'train\']: {0}'.format(label_count['train']))
        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse=False)
        if self.verbose: print('label_to_index: {0}'.format(label_to_index))
        index_to_label = utils.reverse_dictionary(label_to_index)
        if self.verbose: print('index_to_label: {0}'.format(index_to_label))

        character_to_index = utils.order_dictionary(character_to_index, 'value', reverse=False)
        index_to_character = utils.reverse_dictionary(character_to_index)
        if self.verbose: print('character_to_index: {0}'.format(character_to_index))
        if self.verbose: print('index_to_character: {0}'.format(index_to_character))

        if self.verbose: print('labels[\'train\'][0:10]: {0}'.format(labels['train'][0:10]))
        if self.verbose: print('tokens[\'train\'][0:10]: {0}'.format(tokens['train'][0:10]))

        if self.verbose:
            # Print sequences of length 1 in train set
            for token_sequence, label_sequence in zip(tokens['train'], labels['train']):
                if len(label_sequence) == 1 and label_sequence[0] != 'O':
                    print("{0}\t{1}".format(token_sequence[0], label_sequence[0]))

        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        if self.verbose: print("len(self.token_to_index): {0}".format(len(self.token_to_index)))
        if self.verbose: print("len(self.index_to_token): {0}".format(len(self.index_to_token)))
        self.tokens = tokens
        self.labels = labels

        token_indices, label_indices, character_indices_padded, character_indices, token_lengths, characters, label_vector_indices = \
            self._convert_to_indices(dataset_filepaths.keys(), parameters)

        self.token_indices = token_indices
        self.label_indices = label_indices
        self.character_indices_padded = character_indices_padded
        self.character_indices = character_indices
        self.token_lengths = token_lengths
        self.characters = characters
        self.label_vector_indices = label_vector_indices

        self.number_of_classes = max(self.index_to_label.keys()) + 1
        self.vocabulary_size = max(self.index_to_token.keys()) + 1
        self.alphabet_size = max(self.index_to_character.keys()) + 1
        if self.verbose: print("self.number_of_classes: {0}".format(self.number_of_classes))
        if self.verbose: print("self.alphabet_size: {0}".format(self.alphabet_size))
        if self.verbose: print("self.vocabulary_size: {0}".format(self.vocabulary_size))

        # unique_labels_of_interest is used to compute F1-scores.
        self.unique_labels_of_interest = list(self.unique_labels)
        self.unique_labels_of_interest.remove('O')

        self.unique_label_indices_of_interest = []
        for lab in self.unique_labels_of_interest:
            self.unique_label_indices_of_interest.append(label_to_index[lab])

        self.infrequent_token_indices = infrequent_token_indices

        if self.verbose: print('self.unique_labels_of_interest: {0}'.format(self.unique_labels_of_interest))
        if self.verbose: print(
            'self.unique_label_indices_of_interest: {0}'.format(self.unique_label_indices_of_interest))

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

        return token_to_vector
