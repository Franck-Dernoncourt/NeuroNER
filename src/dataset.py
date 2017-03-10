import sklearn.preprocessing
import utils
import collections
import codecs
import utils_nlp
import re
import time
import token


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name=''):
        self.name = name
        self.verbose = False

    def _parse_dataset(self, dataset_filepath,dataset_type):#,all_pretrained_tokens,previous_token_count):
        '''

        '''
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)

        # http://stackoverflow.com/questions/491921/unicode-utf-8-reading-and-writing-to-files-in-python
        f = codecs.open(dataset_filepath, 'r', 'UTF-8')
        line_count = -1
        tokens = []
        labels = []
        characters = []
        token_lengths = []
        new_token_sequence = []
        new_label_sequence = []
        for line in f:
            line_count += 1
            #print(line, end='')
            line = line.strip().split(' ')
            if len(line) == 0 or len(line[0]) == 0 or (line_count == 0 and 'DOCSTART' in line[0]):
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

            #if dataset_type in ['valid', 'test'] and token not in previous_token_count['train'].keys() and token not in all_pretrained_tokens:
            #    token = self.UNK

            new_token_sequence.append(token)
            new_label_sequence.append(label)

            # update the character_count
            for character in token:
                character_count[character] += 1

            #if line_count > 200: break# for debugging purposes



        if len(new_token_sequence) > 0:
            labels.append(new_label_sequence)
            tokens.append(new_token_sequence)

        #token_count = utils.order_dictionary(token_count, 'value', reverse = True)
        #label_count = utils.order_dictionary(label_count, 'key', reverse = False)

        #token_indices.append([token_to_index[token] for token in token_sequence])



        return labels, tokens, token_count, label_count, character_count

    def load_dataset(self, dataset_filepaths, parameters):
        '''
        args:
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test'
        http://stackoverflow.com/questions/27416164/what-is-conll-data-format
        '''
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)
        all_pretrained_tokens = None
        if parameters['token_pretrained_embedding_filepath'] != '':
            all_pretrained_tokens = utils_nlp.load_tokens_from_pretrained_token_embeddings(parameters)
        if self.verbose: print("len(all_pretrained_tokens): {0}".format(len(all_pretrained_tokens)))

        remap_to_unk_count_threshold = 1
        #if ['train'] not in dataset_filepaths.keys(): raise ValueError('')
        self.UNK_TOKEN_INDEX = 0
        self.PADDING_CHARACTER_INDEX = 0
        self.tokens_mapped_to_unk = []
        self.UNK = 'UNK'
        self.unique_labels = []
        labels = {}
        tokens = {}
        characters = {}
        token_lengths = {}
        label_count = {}
        token_count = {}
        character_count = {}
        for dataset_type in ['train', 'valid', 'test']:
            labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type], \
                character_count[dataset_type] = self._parse_dataset(dataset_filepaths[dataset_type],dataset_type)#,all_pretrained_tokens,token_count)
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        token_count['all'] = {} # utils.merge_dictionaries()
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(token_count['test'].keys()):
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token] + token_count['test'][token]

        for dataset_type in ['train', 'valid', 'test']:
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        character_count['all'] = {} # utils.merge_dictionaries()
        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()) + list(character_count['test'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][character] + character_count['test'][character]

        label_count['all'] = {} # utils.merge_dictionaries()
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(label_count['test'].keys()):
            label_count['all'][character] = label_count['train'][character] + label_count['valid'][character] + label_count['test'][character]

        token_count['all'] = utils.order_dictionary(token_count['all'], 'value', reverse = True)
        #label_count['train'] = utils.order_dictionary(label_count['train'], 'key', reverse = False)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse = False)
        label_count['train'] = utils.order_dictionary(label_count['train'], 'key', reverse = False)
        character_count['all'] = utils.order_dictionary(character_count['all'], 'value', reverse = True)
        if self.verbose: print('character_count[\'all\']: {0}'.format(character_count['all']))

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
#         if self.verbose: print("parameters['remove_unknown_tokens']: {0}".format(parameters['remove_unknown_tokens']))
#         if self.verbose: print("len(token_count['train'].keys()): {0}".format(len(token_count['train'].keys())))
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1

            if parameters['remove_unknown_tokens'] == 1 and \
                token_count['train'][token] == 0 and \
                (all_pretrained_tokens == None or \
                token not in all_pretrained_tokens and \
                token.lower() not in all_pretrained_tokens and \
                re.sub('\d', '0', token.lower()) not in all_pretrained_tokens):#all( [x not in all_pretrained_tokens for x in [ token, token.lower(), re.sub('\d', '0', token.lower()) ]]):

#                         if self.verbose: print("token: {0}".format(token))
#                         if self.verbose: print("token.lower(): {0}".format(token.lower()))
#                         if self.verbose: print("re.sub('\d', '0', token.lower()): {0}".format(re.sub('\d', '0', token.lower())))
#                         assert(token not in )
#                         assert(token.lower() not in all_pretrained_tokens)
#                         assert(re.sub('\d', '0', token.lower()) not in all_pretrained_tokens)
                token_to_index[token] =  self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1
        if self.verbose: print("number_of_unknown_tokens: {0}".format(number_of_unknown_tokens))
#         0/0

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
        if self.verbose: print("len(token_count['train']): {0}".format(len(token_count['train'])))
        if self.verbose: print("len(infrequent_token_indices): {0}".format(len(infrequent_token_indices)))

        label_to_index = {}
        iteration_number = 0
        #for label, count in label_count['train'].items():
        for label, count in label_count['all'].items():
            label_to_index[label] = iteration_number
            iteration_number += 1
            self.unique_labels.append(label)


        #for label, count in label_count['train'].items():
        #    self.unique_labels.append(label)

        if self.verbose: print('self.unique_labels: {0}'.format(self.unique_labels))

        character_to_index = {}
        iteration_number = 0
        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX: iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1


        if self.verbose: print('token_count[\'train\'][0:10]: {0}'.format(list(token_count['train'].items())[0:10]))
        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse = False)
        #if self.verbose: print('token_to_index[0:10]: {0}'.format(token_to_index[0:10]))
        index_to_token = utils.reverse_dictionary(token_to_index)
        if parameters['remove_unknown_tokens'] == 1: index_to_token[self.UNK_TOKEN_INDEX] = self.UNK
        #if self.verbose: print('index_to_token[0:10]: {0}'.format(index_to_token[0:10]))

        #if self.verbose: print('label_count[\'train\']: {0}'.format(label_count['train']))
        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse = False)
        if self.verbose: print('label_to_index: {0}'.format(label_to_index))
        index_to_label = utils.reverse_dictionary(label_to_index)
        if self.verbose: print('index_to_label: {0}'.format(index_to_label))

        index_to_character = utils.reverse_dictionary(character_to_index)
        if self.verbose: print('character_to_index: {0}'.format(character_to_index))
        if self.verbose: print('index_to_character: {0}'.format(index_to_character))


        if self.verbose: print('labels[\'train\'][0:10]: {0}'.format(labels['train'][0:10]))
        if self.verbose: print('tokens[\'train\'][0:10]: {0}'.format(tokens['train'][0:10]))

        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        character_indices = {}
        character_indices_padded = {}
        for dataset_type in ['train', 'valid', 'test']:
            token_indices[dataset_type] = []
            characters[dataset_type] = []
            character_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            character_indices_padded[dataset_type] = []
            for token_sequence in tokens[dataset_type]:
                token_indices[dataset_type].append([token_to_index[token] for token in token_sequence])
                characters[dataset_type].append([list(token) for token in token_sequence])
                character_indices[dataset_type].append([[character_to_index[character] for character in token] for token in token_sequence])
                token_lengths[dataset_type].append([len(token) for token in token_sequence])

                longest_token_length_in_sequence = max(token_lengths[dataset_type][-1])
                character_indices_padded[dataset_type].append([ utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_CHARACTER_INDEX)
                                                                for temp_token_indices in character_indices[dataset_type][-1]])

            label_indices[dataset_type] = []
            for label_sequence in labels[dataset_type]:
                label_indices[dataset_type].append([label_to_index[label] for label in label_sequence])




        if self.verbose: print('token_lengths[\'train\'][0][0:10]: {0}'.format(token_lengths['train'][0][0:10]))
        if self.verbose: print('characters[\'train\'][0][0:10]: {0}'.format(characters['train'][0][0:10]))
        if self.verbose: print('token_indices[\'train\'][0:10]: {0}'.format(token_indices['train'][0:10]))
        if self.verbose: print('label_indices[\'train\'][0:10]: {0}'.format(label_indices['train'][0:10]))
        if self.verbose: print('character_indices[\'train\'][0][0:10]: {0}'.format(character_indices['train'][0][0:10]))
        if self.verbose: print('character_indices_padded[\'train\'][0][0:10]: {0}'.format(character_indices_padded['train'][0][0:10]))

        #  Vectorize the labels
        # [Numpy 1-hot array](http://stackoverflow.com/a/42263603/395857)
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(max(index_to_label.keys())+1))
        label_vector_indices = {}
        for dataset_type in ['train', 'valid', 'test']:
            label_vector_indices[dataset_type] = []
            for label_indices_sequence in label_indices[dataset_type]:
                label_vector_indices[dataset_type].append(label_binarizer.transform(label_indices_sequence))

        if self.verbose: print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))

        if self.verbose: print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.token_indices = token_indices
        self.label_indices = label_indices
        self.character_indices_padded = character_indices_padded
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.character_indices = character_indices
        self.token_lengths = token_lengths
        self.characters = characters
        self.tokens = tokens
        self.labels = labels
        self.label_vector_indices = label_vector_indices
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        if self.verbose: print("len(self.token_to_index): {0}".format(len(self.token_to_index)))
        if self.verbose: print("len(self.index_to_token): {0}".format(len(self.index_to_token)))

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
        if self.verbose: print('self.unique_label_indices_of_interest: {0}'.format(self.unique_label_indices_of_interest))

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

