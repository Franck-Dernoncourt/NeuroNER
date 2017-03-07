'''
Miscellaneous utility functions for natural language processing
'''

import collections
import operator
import os
import time
import datetime
import codecs


def load_tokens_from_pretrained_token_embeddings(parameters):
    # Load embeddings
    #print('Load embeddings')
    file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
    count = -1
    case_sensitive = False
    tokens = set()
    number_of_loaded_word_vectors = 0
    for cur_line in file_input:
        count += 1
        #if count > 1000:break
        cur_line = cur_line.strip()
        cur_line = cur_line.split(' ')
        if len(cur_line)==0:continue
        token=cur_line[0]
        tokens.add(cur_line[0])
        number_of_loaded_word_vectors += 1
#         print("number_of_loaded_word_vectors: {0}".format(number_of_loaded_word_vectors))
    file_input.close()
    #print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))

    #print('Load embeddings completed')
    return tokens



def get_parsed_conll_output(conll_output_filepath):
    conll_output = [l.rstrip().replace('%','').replace(';','').replace(':', '').strip() for l in codecs.open(conll_output_filepath, 'r', 'utf8')]
    parsed_output = {}
    line = conll_output[1].split()
    parsed_output['all'] = {'precision': float(line[3]),
                            'recall':float(line[5]),
                            'f1':float(line[7])}
    total_support = 0
    for line in conll_output[2:]:
        line = line.split()
        phi_type = line[0].replace('_', '-')
        support = int(line[7])
        total_support += support
#         print("phi_type: {0}".format(phi_type))
        parsed_output[phi_type] = {'precision': float(line[2]),
                            'recall':float(line[4]),
                            'f1':float(line[6]),
                            'support':support}
    parsed_output['all']['support'] = total_support
#     print("parsed_output: {0}".format(parsed_output))
    return parsed_output