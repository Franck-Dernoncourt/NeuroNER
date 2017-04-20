'''
Miscellaneous utility functions for natural language processing
'''
import codecs
import re

def load_tokens_from_pretrained_token_embeddings(parameters):
    file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
    count = -1
    tokens = set()
    number_of_loaded_word_vectors = 0
    for cur_line in file_input:
        count += 1
        cur_line = cur_line.strip()
        cur_line = cur_line.split(' ')
        if len(cur_line)==0:continue
        token=cur_line[0]
        tokens.add(token)
        number_of_loaded_word_vectors += 1
    file_input.close()
    return tokens

def load_pretrained_token_embeddings(parameters):
    file_input = codecs.open(parameters['token_pretrained_embedding_filepath'], 'r', 'UTF-8')
    count = -1
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
    file_input.close()
    return token_to_vector

def is_token_in_pretrained_embeddings(token, all_pretrained_tokens, parameters):
    return token in all_pretrained_tokens or \
        parameters['check_for_lowercase'] and token.lower() in all_pretrained_tokens or \
        parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token) in all_pretrained_tokens or \
        parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token.lower()) in all_pretrained_tokens


def get_parsed_conll_output(conll_output_filepath):
    conll_output = [l.rstrip().replace('%','').replace(';','').replace(':', '').strip() for l in codecs.open(conll_output_filepath, 'r', 'utf8')]
    parsed_output = {}
    line = conll_output[1].split()
    parsed_output['all'] = {'accuracy': float(line[1]),
                            'precision': float(line[3]),
                            'recall':float(line[5]),
                            'f1':float(line[7])}
    total_support = 0
    for line in conll_output[2:]:
        line = line.split()
        phi_type = line[0].replace('_', '-')
        support = int(line[7])
        total_support += support
        parsed_output[phi_type] = {'precision': float(line[2]),
                            'recall':float(line[4]),
                            'f1':float(line[6]),
                            'support':support}
    parsed_output['all']['support'] = total_support
    return parsed_output

def remove_bio_from_label_name(label_name):
    if label_name[:2] in ['B-', 'I-']:
        new_label_name = label_name[2:]
    else:
        assert(label_name == 'O')
        new_label_name = label_name
    return new_label_name

def replace_unicode_whitespaces_with_ascii_whitespace(string):
    return ' '.join(string.split())