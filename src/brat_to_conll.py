'''
https://github.com/smilli/py-corenlp/blob/master/example.py

To connect to StanfordCoreNLP server
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000
'''
from pycorenlp import StanfordCoreNLP
import pprint
import utils_deid
import xml.etree.ElementTree
import utils
import os
import string
import numpy as np
from src.classes import Token
import json
from src import utils_deid
import time
import glob
import ConfigParser
import psycopg2
import pandas as pd
import unicodedata
import utils_nlp
import spacy
import codecs
# from spacy.symbols import ORTH, LEMMA, POS
nlp = spacy.load('en')

# nlp.tokenizer.add_special_case(u'shell',
#     [
#         {
#             ORTH: u'shell',
#             LEMMA: u'shell',
#             POS: u'NOUN'}
#      ])

# https://spacy.io/docs/usage/processing-text
# document = nlp(u'Once unregistered, the folder went away from the shell.')
# document = nlp(u'Both the techniques show that soot-in-oil exists as agglomerates with average size of 120nm. NTA is able to measure particles in polydisperse solutions and reports the size and volume distribution of soot-in-oil aggregates; it has the advantages of being fast and relatively low cost if compared with TEM.Nanoparticle Tracking Analysis (NTA) has been applied to characterising soot agglomerates of particles and compared with Transmission Electron Microscoscopy (TEM).')

# # tokens in document
# for token in document:
#     print('token.i: {2}\ttoken.idx: {0}\ttoken_end: {4}\ttoken.pos: {3:10}token.text: {1}'.
#       format(token.idx, token.text,token.i,token.pos_, token.idx+len(token)))


def get_start_and_end_offset_from_spacy_token(token):
    start = token.idx
    end = start + len(token)
    return start, end
 
def get_sentences_from_spacy(text):
    document = nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
#         print("span.start: {0}\tspan.end: {1}".format(span.start, span.end)) 
        sentence = [document[i] for i in range(span.start, span.end)]
        sentences.append(sentence)
    return sentences

'''
    for sentence in sentences:
        for token in sentence:
            start, end = get_start_and_end_offset_from_spacy_token(token)
            print('token.start: {0}\ttoken.end: {1}\ttoken.text: {2}'.format(start, end, token.text))
'''

def create_folder_if_not_exists(dir):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_sql_connection_and_cursor():
    conf_db = ConfigParser.ConfigParser()
    conf_db.read("database.ini")
    user = conf_db.get('database','user')
    password = conf_db.get('database','password')
    host = conf_db.get('database','host')

    connection = psycopg2.connect("dbname='mimic' user='{0}' host='{2}' password='{1}'".format(user, password, host))
    cursor = connection.cursor()
    return connection, cursor

def clean_name_string(string):
    if isinstance(string, basestring):
        return [chunk.lower() for chunk in string.split() if len(chunk) > 2]
    else:
        return []

# nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))
# def get_stanford_annotations(text, port=9000, annotators='tokenize,ssplit,pos,lemma'):
#     print('port: {0}'.format(port))
# #     fp = open("test2.txt")
# #     text = fp.read()
# #     print('text: {0}'.format(text[:9300]))
#     
#     '''
#     output = nlp.annotate(str(text), properties={
#     #output = nlp.annotate('this is a test', properties={
#     'timeout': '50000',
#     'annotators': 'tokenize,ssplit',
#     'outputFormat': 'json'
#     })
#     '''
#     output = nlp.annotate(text, properties={
#         "timeout": "10000",
#         "ssplit.newlineIsSentenceBreak": "two",
#         'annotators': annotators,
#         'outputFormat': 'json'
#     })
# 
#     print('output: {0}'.format(output))
#     
#     return output

def convert_stanford_output_to_ann_txt_old(output_filepath, xml_filename, stanford_output, phis):
    '''
    version without any dictionary or feature
    '''
    output_file = open(output_filepath, 'a')
    #output_file.write('-DOCSTART- -X- -X- O\n')
    for sentence in stanford_output['sentences']:
        inside=False
        output_file.write('\n')
        for token in sentence['tokens']:
            token['text'] = token['originalText'].replace(' ', '-')
            try:
                assert(" " not in token['text'])
            except AssertionError:
                print(token['text'])
            token['start'] = token['characterOffsetBegin']
            token['end'] = token['characterOffsetEnd']
            token['label'] = 'default_label'
            for phi in phis:
                if phi['start'] <= token['start'] < phi['end'] or \
                   phi['start'] < token['end'] <= phi['end'] or \
                   token['start'] < phi['start'] < phi['end'] < token['end']:
                    
                    token['label'] = phi['type'].replace('-', '_') # Because the ANN doesn't support tag with '-' in it
                    break
            if token['label'] == 'default_label':
                gold_label = 'O'
                inside = False
            elif inside:
                gold_label = 'I-{0}'.format(token['label'])
            else:
                inside = True
                gold_label = 'B-{0}'.format(token['label'])
            output_file.write('{0} {1} {2}_{3} {4}\n'.format(token['text'], xml_filename, token['start'], token['end'], gold_label))


def format_for_ann_old(dataset_base_filename, split):
    '''
    version without any dictionary or feature
    '''
    print("Started formatting for ann")
    start_time = time.time()
    filepaths = utils_deid.get_original_dataset_filepaths(dataset_base_filename, split=split)
    output_folder = os.path.join('ann', 'data', dataset_base_filename, 'stanford', split)
    create_folder_if_not_exists(output_folder)
    
    number_of_unicode_characters = 0
    open('unicode.txt','w').close()
    for dataset_type in filepaths:
#         if dataset_type == 'test':
#             continue
        output_filepath = os.path.join(output_folder, '{0}.txt'.format(dataset_type))
        open(output_filepath, 'w').close()
#         output_file.write('')
        for filepath in filepaths[dataset_type]:
            print("filepath: {0}".format(filepath))
#             filepath = '../data/datasets/original/i2b2deid2016/60_40/training-PHI-Gold-Set1/0666_gs.xml'
            xmldoc = xml.etree.ElementTree.parse(filepath).getroot()            
            # Get text
            text = xmldoc.findtext('TEXT')#.replace(u"\u2019", "'")
#             .encode('ascii', 'replace')
#             try:
#                 print("text: {0}".format(text))
#             except:
#                 number_of_unicode_characters += 1
#                 text_replaced = utils_nlp.normalize_unicode_text(text)
#                 with open('unicode.txt','a') as f:
#                     f.write(filepath+'\n\n')
# #                     f.write(text+'\n\n')
#                     f.write(text_replaced + '\n\n\n======================================================================{0}'.format(number_of_unicode_characters))
#                 text = text_replaced
#             text = text.replace(u"\u2019", "'")
#             text = text.replace(u"\u00E3", "a")
            
            # Get stanford output
            stanford_output = get_stanford_annotations(text, annotators='tokenize,ssplit')
            
            # Get PHI tags
            tags = xmldoc.findall('TAGS')[0] # [0] because there is only one <TAGS>...</TAGS>
            phis = []        
            for tag in tags:            
                #print(tag)
                phi = {}
                phi['main_type'] = tag.tag
                phi['type'] = tag.get('TYPE')
                phi['text'] = tag.get('text')#.replace(u"\u00E3", "a")
                phi['start'] = int(tag.get('start'))
                phi['end'] = int(tag.get('end'))
                phis.append(phi)
            
            xml_filename = utils.get_basename_without_extension(filepath)
            
            convert_stanford_output_to_ann_txt_old(output_filepath, xml_filename, stanford_output, phis)
#             0/0
    time_spent = time.time() - start_time
    print("Time spent formatting for ann: {0}".format(time_spent))



def get_dataset_folder_original(model='crf'):
    if model=='ann':
        return os.path.join('..', '..', 'data', 'datasets', 'original')
    elif model == 'crf':
        return os.path.join('..', '..', 'ner-deid', 'data', 'datasets', 'original')
        
def get_original_dataset_folders(dataset_base_filename, split='60_40', model='crf'):
    folder = {'train':[], 'dev':[], 'test':[]}
    dataset_folder_original = get_dataset_folder_original(model=model)
#     print("dataset_folder_original: {0}".format(dataset_folder_original))
#     print("split: {0}".format(split))
#     print("dataset_base_filename: {0}".format(dataset_base_filename))
    folder['train'] = os.path.join(dataset_folder_original, dataset_base_filename, split, 'training-PHI-Gold-Set1')
#     filepaths['train'] = sorted(glob.glob(os.path.join(train_folder, '*.xml')))
    folder['dev'] = os.path.join(dataset_folder_original, dataset_base_filename, split, 'training-PHI-Gold-Set2')
#     filepaths['dev'] = sorted(glob.glob(os.path.join(train_folder, '*.xml')))
    folder['test'] = os.path.join(dataset_folder_original, dataset_base_filename, split, 'testing-PHI-Gold-fixed')
#     filepaths['test'] = sorted(sorted(glob.glob(os.path.join(test_folder, '*.xml')))) 
    return folder

def get_original_dataset_filepaths(dataset_base_filename, split='60_40', model='crf'):
    filepaths = {'train':[], 'dev':[], 'test':[]}
    dataset_folder_original = get_dataset_folder_original(model=model)
#     print("dataset_folder_original: {0}".format(dataset_folder_original))
#     print("split: {0}".format(split))
#     print("dataset_base_filename: {0}".format(dataset_base_filename))
    train_folder = os.path.join(dataset_folder_original, dataset_base_filename, split, 'training-PHI-Gold-Set1')
    filepaths['train'] = sorted(glob.glob(os.path.join(train_folder, '*.xml')))
    train_folder = os.path.join(dataset_folder_original, dataset_base_filename, split, 'training-PHI-Gold-Set2')
    filepaths['dev'] = sorted(glob.glob(os.path.join(train_folder, '*.xml')))
    test_folder = os.path.join(dataset_folder_original, dataset_base_filename, split, 'testing-PHI-Gold-fixed')
    filepaths['test'] = sorted(sorted(glob.glob(os.path.join(test_folder, '*.xml')))) 
    return filepaths


def get_basename_without_extension(filepath):
    '''
    Getting the basename of the filepath without the extension
    E.g. 'data/formatted/movie_reviews.pickle' -> 'movie_reviews' 
    '''
    return os.path.basename(os.path.splitext(filepath)[0])

if __name__ == '__main__': 
    # XML to TXT
# #     start_time = time.time()
#     dataset_base_filename = 'i2b2deid2014'
#     filepaths = get_original_dataset_filepaths(dataset_base_filename)
#     output_folder = '/Users/jjylee/Documents/workspace/nlp/brat-master/tools/out/train'
# #     output_folder = os.path.join('ann', 'data', t)
# #     create_folder_if_not_exists(output_folder)    
#     for filepath in filepaths['train']:
#         print("filepath: {0}".format(filepath))
#     #             filepath = '../data/datasets/original/i2b2deid2016/60_40/training-PHI-Gold-Set1/0666_gs.xml'
#         xmldoc = xml.etree.ElementTree.parse(filepath).getroot()            
#         # Get text
#         text = xmldoc.findtext('TEXT')#.replace(u"\u2019", "'")
#         outfn = os.path.join(output_folder, get_basename_without_extension(filepath) + '.txt')
# #         out = open(outfn, "wt")
#         out = codecs.open(outfn, "wt", "UTF-8")
#         out.write(text)
#         out.close()

    # TEST SPACY
    output_folder = '/Users/jjylee/Documents/workspace/nlp/brat-master/tools/out/train'
    for filepath in glob.glob(os.path.join(output_folder, '250-01.txt')):
        with codecs.open(filepath, "rt", "UTF-8") as f:
            text = f.read()
        sentences = get_sentences_from_spacy(text)
        for sentence in sentences:
            tokens = [token.text for token in sentence]
            print(' '.join(tokens).encode("UTF-8"))
            print('\n')
#             for token in sentence:
#                 start, end = get_start_and_end_offset_from_spacy_token(token)
#                 print('token.start: {0}\ttoken.end: {1}\ttoken.text: {2}'.format(start, end, token.text))
        
    0/0
#     input_filepath = '/Users/jjylee/Documents/workspace/nlp/ner-deid/src/ann/data/mimic/stanford/140_20_40_names_dictionary_decimal/train.txt'
#     output_filepath = '/Users/jjylee/Documents/workspace/nlp/ner-deid/src/ann/data/mimic/stanford/140_20_40/train.txt'
#     input_filepath = '/Users/jjylee/Documents/workspace/nlp/ner-deid/src/ann/data/mimic/stanford/140_20_40_names_dictionary_decimal/dev.txt'
#     output_filepath = '/Users/jjylee/Documents/workspace/nlp/ner-deid/src/ann/data/mimic/stanford/140_20_40/dev.txt'
#     input_filepath = '/Users/jjylee/Documents/workspace/nlp/ner-deid/src/ann/data/mimic/stanford/140_20_40_names_dictionary_decimal/test.txt'
#     output_filepath = '/Users/jjylee/Documents/workspace/nlp/ner-deid/src/ann/data/mimic/stanford/140_20_40/test.txt'
#     convert_decimal_to_binary(input_filepath, output_filepath)
#     0/0
    dictionary_folder = os.path.join('..', 'data', 'DeidDictionaries-6-26-2016', 'lists-internal')
    dictionary_filepaths = [os.path.join(dictionary_folder, 'doctor_first_names.txt'), os.path.join(dictionary_folder, 'doctor_last_names.txt')]
    dataset_base_filename = 'mimic'
    dataset_base_filename = 'i2b2deid2014'
#     dataset_base_filename = 'i2b2deid2016'
    tokenization_identifier = 7 # i2b2deid2014
    tokenization_identifier = 11 # mimic
    # Note: tokenization id of 10 and 11 are reserved for mimic data (as i2b2 data is not big enough).
    token_id_to_split = {8:'40_20_40', 9:'20_20_40', 10:'100_20_40', 11:'140_20_40', 12:'10_20_40', 13:'5_20_40'}
    for tokenization_identifier in [13]:
        
        if tokenization_identifier == 7:
            if dataset_base_filename in ['i2b2deid2014', 'i2b2deid2016']:
                split = '60_40'
            elif dataset_base_filename == 'mimic':
                split = '60_20_40'
        elif tokenization_identifier >= 8:
            split = token_id_to_split[tokenization_identifier]
                
#         format_for_crf(dataset_base_filename, tokenization_identifier, split)
#         format_for_ann(dataset_base_filename, split, dictionary_filepaths=None)
#         format_for_ann(dataset_base_filename, split, dictionary_filepaths=dictionary_filepaths)
        format_for_ann_old(dataset_base_filename, split)
            

