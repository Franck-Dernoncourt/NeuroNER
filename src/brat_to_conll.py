# -*- coding: utf-8 -*-
import os
import glob
import codecs
import spacy
import utils_nlp
import json
from pycorenlp import StanfordCoreNLP


def get_start_and_end_offset_of_token_from_spacy(token):
    start = token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    document = spacy_nlp(text)
    # sentences
    sentences = []
    for span in document.sents:
        sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token_dict['text'], 
                                                                                                                           token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
        sentences.append(sentence_tokens)
    return sentences

def get_stanford_annotations(text, core_nlp, port=9000, annotators='tokenize,ssplit,pos,lemma'):
    output = core_nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    if type(output) is str:
        output = json.loads(output, strict=False)
    return output

def get_sentences_and_tokens_from_stanford(text, core_nlp):
    stanford_output = get_stanford_annotations(text, core_nlp)
    sentences = []
    for sentence in stanford_output['sentences']:
        tokens = []
        for token in sentence['tokens']:
            token['start'] = int(token['characterOffsetBegin'])
            token['end'] = int(token['characterOffsetEnd'])
            token['text'] = text[token['start']:token['end']]
            if token['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token['text'].split(' ')) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token['text'], 
                                                                                                                           token['text'].replace(' ', '-')))
                token['text'] = token['text'].replace(' ', '-')
            tokens.append(token)
        sentences.append(tokens)
    return sentences

def get_entities_from_brat(text_filepath, annotation_filepath, verbose=False, scores = False, indices = False):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text =f.read()

    index_score_map = {}

    if scores:
        num_scores = len(scores)
        num_indices = len(indices)
        assert(num_scores == num_indices), "scores are not in sync with text!"

        cum_scores = [-1]*num_scores
        cum_scores[0] = scores[0]
        for i in range(1, num_scores):
            cum_scores[i] = cum_scores[i-1] + scores[i]

        index_score_map = {}
        i = 0
        while i < num_scores:
            st = indices[i][0]
            j = i
            while j < num_scores:
                ed = indices[j][1]
                try:
                    tag_score = float(cum_scores[j] - cum_scores[i]  + scores[i]) / (j-i+1) # tag score, considering avg.
                except:
                    import pdb; pdb.set_trace()
                key = "%s-%s" % (st, ed)
                index_score_map[key] = tag_score
                j += 1
            i += 1            
        
        
    if verbose: print("\ntext:\n{0}\n".format(text))

    # parse annotation file
    entities = []
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                entity = {}
                entity['id'] = id_anno
                entity['type'] = anno[1]
                entity['start'] = int(anno[2])
                entity['end'] = int(anno[3])
                entity['text'] = ' '.join(anno[4:])
                key = "%s-%s" % (anno[2], anno[3])
                entity_score = index_score_map.get(key, "no score from model")
                if entity_score != "no score from model":
                    entity_score = float("{0:.3f}".format(entity_score))
                entity['score'] = entity_score
                if verbose:
                    print("entity: {0}".format(entity))
                # Check compatibility between brat text and anootation
                if utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                    utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                    print("Warning: brat text and annotation do not match.")
                    print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                    print("\tanno: {0}".format(entity['text']))
                # add to entitys data
                entities.append(entity)
    if verbose: print("\n\n")
    
    return text, entities

def check_brat_annotation_and_text_compatibility(brat_folder):
    '''
    Check if brat annotation and text files are compatible.
    '''
    dataset_type =  os.path.basename(brat_folder)
    print("Checking the validity of BRAT-formatted {0} set... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(brat_folder, '*.txt')))
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # check if annotation file exists
        if not os.path.exists(annotation_filepath):
            raise IOError("Annotation file does not exist: {0}".format(annotation_filepath))
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
    print("Done.")

def brat_to_conll(input_folder, output_filepath, tokenizer, language):
    '''
    Assumes '.txt' and '.ann' files are in the input_folder.
    Checks for the compatibility between .txt and .ann at the same time.
    '''
    if tokenizer == 'spacy':
        spacy_nlp = spacy.load(language)
    elif tokenizer == 'stanford':
        core_nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    else:
        raise ValueError("tokenizer should be either 'spacy' or 'stanford'.")
    verbose = False
    dataset_type =  os.path.basename(input_folder)
    print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))
    output_file = codecs.open(output_filepath, 'w', 'utf-8')
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # create annotation file if it does not exist
        if not os.path.exists(annotation_filepath):
            codecs.open(annotation_filepath, 'w', 'UTF-8').close()

        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
        entities = sorted(entities, key=lambda entity:entity["start"])
        
        if tokenizer == 'spacy':
            sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)
        elif tokenizer == 'stanford':
            sentences = get_sentences_and_tokens_from_stanford(text, core_nlp)
        
        for sentence in sentences:
            inside = False
            previous_token_label = 'O'
            for token in sentence:
                token['label'] = 'O'
                for entity in entities:
                    if entity['start'] <= token['start'] < entity['end'] or \
                       entity['start'] < token['end'] <= entity['end'] or \
                       token['start'] < entity['start'] < entity['end'] < token['end']:

                        token['label'] = entity['type'].replace('-', '_') # Because the ANN doesn't support tag with '-' in it

                        break
                    elif token['end'] < entity['start']:
                        break
                        
                if len(entities) == 0:
                    entity={'end':0}
                if token['label'] == 'O':
                    gold_label = 'O'
                    inside = False
                elif inside and token['label'] == previous_token_label:
                    gold_label = 'I-{0}'.format(token['label'])
                else:
                    inside = True
                    gold_label = 'B-{0}'.format(token['label'])
                if token['end'] == entity['end']:
                    inside = False
                previous_token_label = token['label']
                if verbose: print('{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'], gold_label))
                output_file.write('{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'], gold_label))
            if verbose: print('\n')
            output_file.write('\n')

    output_file.close()
    print('Done.')
    if tokenizer == 'spacy':
        del spacy_nlp
    elif tokenizer == 'stanford':
        del core_nlp
