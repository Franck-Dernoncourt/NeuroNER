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

def get_entities_from_brat(text_filepath, annotation_filepath, split_discontinuous, verbose=False):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text = f.read()
    if verbose: print("\ntext:\n{0}\n".format(text))
    # parse annotation file
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        ann = f.read().splitlines()
    entities = parse_brat_annotations(ann, text, split_discontinuous)
    return text, entities

def parse_brat_annotations(ann, text, split_discontinuous, verbose=False):
    '''
    Parse the contents of brat annotation files (.ann and .txt) for entities.

    For compatibility with discontinuous annotations in brat >= 1.3, entity text
    is from slicing the text per the annotation offsets, rather than from the
    annotation reference text.

    :param split_discontinuous: If True, split each discontinuous annotation
    (brat >= 1.3) into separate annotations. If False, join the fragments into a
    continuous annotation that starts with the first fragment and ends with the
    last.
    '''
    ann = [line for line in ann if line[0] == 'T']
    entities = []
    for line in ann:
        brat_id, entity_type, offsets, line_text = split_ann(line)
        if split_discontinuous:
            offsets = [(min(pair[0] for pair in offsets), max(pair[1] for pair in offsets))]
        for start, end in offsets:
            entity = {
                    'id': brat_id,
                    'type': entity_type,
                    'start': start,
                    'end': end,
                    'text': text[start:end],
                    }
            entities.append(entity)
    return entities

def split_ann(line):
    '''
    Split a line from a brat .ann file into its components.

    In a line from an .ann file that represents a text-bound annotation, a
    sequential numeric ID prefixed with 'T' is followed by a tab, then an entity
    type, a space, and at least one pair of space-delimited offsets.

    Each of the offset pairs gives the range of a zero-indexed annotation span,
    [start, end). With brat >= 1.3, annotations can be composed of discontinuous
    "fragments." Multiple offset pairs are delimited by semicolons.

    After the offset pair(s) and a tab comes the reference text. For
    discontinous annotations, the reference text is the concatenation of the
    fragments delimited by spaces. Note that this means that the annotated
    entity as it appears in the text cannot necessarily be recovered from the
    reference text.

    See http://brat.nlplab.org/standoff.html.

    Return:
    - brat_id: brat annotation ID, e.g. 'T1'.
    - entity_type: entity type, e.g. 'Org'.
    - offsets: list of int offset tuples, e.g., [(0, 4)] for a continuous
      annotation or [(0, 4), (6, 9)] for a discontinuous annotation with 2
      fragments.
    - line_text: reference text, e.g. 'Lorem ipsum'.
    '''
    brat_id, type_offsets, line_text = line.split('\t', maxsplit=2)
    entity_type, offsets = type_offsets.split(maxsplit=1)
    offsets = [pair.split() for pair in offsets.split(';')]
    offsets = [(int(pair[0]), int(pair[1])) for pair in offsets]
    return brat_id, entity_type, offsets, line_text

def check_brat_annotation_and_text_compatibility(brat_folder, split_discontinuous):
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
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath, split_discontinuous)
        for entity in entities:
            if utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                print('Warning: brat text and annotation do not match:')
                print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                print("\tanno: {0}".format(entity['text']))
    print("Done.")

def brat_to_conll(input_folder, output_filepath, tokenizer, language, split_discontinuous):
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

        text, entities = get_entities_from_brat(text_filepath,
                annotation_filepath, split_discontinuous)
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
