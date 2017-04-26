import os
import codecs
import utils
import glob
import shutil
import utils_nlp

def generate_reference_text_file_for_conll(conll_input_filepath, conll_output_filepath, text_folder):
    '''
    generates reference text files and adds the corresponding filename and token offsets to conll file.
    
    conll_input_filepath: path to a conll-formatted file without filename and token offsets
    text_folder: folder to write the reference text file to
    '''
    dataset_type =  utils.get_basename_without_extension(conll_input_filepath)
    conll_file = codecs.open(conll_input_filepath, 'r', 'UTF-8')   
    utils.create_folder_if_not_exists(text_folder)
    text = ''
    new_conll_string = ''
    character_index = 0
    document_count = 0
    text_base_filename = '{0}_text_{1}'.format(dataset_type, str(document_count).zfill(5))
    for line in conll_file:
        split_line = line.strip().split(' ')
        # New document
        if '-DOCSTART-' in split_line[0]:
            new_conll_string += line
            if len(text) != 0:
                with codecs.open(os.path.join(text_folder, '{0}.txt'.format(text_base_filename)), 'w', 'UTF-8') as f:
                    f.write(text)
            text = ''
            character_index = 0
            document_count += 1
            text_base_filename = '{0}_text_{1}'.format(dataset_type, str(document_count).zfill(5))
            continue            
        # New sentence
        elif len(split_line) == 0 or len(split_line[0]) == 0:
            new_conll_string += '\n'
            if text != '':
                text += '\n'
                character_index += 1
            continue
        token = split_line[0]
        start = character_index
        end = start + len(token)
        text += token + ' '
        character_index += len(token) + 1
        new_conll_string += ' '.join([token, text_base_filename, str(start), str(end)] + split_line[1:]) + '\n' 
    if len(text) != 0:
        with codecs.open(os.path.join(text_folder, '{0}.txt'.format(text_base_filename)), 'w', 'UTF-8') as f:
            f.write(text)
    conll_file.close()
    
    with codecs.open(conll_output_filepath, 'w', 'UTF-8') as f:
        f.write(new_conll_string)

def check_compatibility_between_conll_and_brat_text(conll_filepath, brat_folder):
    '''
    check if token offsets match between conll and brat .txt files. 

    conll_filepath: path to conll file
    brat_folder: folder that contains the .txt (and .ann) files that are formatted according to brat.
                                
    '''
    verbose = False
    dataset_type = utils.get_basename_without_extension(conll_filepath)
    print("Checking compatibility between CONLL and BRAT for {0} set ... ".format(dataset_type), end='')
    conll_file = codecs.open(conll_filepath, 'r', 'UTF-8')

    previous_filename = ''
    for line in conll_file:
        line = line.strip().split(' ')
        # New sentence
        if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
            continue
        
        filename = str(line[1])
        # New file
        if filename != previous_filename:
            text_filepath = os.path.join(brat_folder, '{0}.txt'.format(filename))
            with codecs.open(text_filepath, 'r', 'UTF-8') as f:
                text = f.read()
            previous_filename = filename 
            
        label = str(line[-1]).replace('_', '-') # For LOCATION-OTHER
        
        token = {}
        token['text'] = str(line[0])
        token['start'] = int(line[2])
        token['end'] = int(line[3])

        # check that the token text matches the original
        if token['text'] != text[token['start']:token['end']]:
            print("Warning: conll and brat text do not match.")
            print("\tCONLL: {0}".format(token['text']))
            print("\tBRAT : {0}".format(text[token['start']:token['end']]))
    
    print("Done.")

def output_entities(brat_output_folder, previous_filename, entities, text_filepath, text, overwrite=False):
    if previous_filename == '':
        return
    output_filepath = os.path.join(brat_output_folder, '{0}.ann'.format(previous_filename))
    if not overwrite:
        # Avoid overriding existing annotation
        if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
            raise AssertionError("The annotation already exists at: {0}".format(output_filepath))
    # Write the entities to the annotation file
    with codecs.open(output_filepath, 'w', 'utf-8') as output_file:
        for i, entity in enumerate(entities):
            output_file.write('T{0}\t{1} {2} {3}\t{4}\n'.format(i+1, entity['label'], entity['start'], entity['end'], 
                                                           utils_nlp.replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']])))
    # Copy the corresponding text file
    if text_filepath != os.path.join(brat_output_folder, os.path.basename(text_filepath)):
        shutil.copy(text_filepath, brat_output_folder)

def conll_to_brat(conll_input_filepath, conll_output_filepath, brat_original_folder, brat_output_folder, overwrite=False):
    '''
    convert conll file in conll-filepath to brat annotations and output to brat_output_folder, 
    with reference to the existing text files in brat_original_folder 
    if brat_original_folder does not exist or contain any text file, then the text files are generated from conll files,
    and conll file is updated with filenames and token offsets accordingly. 
    
    conll_input_filepath: path to conll file to convert to brat annotations
    conll_output_filepath: path to output conll file with filename and offsets that are compatible with brat annotations
    brat_original_folder: folder that contains the original .txt (and .ann) files that are formatted according to brat.
                          .txt files are used to check if the token offsets match and generate the annotation from conll.                      
    brat_output_folder: folder to output the text and brat annotations 
                        .txt files are copied from brat_original_folder to brat_output_folder
    '''
    verbose = False
    dataset_type = utils.get_basename_without_extension(conll_input_filepath)
    print("Formatting {0} set from CONLL to BRAT... ".format(dataset_type), end='')
    
    # if brat_original_folder does not exist or have any text file
    if not os.path.exists(brat_original_folder) or len(glob.glob(os.path.join(brat_original_folder, '*.txt'))) == 0:
        assert(conll_input_filepath != conll_output_filepath)
        generate_reference_text_file_for_conll(conll_input_filepath, conll_output_filepath, brat_original_folder)

    utils.create_folder_if_not_exists(brat_output_folder)
    conll_file = codecs.open(conll_output_filepath, 'r', 'UTF-8')

    previous_token_label = 'O'
    previous_filename = ''
    text_filepath = ''
    text = ''
    entity_id = 1
    entities = []
    entity = {}
    for line in conll_file:
        line = line.strip().split(' ')
        # New sentence
        if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
            # Add the last entity 
            if entity != {}:
                if verbose: print("entity: {0}".format(entity))
                entities.append(entity)
                entity_id += 1
                entity = {}
            previous_token_label = 'O'
            continue
        
        filename = str(line[1])
        # New file
        if filename != previous_filename:    
            output_entities(brat_output_folder, previous_filename, entities, text_filepath, text, overwrite=overwrite)
            text_filepath = os.path.join(brat_original_folder, '{0}.txt'.format(filename))
            with codecs.open(text_filepath, 'r', 'UTF-8') as f:
                text = f.read()
            previous_token_label = 'O'
            previous_filename = filename 
            entity_id = 1
            entities = []
            entity = {}
            
        label = str(line[-1]).replace('_', '-') # For LOCATION-OTHER
        if label == 'O':
            # Previous entity ended
            if previous_token_label != 'O':
                if verbose: print("entity: {0}".format(entity))
                entities.append(entity)
                entity_id += 1
                entity = {}
            previous_token_label = 'O'
            continue
        
        token = {}
        token['text'] = str(line[0])
        token['start'] = int(line[2])
        token['end'] = int(line[3])
        # check that the token text matches the original
        if token['text'] != text[token['start']:token['end']].replace(' ', '-'):
            print("Warning: conll and brat text do not match.")
            print("\tCONLL: {0}".format(token['text']))
            print("\tBRAT : {0}".format(text[token['start']:token['end']]))
        token['label'] = label[2:]
    
        if label[:2] == 'B-':
            if previous_token_label != 'O':
                # End the previous entity
                if verbose: print("entity: {0}".format(entity))
                entities.append(entity)
                entity_id += 1
            # Start a new entity
            entity = token
        elif label[:2] == 'I-':
            # Entity continued
            if previous_token_label == token['label']:
                # if there is no newline between the entity and the token
                if '\n' not in text[entity['end']:token['start']]:
                    # Update entity 
                    entity['text'] = entity['text'] + ' ' + token['text']
                    entity['end'] = token['end']
                else: # newline between the entity and the token
                    # End the previous entity
                    if verbose: print("entity: {0}".format(entity))
                    entities.append(entity)
                    entity_id += 1
                    # Start a new entity
                    entity = token
            elif previous_token_label != 'O':
                # TODO: count BI or II incompatibility
                # End the previous entity
                if verbose: print("entity: {0}".format(entity))
                entities.append(entity)
                entity_id += 1
                # Start new entity
                entity = token
            else: # previous_token_label == 'O'
                # TODO: count  OI incompatibility
                # Start new entity
                entity = token
        previous_token_label = token['label']
    output_entities(brat_output_folder, previous_filename, entities, text_filepath, text, overwrite=overwrite)
    conll_file.close()
    print('Done.')

def output_brat(output_filepaths, dataset_brat_folders, stats_graph_folder, overwrite=False):
    # Output brat files
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in output_filepaths.keys():
            continue
        brat_output_folder = os.path.join(stats_graph_folder, 'brat', dataset_type)
        utils.create_folder_if_not_exists(brat_output_folder)
        conll_to_brat(output_filepaths[dataset_type], output_filepaths[dataset_type], dataset_brat_folders[dataset_type], brat_output_folder, overwrite=overwrite)
