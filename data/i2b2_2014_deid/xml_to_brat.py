'''
This script convert the i2b2 2014 dataset into BRAT format.
'''
import xml.etree.ElementTree
import os
import sys
import time
import glob
import codecs
import shutil

sys.path.append(os.path.join('..','..','src'))
from conll_to_brat import output_entities
import utils


def xml_to_brat(input_folder, output_folder, overwrite=True):
    print('input_folder: {0}'.format(input_folder))
    start_time = time.time()
    if overwrite:
        shutil.rmtree(output_folder, ignore_errors=True)
    utils.create_folder_if_not_exists(output_folder)

    for input_filepath in sorted(glob.glob(os.path.join(input_folder, '*.xml'))):
        filename = utils.get_basename_without_extension(input_filepath)
        output_text_filepath = os.path.join(output_folder, '{0}.txt'.format(filename))
        xmldoc = xml.etree.ElementTree.parse(input_filepath).getroot()
        # Get text
        text = xmldoc.findtext('TEXT')
        with codecs.open(output_text_filepath, 'w', 'UTF-8') as f:
            f.write(text)

        # Get PHI tags
        tags = xmldoc.findall('TAGS')[0] # [0] because there is only one <TAGS>...</TAGS>
        entities = []
        for tag in tags:
            entity = {}
            entity['label'] = tag.get('TYPE')
            entity['text'] = tag.get('text')
            entity['start'] = int(tag.get('start'))
            entity['end'] = int(tag.get('end'))
            entities.append(entity)
        output_entities(output_folder, filename, entities, output_text_filepath, text, overwrite=overwrite)

    time_spent = time.time() - start_time
    print("Time spent formatting: {0:.2f} seconds".format(time_spent))

if __name__ == '__main__':
    print("Started formatting i2b2_2014_deid's XML files to BRAT")
    dataset_base_filename = '.'
    split = '.'
    dataset_folder_original = os.path.join('.')
    input_folder = os.path.join(dataset_folder_original, dataset_base_filename, split, 'training-PHI-Gold-Set1')
    output_base_folder = os.path.join('.')
    output_folder = os.path.join(output_base_folder, 'train')
    xml_to_brat(input_folder, output_folder)

    input_folder = os.path.join(dataset_folder_original, dataset_base_filename, split, 'training-PHI-Gold-Set2')
    output_folder = os.path.join(output_base_folder, 'valid')
    xml_to_brat(input_folder, output_folder)

    input_folder = os.path.join(dataset_folder_original, dataset_base_filename, split, 'testing-PHI-Gold-fixed')
    output_folder = os.path.join(output_base_folder, 'test')
    xml_to_brat(input_folder, output_folder)